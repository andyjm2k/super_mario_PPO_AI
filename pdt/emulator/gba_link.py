"""Local GBA link cable: two mGBA cores on one machine (SIO multiplayer)."""

from __future__ import annotations

import os
import threading
import time

from pdt.emulator.mgba_backend import MgbaBackend, _boot_core, _require_mgba, _run_one_frame
from pdt.emulator.cable_sync import _CableSync
from pdt.emulator.mgba_support import _GBA_FPS

# SIO modes mGBA's desktop link uses for a GBA-to-GBA cable.
_SIO_NORMAL_32 = 1
# Four-player multiplayer mode (Street Fighter link versus uses this).
_SIO_MULTI = 2


def _dll_path() -> str:
    """Return the mgba.dll that sits beside the Python package."""
    # Import only after the caller checked the module exists.
    import mgba
    # The Windows build copies the DLL next to the package.
    return os.path.join(os.path.dirname(mgba.__file__), "mgba.dll")


def link_symbols_available() -> bool:
    """True when this mGBA build exports the lockstep link-cable API."""
    # A missing package is not a link-capable build.
    try:
        # Raises ImportError with install help when mgba is absent.
        _require_mgba()
    except ImportError:
        # No DLL to query.
        return False
    # ctypes can see exports even when the generated cffi header omitted them.
    import ctypes
    # Load the same DLL the extension uses.
    lib = ctypes.WinDLL(_dll_path())
    # Every name the pair needs in order to plug the cable in.
    needed = (
        "GBASIOLockstepInit",
        "GBASIOLockstepNodeCreate",
        "GBASIOLockstepAttachNode",
        "mLockstepInit",
    )
    # All of them must resolve.
    return all(hasattr(lib, name) for name in needed)




def _bind_cable(ffi, lib, sync: _CableSync, lockstep):
    """Install Python callbacks into the mLockstep function table."""
    # Keep every callback alive; cffi does not own them.
    held = []

    # Ignore the C pointer; the Python sync object is the state.
    def _lock(_ptr):
        # Hold this across the SIO event so wait() can release it while sleeping.
        sync.lock()

    # Pair of the lock hook.
    def _unlock(_ptr):
        # The event handler is done with the cable.
        sync.unlock()

    # Slaves call this when they reach the transfer point.
    def _signal(_ptr, mask):
        # Record the bits and wake the master.
        return sync.signal(int(mask))

    # Master calls this and must not return False (mGBA would abort).
    def _wait(_ptr, mask):
        # Block until slaves signal, releasing the GIL while we sleep.
        return sync.wait(int(mask))

    # Cycle accounting used to keep the two GBAs on the same transfer.
    def _add(_ptr, player_id, cycles):
        # Credit the posted-cycle pool.
        sync.add_cycles(int(player_id), int(cycles))

    # Slave spends the cycles it was allowed to run.
    def _use(_ptr, player_id, cycles):
        # Return the remainder, which the C driver schedules with.
        return int(sync.use_cycles(int(player_id), int(cycles)))

    # Slave peeks without spending.
    def _unused(_ptr, player_id):
        # How far ahead or behind this GBA still is.
        return int(sync.unused_cycles(int(player_id)))

    # Called when a core drops the driver. Must not be NULL.
    def _unload(_ptr, _player_id):
        # The Python session closes the cores itself.
        return

    # Register each callback and store it on the lockstep struct.
    for name, fn, ctype in (
        ("lock", _lock, "void(*)(struct mLockstep*)"),
        ("unlock", _unlock, "void(*)(struct mLockstep*)"),
        ("signal", _signal, "_Bool(*)(struct mLockstep*, unsigned)"),
        ("wait", _wait, "_Bool(*)(struct mLockstep*, unsigned)"),
        ("addCycles", _add, "void(*)(struct mLockstep*, int, int32_t)"),
        ("useCycles", _use, "int32_t(*)(struct mLockstep*, int, int32_t)"),
        ("unusedCycles", _unused, "int32_t(*)(struct mLockstep*, int)"),
        ("unload", _unload, "void(*)(struct mLockstep*, int)"),
    ):
        # cffi callback with the C signature mGBA will call.
        cb = ffi.callback(ctype, fn)
        # Prevent garbage collection for the life of the cable.
        held.append(cb)
        # Write the function pointer into the struct.
        setattr(lockstep.d, name, cb)
    # Caller must retain ``held``.
    return held


_CDEF = """
typedef struct GBASIO GBASIO;
typedef struct mLockstep mLockstep;
typedef struct GBASIOLockstep GBASIOLockstep;
typedef struct GBASIOLockstepNode GBASIOLockstepNode;
typedef struct mTiming mTiming;
struct mTimingEvent {
    void* context;
    void (*callback)(struct mTiming*, void*, uint32_t);
    const char* name;
    uint32_t when;
    unsigned priority;
    struct mTimingEvent* next;
};
struct GBASIODriver {
    struct GBASIO* p;
    _Bool (*init)(struct GBASIODriver*);
    void (*deinit)(struct GBASIODriver*);
    _Bool (*load)(struct GBASIODriver*);
    _Bool (*unload)(struct GBASIODriver*);
    uint16_t (*writeRegister)(struct GBASIODriver*, uint32_t, uint16_t);
};
struct mLockstep {
    int attached;
    int transferActive;
    int32_t transferCycles;
    void (*lock)(struct mLockstep*);
    void (*unlock)(struct mLockstep*);
    _Bool (*signal)(struct mLockstep*, unsigned);
    _Bool (*wait)(struct mLockstep*, unsigned);
    void (*addCycles)(struct mLockstep*, int, int32_t);
    int32_t (*useCycles)(struct mLockstep*, int, int32_t);
    int32_t (*unusedCycles)(struct mLockstep*, int);
    void (*unload)(struct mLockstep*, int);
    void* context;
};
struct GBASIOLockstepNode {
    struct GBASIODriver d;
    struct GBASIOLockstep* p;
    struct mTimingEvent event;
    int32_t nextEvent;
    int32_t eventDiff;
    _Bool normalSO;
    int id;
    int mode;
    _Bool transferFinished;
};
struct GBASIOLockstep {
    struct mLockstep d;
    struct GBASIOLockstepNode* players[4];
    int attachedMulti;
    int attachedNormal;
    uint16_t multiRecv[4];
    uint32_t normalRecv[4];
};
void mLockstepInit(struct mLockstep*);
void GBASIOLockstepInit(struct GBASIOLockstep*);
void GBASIOLockstepNodeCreate(struct GBASIOLockstepNode*);
_Bool GBASIOLockstepAttachNode(struct GBASIOLockstep*, struct GBASIOLockstepNode*);
void GBASIOLockstepDetachNode(struct GBASIOLockstep*, struct GBASIOLockstepNode*);
"""


class GbaLink:
    """Two headless mGBA cores joined by a local link cable."""

    def __init__(self, rom_path: str, bios_path: str, skip_bios: bool = True, emulation_speed: float = 0):
        # The generated bindings do not declare lockstep, so check the DLL.
        if not link_symbols_available():
            # Tell the user this build cannot do link play.
            raise RuntimeError(
                "This mGBA build does not export the link-cable lockstep API."
            )
        # Second FFI in ABI mode, pointed at mgba.dll.
        from cffi import FFI
        # Declarations only; no compile step.
        self._ffi = FFI()
        # Match the Release/NDEBUG struct layout used by this DLL.
        self._ffi.cdef(_CDEF)
        # Load the DLL that exports GBASIOLockstep*.
        self._lib = self._ffi.dlopen(_dll_path())
        # Cycle/signal state shared by the two cores.
        self._sync = _CableSync(2)
        # Allocate the lockstep object the C functions fill in.
        self._lockstep = self._ffi.new("struct GBASIOLockstep*")
        # Zero attached and the function pointers first.
        self._lib.mLockstepInit(self._ffi.addressof(self._lockstep.d))
        # Clear player slots and receive buffers.
        self._lib.GBASIOLockstepInit(self._lockstep)
        # Install wait/signal so transfers do not abort().
        self._callbacks = _bind_cable(self._ffi, self._lib, self._sync, self._lockstep)
        # One node per GBA. Node 0 is the cable master.
        self._nodes = [self._ffi.new("struct GBASIOLockstepNode*") for _ in range(2)]
        # Create and attach both nodes before either core resets.
        for node in self._nodes:
            # Fill the driver function pointers inside the DLL.
            self._lib.GBASIOLockstepNodeCreate(node)
            # Give the node an id and a back-pointer to the lockstep.
            if not self._lib.GBASIOLockstepAttachNode(self._lockstep, node):
                # Four-player cap or a bad node.
                raise RuntimeError("mGBA refused to attach a link-cable node")
        # Waking a slave adds posted cycles onto that node's nextEvent.
        self._sync.bind_nodes(self._nodes)
        # Boot both ROMs without resetting so the cable is installed first.
        core_a, image_a = _boot_core(rom_path, bios_path, skip_bios, reset=False)
        # Second GBA, same ROM and BIOS.
        core_b, image_b = _boot_core(rom_path, bios_path, skip_bios, reset=False)
        # Plug the cable into both serial ports.
        self._attach_sio(core_a, self._nodes[0])
        # Player 2 is the slave (id 1).
        self._attach_sio(core_b, self._nodes[1])
        # Reset now so each core loads the driver.
        core_a.reset()
        # Slave reset after the master, matching a second GBA powering on.
        core_b.reset()
        # Public backends: buttons, frames, and memory reads.
        self.players = [
            MgbaBackend.wrap_core(core_a, image_a, emulation_speed),
            MgbaBackend.wrap_core(core_b, image_b, emulation_speed),
        ]
        # Remember speed for the pair tick.
        self._speed = float(emulation_speed)
        # Workers block here until the main thread asks for a frame.
        self._gate = threading.Condition()
        # How many frames have been requested.
        self._target = 0
        # How many frames each worker has finished.
        self._finished = [0, 0]
        # Exceptions from a worker, cleared at the start of each frame.
        self._worker_errors = []
        # Set by stop() so the workers leave.
        self._stop_workers = False
        # One thread per GBA for the life of the cable. Per-frame threads exhaust Windows.
        self._workers = [
            threading.Thread(target=self._worker, args=(index,), daemon=True)
            for index in range(2)
        ]
        # Start both before the first tick.
        for worker in self._workers:
            # They wait on the gate until tick().
            worker.start()

    def _attach_sio(self, core, node) -> None:
        """Point one core's SIO at a lockstep node (multi + 32-bit normal)."""
        # The existing cffi module knows GBASIOSetDriver and struct GBA.
        from mgba._pylib import ffi, lib
        # Address of the node, which starts with GBASIODriver.
        addr = int(self._ffi.cast("uintptr_t", node))
        # Cast into the main FFI so SetDriver accepts it.
        driver = ffi.cast("struct GBASIODriver*", addr)
        # GBA board pointer created by Core._detect.
        sio = ffi.addressof(core._native.sio)
        # Multiplayer is what link-versus games speak.
        lib.GBASIOSetDriver(sio, driver, _SIO_MULTI)
        # Normal 32-bit is also attached by desktop mGBA.
        lib.GBASIOSetDriver(sio, driver, _SIO_NORMAL_32)

    def _worker(self, index: int) -> None:
        """Run frames for one GBA whenever the main thread bumps the target."""
        # This thread lives until stop().
        while True:
            # Wait until a new frame is requested or the cable is closing.
            with self._gate:
                # Sleep while this worker is caught up.
                while self._finished[index] == self._target and not self._stop_workers:
                    # Releases the gate so the other worker and the main thread can run.
                    self._gate.wait()
                # Snapshot the frame we owe.
                target = self._target
                # stop() woke us.
                if self._stop_workers:
                    # Leave the thread.
                    return
            # One rendered frame. Cable callbacks may block until the partner arrives.
            try:
                # Draws into that player's framebuffer.
                _run_one_frame(self.players[index]._core)
            except Exception as exc:
                # The main thread raises this after the pair joins.
                self._worker_errors.append(exc)
            # Publish that this side finished the requested frame.
            with self._gate:
                # Record the generation we completed.
                self._finished[index] = target
                # Wake the main thread if it is waiting for both sides.
                self._gate.notify_all()

    def _run_frame_pair(self, poll=None) -> None:
        """Run one frame on both cores at once so the cable callbacks can rendezvous."""
        # Drop errors from the previous frame.
        self._worker_errors.clear()
        # Cable waits are allowed for this frame.
        self._sync.begin_frame()
        # Ask both workers for the next frame.
        with self._gate:
            # Next generation.
            self._target += 1
            # The number both workers must report.
            target = self._target
            # Wake both workers.
            self._gate.notify_all()
        # Link slices are many Python calls. Allow several seconds, not a hard 3s kill.
        deadline = time.monotonic() + 15.0
        # Block until both sides report this generation, pumping input between waits.
        finished = self._wait_for_target(target, deadline, poll)
        # A stuck transfer should unblock instead of freezing the window.
        if not finished:
            # Release callbacks blocked in the cable wait.
            self._sync.abort_waits()
            # Give the workers a moment to leave run_frame.
            late = time.monotonic() + 5.0
            # Wait again now that the waits were aborted.
            finished = self._wait_for_target(target, late, poll)
        # Callbacks outside tick must not block.
        self._sync.cooperative = False
        # The cores never came back.
        if not finished:
            # The window cannot safely tick again on top of a stuck core.
            raise RuntimeError("GBA link cable timed out while stepping both players")
        # Surface an emulator exception.
        if self._worker_errors:
            # The first core error is enough.
            raise self._worker_errors[0]

    def _cores_caught_up(self, target: int) -> bool:
        """True when both workers have finished this frame generation."""
        # Read under the gate so a worker cannot update half the pair.
        with self._gate:
            # Both sides must match the requested generation.
            return self._finished[0] == target and self._finished[1] == target

    def _wait_for_target(self, target: int, deadline: float, poll) -> bool:
        """Wait until both GBAs finish, calling ``poll`` so the window stays live."""
        # Until the deadline or both workers report in.
        while time.monotonic() < deadline:
            # Read keys and paint while the cores are inside this frame.
            if poll is not None:
                # The interactive tool uses this to apply buttons mid-frame.
                poll()
            # Both cores finished.
            if self._cores_caught_up(target):
                # The frame is done.
                return True
            # Sleep on the gate so we do not spin, but wake when a worker finishes.
            with self._gate:
                # Re-check under the lock. A worker may have finished while we polled.
                if self._finished[0] == target and self._finished[1] == target:
                    # Done.
                    return True
                # Remaining time before we give up on this frame.
                remaining = deadline - time.monotonic()
                # Already past the deadline.
                if remaining <= 0:
                    # Caller decides whether to abort.
                    break
                # Wake early when a worker publishes, otherwise poll again shortly.
                self._gate.wait(timeout=min(0.01, remaining))
        # One last look after the deadline.
        return self._cores_caught_up(target)

    def tick(self, count: int = 1, poll=None) -> None:
        """Advance both GBAs the same number of frames, together."""
        # At least one frame.
        frames = max(1, int(count))
        # One output frame at a time so a transfer can pause inside that frame.
        for _ in range(frames):
            # Wall time for this frame, so a slow link slice does not add extra delay.
            started = time.monotonic()
            # Keys that are already down should be visible on this frame.
            if poll is not None:
                # Pump SDL before the cores sample the keypad.
                poll()
            # Both cores run this frame concurrently.
            self._run_frame_pair(poll)
            # Optional real-time throttle.
            if self._speed > 0:
                # How long this frame is supposed to last.
                budget = 1.0 / (_GBA_FPS * float(self._speed))
                # Do not sleep if the cores already used the whole slice.
                delay = budget - (time.monotonic() - started)
                # Pump input during the sleep so a held button is not missed.
                while delay > 0 and poll is not None:
                    # Apply keys while we wait out the rest of the frame.
                    poll()
                    # Short slice so the window keeps receiving events.
                    step = min(0.005, delay)
                    # Wall-clock wait.
                    time.sleep(step)
                    # Account for the slice.
                    delay -= step
                # Training calls tick without a poll and just sleeps.
                if delay > 0 and poll is None:
                    # Pace an unattended frame.
                    time.sleep(delay)

    def stop(self) -> None:
        """Stop the frame workers, then drop both cores."""
        # Workers should leave even if they are blocked in a cable wait.
        self._stop_workers = True
        # Unblock use_cycles / add_cycles.
        self._sync.abort_waits()
        # Wake workers parked on the gate.
        with self._gate:
            # Everyone waiting on the gate should re-check stop.
            self._gate.notify_all()
        # Let each worker exit run_frame.
        for worker in self._workers:
            # Bound the wait so a stuck core cannot hang shutdown.
            worker.join(timeout=2)
        # Each backend forgets its core.
        for player in self.players:
            # Safe if already stopped.
            player.stop()
