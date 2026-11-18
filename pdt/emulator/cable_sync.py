"""Lockstep wait/signal used by the local GBA link cable.

mGBA's desktop link does not block inside these callbacks. ``wait`` and
``use_cycles`` only mark a GBA asleep and return. The owner then skips that
core until ``signal`` or ``add_cycles`` wakes it. Blocking here deadlocks the
transfer, because the master posts cycles only after ``wait`` returns.
"""

from __future__ import annotations

import threading
import time

# Cycles granted when a slave is parked and the master is no longer posting.
_STALL_GRANT = 2000
# How far the slave may run ahead before the master waits. Larger is fewer Python syncs.
_SYNC_BUDGET = 32768


class _CableSync:
    """Flag-based stand-in for mGBA Qt's multiplayer lockstep."""

    def __init__(self, player_count: int):
        # How many GBAs share the cable.
        self.player_count = int(player_count)
        # Held by the core across one SIO event. Callbacks must not sleep on it.
        self._cv = threading.Condition()
        # Bits the master still needs a slave to clear.
        self.wait_mask = 0
        # 1 means that core should be stepped. 0 means it is parked.
        self.awake = [1 for _ in range(self.player_count)]
        # Cycle credit posted to each player.
        self.posted = [0 for _ in range(self.player_count)]
        # Filled by GbaLink so a wake can push the slave's next SIO event.
        self._nodes = None
        # True while both cores are inside run_frame on their own threads.
        self.cooperative = False
        # Set when a frame must stop blocking so the window cannot freeze.
        self._skip_waits = False

    def bind_nodes(self, nodes) -> None:
        """Remember the C lockstep nodes so a wake can advance ``nextEvent``."""
        # The scheduler reads these only from the thread that steps the cores.
        self._nodes = list(nodes)

    def lock(self) -> None:
        """Acquire the lockstep mutex (mLockstep.lock)."""
        # Held by the core that is inside the SIO event.
        self._cv.acquire()

    def unlock(self) -> None:
        """Release the lockstep mutex (mLockstep.unlock)."""
        # Pairs with lock() once the event handler returns.
        self._cv.release()

    def _owned(self) -> bool:
        """True when this thread already holds the condition lock."""
        # The C driver locks before the callback. Tests call in without that.
        return self._cv._is_owned()

    def _enter(self) -> bool:
        """Take the lock unless the caller already holds it. Return True if we took it."""
        # C already holds the lock during an SIO event.
        owned = self._owned()
        # A direct Python call still needs the mutex.
        if not owned:
            # Matches a unit test that does not call lock() first.
            self._cv.acquire()
        # True means this method must release before returning.
        return not owned

    def _leave(self, took: bool) -> None:
        """Release the lock when ``_enter`` took it."""
        # Leave a caller-owned lock alone.
        if took:
            # Unit-test path.
            self._cv.release()

    def wait(self, mask: int) -> bool:
        """Mark the master asleep until ``signal`` clears ``mask``.

        Returns immediately. mGBA aborts if this returns false, and it posts
        slave cycles only after this call returns.
        """
        # C already holds the lock; tests may not.
        took = self._enter()
        try:
            # Master now expects these slave bits.
            self.wait_mask |= int(mask)
            # The scheduler must not step the master again until a slave signals.
            self.awake[0] = 0
            # C treats false as a fatal cable error.
            return True
        finally:
            # Release only the lock this method took.
            self._leave(took)

    def signal(self, mask: int) -> bool:
        """Clear ``mask`` from the master's wait and wake it when the wait is done."""
        # C already holds the lock; tests may not.
        took = self._enter()
        # Whether this signal woke the master.
        woke = False
        try:
            # A slave's bit means it has reached this transfer phase.
            self.wait_mask &= ~int(mask)
            # Wake the master only when every waited bit is clear.
            if self.wait_mask == 0 and self.awake[0] < 1:
                # The master may leave the ping-pong wait.
                self.awake[0] = 1
                # Remember that a parked master was released.
                woke = True
                # Unblock add_cycles if it is waiting for this signal.
                self._cv.notify_all()
            # mGBA ignores this except as a bool.
            return woke
        finally:
            # Release only the lock this method took.
            self._leave(took)

    def _bump_next_event(self, index: int) -> None:
        """Add the slave's posted cycles onto its pending SIO event."""
        # Unit tests have no C node.
        if not self._nodes:
            # Nothing to reschedule.
            return
        # The node for this player id.
        node = self._nodes[index]
        # Qt does this so the slave jumps to the master instead of 1 cycle at a time.
        node.nextEvent = int(node.nextEvent) + int(self.posted[index])

    def _wait_for_slaves(self) -> None:
        """Block the master until slaves have spent this grant, or answered a transfer."""
        # Unit tests and single-core calls must not block.
        if not self.cooperative or self._skip_waits:
            # The caller is not running the partner on another thread.
            return
        # Deadline so a missed handshake cannot hang the process.
        deadline = time.monotonic() + 0.5
        # The lock is already held. cv.wait releases it while the slave runs.
        while True:
            # Shutdown or a frame abort asked every waiter to return.
            if self._skip_waits:
                # Leave the callback so run_frame can finish.
                return
            # A transfer is in progress. Stay until the slave signals.
            if self.awake[0] < 1:
                # Slaves parked with no credit cannot reach the signal.
                stalled = True
                # Check every slave.
                for index in range(1, self.player_count):
                    # Still running or still holding credit.
                    if self.awake[index] >= 1 or self.posted[index] > 0:
                        # Not stalled.
                        stalled = False
                # Give them a slice so they can signal.
                if stalled:
                    # Same size as mGBA's idle lockstep step.
                    for index in range(1, self.player_count):
                        # Credit and reschedule.
                        self.posted[index] += _STALL_GRANT
                        # Jump the pending event to the new credit.
                        self._bump_next_event(index)
                        # Mark runnable before waking the thread.
                        self.awake[index] = 1
                    # Unblock use_cycles.
                    self._cv.notify_all()
                # Give up rather than deadlock.
                if time.monotonic() >= deadline:
                    # Drop the wait so the frame can finish.
                    self.wait_mask = 0
                    # Master runs again.
                    self.awake[0] = 1
                    # Leave.
                    return
                # Slave runs while this releases the lock.
                self._cv.wait(min(0.05, max(0.0, deadline - time.monotonic())))
                # Look again.
                continue
            # Idle slice: wait until every slave has spent what we just posted.
            if all(self.posted[index] <= 0 for index in range(1, self.player_count)):
                # Caught up.
                return
            # Give up if the slave never spends.
            if time.monotonic() >= deadline:
                # Leave the posted credit for the next event.
                return
            # Slave use_cycles notifies when the balance is spent.
            self._cv.wait(min(0.05, max(0.0, deadline - time.monotonic())))

    def add_cycles(self, player_id: int, cycles: int) -> None:
        """Credit ``cycles`` to a player (the master credits every slave)."""
        # Ignore negative posts. The C side aborts on those.
        if int(cycles) < 0:
            # Nothing to add.
            return
        # C already holds the lock; tests may not.
        took = self._enter()
        try:
            # Master id 0 posts the same credit to each other GBA.
            if int(player_id) == 0:
                # Slaves are 1..n-1.
                for index in range(1, self.player_count):
                    # Accumulate until use_cycles spends it.
                    self.posted[index] += int(cycles)
                    # A parked slave has to be told how far it may run.
                    if self.awake[index] < 1:
                        # Push the pending event out by the full posted pool.
                        self._bump_next_event(index)
                        # The scheduler may step this slave again.
                        self.awake[index] = 1
                # The slave may be blocked in use_cycles.
                self._cv.notify_all()
                # A real transfer has to rendezvous. Idle play can share a bigger slice.
                needs_rendezvous = self.awake[0] < 1 or any(
                    self.posted[index] > _SYNC_BUDGET for index in range(1, self.player_count)
                )
                # Waiting on every 2000-cycle post made Game Link sit on Please wait.
                if needs_rendezvous:
                    # Block until the slave spends the slice or answers the transfer.
                    self._wait_for_slaves()
            else:
                # A slave posting its own slack.
                self.posted[int(player_id)] += int(cycles)
        finally:
            # Release only the lock this method took.
            self._leave(took)

    def use_cycles(self, player_id: int, spent: int) -> int:
        """Spend posted cycles and park the slave when it runs ahead."""
        # C already holds the lock; tests may not.
        took = self._enter()
        try:
            # Which GBA is spending.
            index = int(player_id)
            # The C driver passes how far this GBA has run.
            self.posted[index] -= int(spent)
            # Slaves that pass the master must wait. The master is parked by wait().
            if index != 0 and self.posted[index] <= 0:
                # Park until the master posts another slice.
                self.awake[index] = 0
                # The master ping-pong is waiting for this spend.
                self._cv.notify_all()
                # Only the live pair blocks. Unit tests must return immediately.
                if self.cooperative:
                    # Deadline so a missing master cannot hang the process.
                    deadline = time.monotonic() + 0.5
                    # Stay here until add_cycles posts a positive balance.
                    while self.posted[index] <= 0 and not self._skip_waits:
                        # Remaining budget.
                        remaining = deadline - time.monotonic()
                        # Give the core one cycle and continue.
                        if remaining <= 0:
                            # Avoid a 0 return, which reschedules the same wait.
                            self.posted[index] = 1
                            # Leave the loop.
                            break
                        # Releases this lock so the master callback can run.
                        self._cv.wait(remaining)
                    # Runnable again once credit is available.
                    self.awake[index] = 1
            else:
                # The master may be waiting for this spend.
                self._cv.notify_all()
            # The balance mGBA uses to schedule the next SIO event.
            return int(self.posted[index])
        finally:
            # Release only the lock this method took.
            self._leave(took)

    def unused_cycles(self, player_id: int) -> int:
        """Return posted cycles that this player has not spent."""
        # C already holds the lock; tests may not.
        took = self._enter()
        try:
            # Read-only view used by the slave update.
            return int(self.posted[int(player_id)])
        finally:
            # Release only the lock this method took.
            self._leave(took)

    def abort_waits(self) -> None:
        """Unblock both cores so a slow frame or shutdown can return."""
        # C may be inside a wait on the other thread. Take the lock ourselves.
        took = self._enter()
        try:
            # Further waits in this frame return immediately.
            self._skip_waits = True
            # Nobody stays parked.
            self.awake = [1 for _ in range(self.player_count)]
            # Drop a transfer wait so the master can leave add_cycles.
            self.wait_mask = 0
            # A slave blocked on an empty pool needs one cycle to return.
            for index in range(self.player_count):
                # Only lift a non-positive balance.
                if self.posted[index] <= 0:
                    # One cycle is enough to leave use_cycles.
                    self.posted[index] = 1
            # Wake every callback blocked in cv.wait.
            self._cv.notify_all()
        finally:
            # Release only the lock this method took.
            self._leave(took)

    def begin_frame(self) -> None:
        """Allow cable waits again for a new pair of frames."""
        # The previous frame may have aborted its waits.
        self._skip_waits = False
        # Both worker threads are inside run_frame for this frame.
        self.cooperative = True

    def wake_stalled_slave(self) -> bool:
        """Wake a parked slave that will never be posted more cycles."""
        # C is not inside an event here. Take the lock ourselves.
        took = self._enter()
        # Whether any slave was released.
        woke = False
        try:
            # Masters are woken only by signal().
            for index in range(1, self.player_count):
                # This slave is already runnable.
                if self.awake[index] >= 1:
                    # Leave it.
                    continue
                # No credit left, so grant a slice and reschedule its SIO event.
                if self.posted[index] <= 0:
                    # Same size as mGBA's idle lockstep step.
                    self.posted[index] += _STALL_GRANT
                    # Make the pending event consume that grant.
                    self._bump_next_event(index)
                # The scheduler can step it on the next pass.
                self.awake[index] = 1
                # Remember that someone was released.
                woke = True
            # True when a slave can run again.
            return woke
        finally:
            # Release only the lock this method took.
            self._leave(took)
