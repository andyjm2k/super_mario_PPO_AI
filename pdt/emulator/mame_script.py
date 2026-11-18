"""Lua boot script for the MAME training session.

Commands arrive on stdin. The screen is copied only when a step asks for it,
and that copy travels on the stdout pipe instead of a file.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# One watched word: address, then width in bytes.
Watch = Tuple[int, int]


def build_boot_script(directory: str, watches: Sequence[Watch]) -> str:
    """Return the Lua MAME runs after the machine has booted."""
    # Forward slashes so the Lua string does not treat backslashes as escapes.
    folder = directory.replace("\\", "/").rstrip("/") + "/"
    # One read expression per watched address.
    readers = []
    # Walk the pack's memory map.
    for address, width in watches:
        # 16-bit health versus an 8-bit flag.
        reader = "read_u16" if int(width) == 2 else "read_u8"
        # Decimal address, which is what the SH-2 space expects.
        readers.append(f"tostring(mem:{reader}({int(address)}))")
    # Empty watch list still produces a legal sample.
    joined = " .. ' ' .. ".join(readers) if readers else "'0'"
    # The script registers one frame callback and then returns.
    return "\n".join(
        [
            f"local dir = [[{folder}]]",
            "local mem = manager.machine.devices[':maincpu'].spaces['program']",
            "local screen = manager.machine.screens[':screen']",
            "local iop = manager.machine.ioport",
            "local held = {}",
            "local frame_no = 0",
            "local snap = false",
            "local todo = 0",
            "local function release_held()",
            "  for index = 1, #held do",
            "    local item = held[index]",
            "    local port = iop.ports[item[1]]",
            "    if port ~= nil and port.fields[item[2]] ~= nil then",
            "      port.fields[item[2]]:set_value(0)",
            "    end",
            "  end",
            "  held = {}",
            "end",
            "local primed = false",
            "local function pull_input()",
            "  if not primed then",
            "    primed = true",
            "    io.stdout:write('READY\\n')",
            "    io.stdout:flush()",
            "  end",
            "  local line = io.stdin:read('*l')",
            "  if line == nil then",
            "    todo = 1",
            "    snap = false",
            "    return",
            "  end",
            "  release_held()",
            "  snap = line:sub(1, 1) == '1'",
            "  todo = tonumber(line:match('^%d|(%d+)')) or 1",
            "  if todo < 1 then todo = 1 end",
            "  for token in line:gmatch('[^|]+') do",
            "    local port_name, field_name = token:match('^([^\\t]+)\\t(.+)$')",
            "    if port_name ~= nil and iop.ports[port_name] ~= nil and iop.ports[port_name].fields[field_name] ~= nil then",
            "      iop.ports[port_name].fields[field_name]:set_value(1)",
            "      held[#held + 1] = {port_name, field_name}",
            "    end",
            "  end",
            "end",
            "local function poll_command()",
            "  local request = io.open(dir .. 'request.txt', 'r')",
            "  if request == nil then return end",
            "  local line = request:read('*l') or ''",
            "  request:close()",
            "  pcall(os.remove, dir .. 'request.txt')",
            "  local op, path = line:match('^(%w+)\\t(.+)$')",
            "  local status = 'ok'",
            "  if op == 'save' then",
            "    local ok, err = pcall(function() manager.machine:save(path) end)",
            "    if not ok then status = tostring(err) end",
            "  elseif op == 'load' then",
            "    local ok, err = pcall(function() manager.machine:load(path) end)",
            "    if not ok then status = tostring(err) end",
            "  end",
            "  local result = io.open(dir .. 'result.txt', 'w')",
            "  if result ~= nil then",
            "    result:write(status)",
            "    result:close()",
            "  end",
            "end",
            "local function pdt_frame()",
            "  if todo <= 0 then",
            "    poll_command()",
            "    pull_input()",
            "  end",
            "  frame_no = frame_no + 1",
            "  todo = todo - 1",
            "  if todo > 0 then return end",
            "  poll_command()",
            f"  local samples = {joined}",
            "  if snap then",
            "    local src = screen:pixels()",
            "    local sw, sh, nbytes = screen.width, screen.height, 0",
            "    if src ~= nil then nbytes = #src end",
            "    io.stdout:write(string.format('PDT %d %d %d %s #%d\\n', sw, sh, frame_no, samples, nbytes))",
            "    if nbytes > 0 then io.stdout:write(src) end",
            "  else",
            "    io.stdout:write(string.format('PDT %d %d %d %s\\n', screen.width, screen.height, frame_no, samples))",
            "  end",
            "  io.stdout:flush()",
            "end",
            "emu.register_frame_done(pdt_frame, 'pdt')",
            "",
        ]
    )
