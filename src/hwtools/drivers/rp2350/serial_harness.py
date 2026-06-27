"""Host-side driver for the RP2350 harness firmware over its USB-CDC channel.

Speaks the firmware's ASCII line protocol (see
``firmware/common/src/protocol.rs``): one command per line, one reply line back.
This realises the :class:`DigitalDUT` contract, so the harness can be driven from
tests and (later) MCP harness tools, and reflashed without pressing BOOTSEL.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hwtools.interfaces.digital_dut import DigitalDUT

# USB VID:PID the harness firmware enumerates under (pid.codes test id; see
# the UsbVidPid in firmware/targets/rp2350/src/main.rs).
HARNESS_USB_VID = 0x16C0
HARNESS_USB_PID = 0x27DD


@runtime_checkable
class SerialLike(Protocol):
    """The slice of ``serial.Serial`` this driver needs (so it can be faked)."""

    def write(self, data: bytes) -> int | None: ...
    def readline(self) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def find_pico_port() -> str | None:
    """Return the serial port of the first attached harness device, if any."""
    from serial.tools import list_ports

    for port in list_ports.comports():
        if port.vid == HARNESS_USB_VID and port.pid == HARNESS_USB_PID:
            return str(port.device)
    return None


class SerialHarness(DigitalDUT):
    """Drives the RP2350 harness firmware over a serial command channel."""

    def __init__(self, port: str, *, baud: int = 115200, timeout_s: float = 1.0) -> None:
        self._port = port
        self._baud = baud
        self._timeout_s = timeout_s
        self._serial: SerialLike | None = None

    @classmethod
    def from_serial(cls, serial: SerialLike) -> SerialHarness:
        """Build a harness around an already-open serial-like object (for tests)."""
        harness = cls(port="<injected>")
        harness._serial = serial
        return harness

    def open(self) -> None:
        if self._serial is None:
            import serial as pyserial

            self._serial = pyserial.Serial(self._port, self._baud, timeout=self._timeout_s)
        self._serial.reset_input_buffer()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    @property
    def _link(self) -> SerialLike:
        if self._serial is None:
            raise RuntimeError("harness is not open; call open() first")
        return self._serial

    def _command(self, line: str) -> str:
        """Send a command line and return the firmware's reply (stripped)."""
        self._link.write(line.encode("ascii") + b"\n")
        return self._link.readline().decode("ascii", errors="replace").strip()

    def idn(self) -> str:
        return self._command("ID?")

    def start_uart_stream(self, value: int, *, baud: int) -> None:
        if not 0 <= value <= 0xFF:
            raise ValueError("value must be a single byte (0..255)")
        reply = self._command(f"EMIT UART {value:02X} {baud}")
        if reply != "OK":
            raise RuntimeError(f"EMIT rejected: {reply!r}")

    def start_square(self, freq_hz: int, *, duty_pct: int = 50) -> None:
        if freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
        if not 0 <= duty_pct <= 100:
            raise ValueError("duty_pct must be 0..100")
        reply = self._command(f"EMIT SQUARE {freq_hz} {duty_pct}")
        if reply != "OK":
            raise RuntimeError(f"EMIT SQUARE rejected: {reply!r}")

    def stop(self) -> None:
        reply = self._command("STOP")
        if reply != "OK":
            raise RuntimeError(f"STOP rejected: {reply!r}")

    def reboot_to_bootloader(self) -> None:
        # The device resets after acknowledging, so don't wait for more traffic.
        self._link.write(b"BOOTSEL\n")
