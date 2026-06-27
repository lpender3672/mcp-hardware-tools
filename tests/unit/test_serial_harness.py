"""Host harness driver: command formatting and reply handling, no hardware."""

from __future__ import annotations

import pytest

from hwtools.drivers.rp2350.serial_harness import SerialHarness


class FakeSerial:
    """Records written commands and serves canned reply lines."""

    def __init__(self, replies: list[str]) -> None:
        self.written: list[bytes] = []
        self._replies = [r.encode() + b"\r\n" for r in replies]

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def readline(self) -> bytes:
        return self._replies.pop(0) if self._replies else b""

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def commands(self) -> list[str]:
        return [w.decode().strip() for w in self.written]


def test_idn_round_trips() -> None:
    fake = FakeSerial(["hwtools-harness rp2350 v0"])
    harness = SerialHarness.from_serial(fake)
    with harness:
        assert harness.idn() == "hwtools-harness rp2350 v0"
    assert fake.commands == ["ID?"]


def test_start_uart_stream_formats_command() -> None:
    fake = FakeSerial(["OK"])
    harness = SerialHarness.from_serial(fake)
    harness.open()
    harness.start_uart_stream(0xA5, baud=9600)
    assert fake.commands == ["EMIT UART A5 9600"]


def test_emit_rejection_raises() -> None:
    fake = FakeSerial(["ERR"])
    harness = SerialHarness.from_serial(fake)
    harness.open()
    with pytest.raises(RuntimeError):
        harness.start_uart_stream(0x00, baud=9600)


def test_stop_and_bootloader_commands() -> None:
    fake = FakeSerial(["OK"])
    harness = SerialHarness.from_serial(fake)
    harness.open()
    harness.stop()
    harness.reboot_to_bootloader()
    assert fake.commands == ["STOP", "BOOTSEL"]


def test_value_out_of_range_rejected() -> None:
    harness = SerialHarness.from_serial(FakeSerial([]))
    harness.open()
    with pytest.raises(ValueError):
        harness.start_uart_stream(256, baud=9600)
