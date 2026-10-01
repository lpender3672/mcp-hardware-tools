"""Passively watch bench instruments for dropping off the LAN.

ICMP ping only — no TCP, no SCPI, no VISA — so the watcher itself cannot be what
knocks an instrument over. Each host is pinged every ``--interval`` seconds; every
up/down transition is logged with a timestamp, together with the host's entry in
this machine's neighbour (ARP) table, which the pings keep re-probing:

* ping dead, neighbour Reachable            -> still answers ARP: link up, IP stack wedged
* ping dead, neighbour Unreachable/Failed   -> not even ARP: port/link down, or powered off
* MAC changed                               -> a different device now holds the IP (DHCP)

Run it and leave the bench alone:

    uv run python scripts/lan_watch.py 192.168.64.7=scope 192.168.64.49=siggen

Every sample goes to a CSV (``--log``, default ``lan_watch.csv``); the console shows
transitions plus a heartbeat every ``--summary`` seconds. Ctrl+C prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

_WINDOWS = sys.platform == "win32"
_RTT = re.compile(r"time[=<]\s*([\d.]+)\s*ms", re.IGNORECASE)
_MAC = re.compile(r"([0-9a-f]{2}(?:[-:][0-9a-f]{2}){5})", re.IGNORECASE)


def ping(host: str, timeout_s: float) -> float | None:
    """Round-trip time in ms, or None when no echo reply came back.

    Windows ``ping`` exits 0 even for "Destination host unreachable" (that reply
    comes from this machine), so success is judged by a TTL in the output.
    """
    if _WINDOWS:
        cmd = ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), host]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, round(timeout_s))), host]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    if "ttl=" not in out.lower():
        return None
    m = _RTT.search(out)
    return float(m.group(1)) if m else 0.0


def neighbour(host: str) -> tuple[str, str]:
    """``(mac, state)`` from this machine's neighbour (ARP) table; ``("", "")``
    when absent. Reads the table only — the pings themselves drive its probing.
    State is the OS's word for it (Reachable, Stale, Unreachable, Incomplete…)."""
    if _WINDOWS:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-NetNeighbor -IPAddress {host} -AddressFamily IPv4 "
            "-ErrorAction SilentlyContinue | Select-Object -First 1 "
            "LinkLayerAddress,State | ConvertTo-Csv -NoTypeInformation",
        ]
    else:
        cmd = ["ip", "neigh", "show", host]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "", ""
    if _WINDOWS:
        rows = list(csv.reader(out.splitlines()))
        if len(rows) < 2 or len(rows[1]) < 2:
            return "", ""
        raw_mac, state = rows[1][0], rows[1][1]
    else:
        line = out.strip().splitlines()[0] if out.strip() else ""
        raw_mac, state = line, (line.split()[-1] if line else "")
    m = _MAC.search(raw_mac)
    mac = m.group(1).lower().replace("-", ":") if m else ""
    if mac in ("00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"):
        mac = ""
    return mac, state.capitalize()


def _link_verdict(state: str, *, first: bool = False) -> str:
    s = state.lower()
    if s in ("reachable", "permanent"):
        if first:
            # Confirmed by the last good ping, not by this failure: wait for the
            # OS to re-probe it (Reachable -> Stale -> Probe -> ...).
            return "ARP entry still fresh from the last reply -> verdict once it is re-probed"
        return "still answers ARP -> link up, IP stack not answering"
    if s in ("unreachable", "incomplete", "failed"):
        return "not answering ARP either -> port/link down or powered off"
    return f"ARP state {state or 'absent'} -> not yet conclusive"


@dataclass
class Host:
    address: str
    name: str
    up: bool | None = None
    since: float = field(default_factory=time.monotonic)
    mac: str = ""
    arp: str = ""
    sent: int = 0
    lost: int = 0
    outages: int = 0
    rtts: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.address})" if self.name != self.address else self.address


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def _say(msg: str) -> None:
    print(f"{_now()}  {msg}", flush=True)


def _parse_hosts(specs: list[str]) -> list[Host]:
    hosts = []
    for spec in specs:
        address, _, name = spec.partition("=")
        hosts.append(Host(address.strip(), name.strip() or address.strip()))
    return hosts


def watch(
    hosts: list[Host], *, interval_s: float, timeout_s: float, summary_s: float, log_path: str
) -> None:
    started = time.monotonic()
    next_summary = started + summary_s
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(["time", "host", "name", "up", "rtt_ms", "mac", "arp_state"])
        _say(
            f"watching {', '.join(h.label for h in hosts)} every {interval_s:g} s "
            f"(ICMP only) -> {log_path}"
        )
        while True:
            tick = time.monotonic()
            for h in hosts:
                rtt = ping(h.address, timeout_s)
                up = rtt is not None
                mac, arp = neighbour(h.address)
                h.sent += 1
                if up:
                    h.rtts.append(rtt)
                else:
                    h.lost += 1
                writer.writerow(
                    [
                        _now(),
                        h.address,
                        h.name,
                        int(up),
                        "" if rtt is None else f"{rtt:.1f}",
                        mac,
                        arp,
                    ]
                )
                if h.up is None:
                    _say(
                        f"{h.label}: {'UP' if up else 'DOWN'} at start"
                        f"  mac {mac or '?'} arp {arp or '-'}"
                    )
                elif up != h.up:
                    held = _duration(time.monotonic() - h.since)
                    if up:
                        _say(f"{h.label}: back UP after {held} down  mac {mac or '?'}")
                    else:
                        h.outages += 1
                        _say(f"{h.label}: DOWN after {held} up; {_link_verdict(arp, first=True)}")
                elif not up and arp != h.arp:
                    # While down, the ARP state settles (Stale -> Probe -> Unreachable).
                    _say(f"{h.label}: still DOWN; {_link_verdict(arp)}")
                if h.mac and mac and mac != h.mac:
                    _say(f"{h.label}: MAC changed {h.mac} -> {mac} (another device has the IP?)")
                if up != h.up:
                    h.since = time.monotonic()
                h.up = up
                h.mac = mac or h.mac
                h.arp = arp
            fh.flush()
            if time.monotonic() >= next_summary:
                next_summary += summary_s
                _say("heartbeat: " + "; ".join(_state(h) for h in hosts))
            time.sleep(max(0.0, interval_s - (time.monotonic() - tick)))


def _state(h: Host) -> str:
    recent = h.rtts[-20:]
    rtt = f", rtt ~{sum(recent) / len(recent):.1f} ms" if recent else ""
    state = "UP" if h.up else "DOWN"
    return f"{h.name} {state} for {_duration(time.monotonic() - h.since)}{rtt}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hosts", nargs="+", help="ADDRESS or ADDRESS=name, e.g. 192.168.64.7=scope")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between rounds")
    ap.add_argument("--timeout", type=float, default=1.0, help="ping timeout (s)")
    ap.add_argument("--summary", type=float, default=600.0, help="heartbeat period (s)")
    ap.add_argument("--log", default="lan_watch.csv", help="CSV of every sample (appended)")
    args = ap.parse_args()
    hosts = _parse_hosts(args.hosts)
    try:
        watch(
            hosts,
            interval_s=args.interval,
            timeout_s=args.timeout,
            summary_s=args.summary,
            log_path=args.log,
        )
    except KeyboardInterrupt:
        print()
        for h in hosts:
            loss = 100.0 * h.lost / h.sent if h.sent else 0.0
            _say(
                f"{h.label}: {h.sent} pings, {loss:.1f}% lost, {h.outages} outage(s); " + _state(h)
            )


if __name__ == "__main__":
    main()
