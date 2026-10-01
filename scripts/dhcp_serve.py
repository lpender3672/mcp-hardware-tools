"""A one-client DHCP server for a point-to-point bench link.

Plug an instrument straight into this machine's Ethernet port and it still gets
its address by DHCP, exactly as on the office network — same IP, same lease
time — so a direct-cable test differs from the switched setup only in what else
is on the wire. It answers **only** the MACs it is given, so nothing else on the
port gets a lease, and it logs every exchange with a timestamp (an instrument
that dies right after a renewal shows up in the log).

This machine's port needs a fixed address first (admin PowerShell):

    Set-NetIPInterface -InterfaceAlias 'Ethernet 2' -Dhcp Disabled
    New-NetIPAddress -InterfaceAlias 'Ethernet 2' -IPAddress 192.168.64.195 -PrefixLength 24

then:

    uv run python scripts/dhcp_serve.py --server-ip 192.168.64.195 \\
        --lease 00:19:af:37:73:c2=192.168.64.7

Undo afterwards: ``Remove-NetIPAddress -InterfaceAlias 'Ethernet 2' -Confirm:$false;
Set-NetIPInterface -InterfaceAlias 'Ethernet 2' -Dhcp Enabled``.
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import struct
from dataclasses import dataclass
from datetime import datetime

SERVER_PORT = 67
CLIENT_PORT = 68
_MAGIC = b"\x63\x82\x53\x63"
_FIXED = struct.Struct("!BBBBIHH4s4s4s4s16s64s128s")  # BOOTP header, 236 bytes

DISCOVER, OFFER, REQUEST, DECLINE, ACK, NAK, RELEASE, INFORM = 1, 2, 3, 4, 5, 6, 7, 8
_NAMES = {
    1: "DISCOVER",
    2: "OFFER",
    3: "REQUEST",
    4: "DECLINE",
    5: "ACK",
    6: "NAK",
    7: "RELEASE",
    8: "INFORM",
}

OPT_MASK, OPT_ROUTER, OPT_REQUESTED_IP, OPT_LEASE = 1, 3, 50, 51
OPT_MSG_TYPE, OPT_SERVER_ID, OPT_T1, OPT_T2, OPT_END = 53, 54, 58, 59, 255


@dataclass(frozen=True)
class Packet:
    op: int
    xid: int
    flags: int
    ciaddr: str
    giaddr: str
    chaddr: bytes  # the 16-byte field, as sent
    mac: str
    msg_type: int
    requested_ip: str
    server_id: str


def _ip(raw: bytes) -> str:
    return str(ipaddress.IPv4Address(raw))


def _options(data: bytes) -> dict[int, bytes]:
    opts: dict[int, bytes] = {}
    i = 0
    while i < len(data):
        code = data[i]
        if code == OPT_END:
            break
        if code == 0:  # pad
            i += 1
            continue
        length = data[i + 1]
        opts[code] = data[i + 2 : i + 2 + length]
        i += 2 + length
    return opts


def parse(data: bytes) -> Packet | None:
    """A client DHCP message, or None for anything that is not one."""
    if len(data) < _FIXED.size + 4 or data[_FIXED.size : _FIXED.size + 4] != _MAGIC:
        return None
    op, htype, hlen, _hops, xid, _secs, flags, ciaddr, _yi, _si, giaddr, chaddr, _s, _f = (
        _FIXED.unpack_from(data)
    )
    opts = _options(data[_FIXED.size + 4 :])
    if op != 1 or htype != 1 or hlen != 6 or OPT_MSG_TYPE not in opts:
        return None
    mac = ":".join(f"{b:02x}" for b in chaddr[:6])
    req = opts.get(OPT_REQUESTED_IP, b"")
    sid = opts.get(OPT_SERVER_ID, b"")
    return Packet(
        op,
        xid,
        flags,
        _ip(ciaddr),
        _ip(giaddr),
        chaddr,
        mac,
        opts[OPT_MSG_TYPE][0],
        _ip(req) if len(req) == 4 else "",
        _ip(sid) if len(sid) == 4 else "",
    )


def _opt(code: int, value: bytes) -> bytes:
    return bytes([code, len(value)]) + value


def build_reply(
    pkt: Packet, msg_type: int, *, yiaddr: str, server_ip: str, mask: str, lease_s: int
) -> bytes:
    """An OFFER/ACK/NAK for ``pkt``. A NAK carries no address or lease."""
    header = _FIXED.pack(
        2,
        1,
        6,
        0,
        pkt.xid,
        0,
        pkt.flags,
        socket.inet_aton(pkt.ciaddr if msg_type == ACK else "0.0.0.0"),
        socket.inet_aton(yiaddr if msg_type != NAK else "0.0.0.0"),
        socket.inet_aton(server_ip if msg_type != NAK else "0.0.0.0"),
        socket.inet_aton(pkt.giaddr),
        pkt.chaddr,
        b"",
        b"",
    )
    opts = _opt(OPT_MSG_TYPE, bytes([msg_type])) + _opt(OPT_SERVER_ID, socket.inet_aton(server_ip))
    if msg_type != NAK:
        opts += (
            _opt(OPT_MASK, socket.inet_aton(mask))
            + _opt(OPT_ROUTER, socket.inet_aton(server_ip))
            + _opt(OPT_LEASE, struct.pack("!I", lease_s))
            + _opt(OPT_T1, struct.pack("!I", lease_s // 2))
            + _opt(OPT_T2, struct.pack("!I", lease_s * 7 // 8))
        )
    return header + _MAGIC + opts + bytes([OPT_END])


def decide(pkt: Packet, leases: dict[str, str], server_ip: str) -> tuple[int, str] | None:
    """``(reply type, address)`` for a client message, or None to stay silent
    (an unknown MAC, or a REQUEST meant for another server)."""
    ip = leases.get(pkt.mac)
    if ip is None:
        return None
    if pkt.msg_type == DISCOVER:
        return OFFER, ip
    if pkt.msg_type == REQUEST:
        if pkt.server_id and pkt.server_id != server_ip:
            return None  # the client chose another server's offer
        wanted = pkt.requested_ip or pkt.ciaddr
        return (ACK, ip) if wanted == ip else (NAK, ip)
    return None  # DECLINE / RELEASE / INFORM: logged, not answered


def _say(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')}  {msg}", flush=True)


def serve(
    server_ip: str,
    leases: dict[str, str],
    *,
    mask: str,
    lease_s: int,
    port: int = SERVER_PORT,
    client_port: int = CLIENT_PORT,
    broadcast: str = "255.255.255.255",
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bound to the interface address, so offers leave through this port only.
    sock.bind((server_ip, port))
    names = ", ".join(f"{m}->{ip}" for m, ip in leases.items())
    _say(f"DHCP on {server_ip}:{port}, lease {lease_s} s, serving only {names}")
    while True:
        data, _addr = sock.recvfrom(4096)
        pkt = parse(data)
        if pkt is None:
            continue
        what = _NAMES.get(pkt.msg_type, str(pkt.msg_type))
        detail = f" req {pkt.requested_ip}" if pkt.requested_ip else ""
        detail += f" ciaddr {pkt.ciaddr}" if pkt.ciaddr != "0.0.0.0" else ""
        _say(f"<- {what} from {pkt.mac}{detail}")
        verdict = decide(pkt, leases, server_ip)
        if verdict is None:
            if pkt.mac not in leases:
                _say(f"   ignored: {pkt.mac} is not ours to serve")
            continue
        reply_type, ip = verdict
        reply = build_reply(
            pkt, reply_type, yiaddr=ip, server_ip=server_ip, mask=mask, lease_s=lease_s
        )
        # A renewing client (ciaddr set) is unicast; one without an address yet
        # can't receive unicast, so broadcast.
        dest = pkt.ciaddr if pkt.ciaddr != "0.0.0.0" and reply_type != NAK else broadcast
        sock.sendto(reply, (dest, client_port))
        if reply_type == NAK:
            wanted = pkt.requested_ip or pkt.ciaddr
            _say(f"-> NAK to {pkt.mac}: asked for {wanted}, its address here is {ip}")
        else:
            _say(f"-> {_NAMES[reply_type]} {ip} to {pkt.mac} via {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server-ip", required=True, help="this machine's address on the link")
    ap.add_argument(
        "--lease",
        action="append",
        required=True,
        metavar="MAC=IP",
        help="a client to serve, e.g. 00:19:af:37:73:c2=192.168.64.7 (repeatable)",
    )
    ap.add_argument("--mask", default="255.255.255.0")
    ap.add_argument(
        "--lease-time",
        type=int,
        default=86_400,
        help="seconds (default 24 h, as the office router hands out)",
    )
    args = ap.parse_args()
    leases = {}
    for spec in args.lease:
        mac, _, ip = spec.partition("=")
        leases[mac.strip().lower().replace("-", ":")] = str(ipaddress.IPv4Address(ip.strip()))
    try:
        serve(args.server_ip, leases, mask=args.mask, lease_s=args.lease_time)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
