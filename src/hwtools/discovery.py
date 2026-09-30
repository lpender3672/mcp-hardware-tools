"""Find SCPI instruments on the local network.

DHCP moves bench instruments around, so an address is a cache, not an identity:
the stable key is the serial number an instrument reports in ``*IDN?``. This
module turns "which instruments are out there, and where are they now" into a
list of :class:`FoundInstrument`, in two stages:

1. **VXI-11 broadcast** — the LXI-standard discovery path. pyvisa-py sends a
   portmapper broadcast on every interface (per-interface broadcast addresses need
   ``psutil``) and returns ``TCPIP::<host>::INSTR`` for each responder. Sub-second.
2. **Port sweep** (fallback) — some switches, VLANs and Windows firewalls drop the
   broadcast. When it finds nothing, every host of each local /24 is tried on the
   raw-SCPI ports (Rigol 5555, the common 5025) with a short connect timeout.

Every hit is then asked ``*IDN?`` (raw socket first, VXI-11 second) and parsed.
:func:`probe` does the same for one address a user typed, and
:func:`open_instrument` builds the matching driver for a found instrument.
"""

from __future__ import annotations

import concurrent.futures as cf
import ipaddress
import socket
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hwtools.drivers.rigol import DG1062, DS1054Z
from hwtools.transport.raw_tcp import RIGOL_RAW_PORT

RAW_SCPI_PORTS: tuple[int, ...] = (RIGOL_RAW_PORT, 5025)

# A sweep of anything wider than a /24 is thousands of connects on a shared network;
# wider interfaces are narrowed to the /24 around the host's own address.
_MAX_SWEEP_PREFIX = 24
_SWEEP_CONNECT_TIMEOUT_S = 0.3
_SWEEP_WORKERS = 128


@dataclass(frozen=True)
class FoundInstrument:
    """An instrument that answered ``*IDN?``.

    ``raw_port`` is the raw-SCPI port it answered on, or ``None`` when it only
    answered over VXI-11. ``resource`` is always a usable VISA resource string.
    """

    host: str
    resource: str
    manufacturer: str
    model: str
    serial: str
    firmware: str
    raw_port: int | None = None

    @property
    def idn(self) -> str:
        return ",".join((self.manufacturer, self.model, self.serial, self.firmware))


def parse_idn(reply: str) -> tuple[str, str, str, str] | None:
    """Split an IEEE 488.2 ``*IDN?`` reply into (manufacturer, model, serial, firmware).

    Returns ``None`` for anything that is not four comma-separated fields — a
    service on 5025 that is not an instrument, a half-read reply.
    """
    parts = [p.strip() for p in reply.strip().split(",")]
    if len(parts) < 4 or not parts[1]:
        return None
    manufacturer, model, serial = parts[0], parts[1], parts[2]
    firmware = ",".join(parts[3:])
    return manufacturer, model, serial, firmware


def _idn_raw(host: str, port: int, timeout_s: float) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as s:
            s.settimeout(timeout_s)
            s.sendall(b"*IDN?\n")
            out = bytearray()
            while not out.endswith(b"\n") and len(out) < 512:
                chunk = s.recv(256)
                if not chunk:
                    break
                out += chunk
    except OSError:
        return None
    text = out.decode(errors="replace").strip()
    return text or None


def _idn_vxi11(resource: str, timeout_s: float) -> str | None:
    try:
        import pyvisa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Only the probe's own resource is closed: the manager's session is
            # shared, and closing it would drop every live instrument link.
            rm = pyvisa.ResourceManager("@py")
            inst = rm.open_resource(resource, open_timeout=int(timeout_s * 1000))
            try:
                inst.timeout = int(timeout_s * 1000)
                reply: str = inst.query("*IDN?")  # type: ignore[attr-defined]
                return reply.strip() or None
            finally:
                inst.close()
    except Exception:
        return None


def _host_of(resource: str) -> str | None:
    """``TCPIP0::192.168.1.5::INSTR`` → ``192.168.1.5``."""
    parts = resource.split("::")
    if len(parts) >= 2 and parts[0].upper().startswith("TCPIP"):
        return parts[1]
    return None


def probe(
    address: str,
    *,
    timeout_s: float = 1.5,
    ports: Sequence[int] = RAW_SCPI_PORTS,
) -> FoundInstrument | None:
    """Identify the instrument at ``address``, or ``None`` if nothing answers.

    ``address`` may be a host/IP (``192.168.1.5``), ``host:port`` (raw SCPI on
    that port only) or a VISA resource (``TCPIP0::192.168.1.5::INSTR``).
    """
    address = address.strip()
    if "::" in address:
        host = _host_of(address)
        if host is None:
            return None
        resource, raw_ports = address, list(ports)
    elif address.count(":") == 1:
        host, _, port_s = address.partition(":")
        resource, raw_ports = f"TCPIP0::{host}::INSTR", [int(port_s)]
    else:
        host = address
        resource, raw_ports = f"TCPIP0::{host}::INSTR", list(ports)

    for port in raw_ports:
        reply = _idn_raw(host, port, timeout_s)
        fields = parse_idn(reply) if reply else None
        if fields:
            return FoundInstrument(host, resource, *fields, raw_port=port)
    reply = _idn_vxi11(resource, timeout_s)
    fields = parse_idn(reply) if reply else None
    if fields:
        return FoundInstrument(host, resource, *fields, raw_port=None)
    return None


def local_networks() -> list[ipaddress.IPv4Network]:
    """The IPv4 networks of this machine's up, non-loopback interfaces.

    Uses ``psutil`` when installed (every interface, true netmask); otherwise the
    /24 around the default route's source address.
    """
    nets: list[ipaddress.IPv4Network] = []
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family != socket.AF_INET or not a.netmask:
                    continue
                ip = ipaddress.IPv4Address(a.address)
                if ip.is_loopback or ip.is_link_local:
                    continue
                net = ipaddress.IPv4Network(f"{a.address}/{a.netmask}", strict=False)
                if net.prefixlen < _MAX_SWEEP_PREFIX:
                    # Narrow a wide interface to the /24 around this host's own address.
                    net = ipaddress.IPv4Network(f"{a.address}/{_MAX_SWEEP_PREFIX}", strict=False)
                nets.append(net)
    except ImportError:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # No packet is sent: connect() on UDP only picks the outbound interface.
                s.connect(("10.255.255.255", 1))
                own = s.getsockname()[0]
            nets.append(ipaddress.IPv4Network(f"{own}/24", strict=False))
        except OSError:
            pass
    return list(dict.fromkeys(nets))


def _sweep_hosts(networks: Iterable[ipaddress.IPv4Network]) -> list[str]:
    hosts: list[str] = []
    for net in networks:
        if net.prefixlen < _MAX_SWEEP_PREFIX:
            raise ValueError(f"refusing to sweep {net}: wider than /{_MAX_SWEEP_PREFIX}")
        hosts.extend(str(h) for h in net.hosts())
    return list(dict.fromkeys(hosts))


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=_SWEEP_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def _broadcast_hosts(wait_s: float) -> list[str]:
    try:
        import pyvisa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Never closed here: the manager's session is shared with every live link.
            resources = pyvisa.ResourceManager("@py").list_resources("TCPIP?*::INSTR")
    except Exception:
        return []
    del wait_s  # pyvisa-py's VXI-11 broadcast uses its own fixed wait
    return [h for r in resources if (h := _host_of(r))]


def discover_instruments(
    timeout_s: float = 1.5,
    *,
    sweep_fallback: bool = True,
    networks: Sequence[ipaddress.IPv4Network] | None = None,
    ports: Sequence[int] = RAW_SCPI_PORTS,
) -> list[FoundInstrument]:
    """Every SCPI instrument reachable on the local networks, deduplicated by serial.

    ``networks`` overrides the swept networks (default: :func:`local_networks`);
    ``sweep_fallback=False`` restricts discovery to the VXI-11 broadcast.
    """
    hosts = _broadcast_hosts(timeout_s)
    if not hosts and sweep_fallback:
        candidates = _sweep_hosts(networks if networks is not None else local_networks())
        with cf.ThreadPoolExecutor(_SWEEP_WORKERS) as ex:
            futs = {ex.submit(_port_open, h, p): h for h in candidates for p in ports}
            hosts = list(dict.fromkeys(futs[f] for f in cf.as_completed(futs) if f.result()))

    found: dict[str, FoundInstrument] = {}
    if hosts:
        with cf.ThreadPoolExecutor(min(len(hosts), 32)) as ex:
            for inst in ex.map(lambda h: probe(h, timeout_s=timeout_s, ports=ports), hosts):
                if inst is not None:
                    found.setdefault(inst.serial or inst.host, inst)
    return sorted(found.values(), key=lambda i: (i.model, i.serial, i.host))


def open_instrument(found: FoundInstrument) -> DS1054Z | DG1062:
    """Build (but do not connect) the driver matching ``found``'s model.

    Raw SCPI is used when the instrument answered on a raw port — markedly faster
    for scope waveform pulls — otherwise its VISA resource. Raises ``ValueError``
    for a model this package has no driver for.
    """
    model = found.model.upper()
    if model.startswith(("DS1", "MSO1")):
        if found.raw_port is not None:
            return DS1054Z.over_tcp(found.host, found.raw_port)
        return DS1054Z.over_visa(found.resource)
    if model.startswith("DG1"):
        if found.raw_port is not None:
            return DG1062.over_tcp(found.host, found.raw_port)
        return DG1062.over_visa(found.resource)
    raise ValueError(f"no driver for {found.manufacturer} {found.model}")
