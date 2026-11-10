"""CAN adapter discovery and role assignment.

The arms are reached over two USB-CAN adapters that the kernel names ``can0``
and ``can1`` in whatever order it enumerates them, while the teleop stack looks
for ``can_right`` and ``can_left``. Which physical adapter gets which name is
therefore a decision someone has to make after every replug, and until now it
was made by hand over SSH with ``ethtool``/``can_activate.sh``.

Two things make that awkward to automate, and both are handled here:

**Identity.** A kernel name is not an identity -- ``can0`` may be the right arm
today and the left one tomorrow. The stable identifier is the USB interface path
(``3-2:1.0``): it names the *port the cable is plugged into*, so it survives
reboots and does not depend on enumeration order. It is read straight from
sysfs rather than from ``ethtool``, which means discovery needs no privileges.

**Swapping.** ``ip link set X name can_right`` fails with EEXIST while another
interface still holds that name, so exchanging the two roles cannot be done one
interface at a time. Any name that is in the way is parked on a temporary name
first, which is what :func:`plan` works out.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYS_NET = Path("/sys/class/net")

# Kernel interface names are short and we hand them to ip(8) as root, so accept
# only an obviously safe shape rather than trying to escape anything.
NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,14}$")
BUS_RE = re.compile(r"^[0-9]+-[0-9.]+:[0-9]+\.[0-9]+$")

# Installed by ops/install.sh, root-owned, and the only command the dashboard is
# allowed to run under sudo. Keeping it outside the repo matters: a helper the
# dashboard user could edit would turn one NOPASSWD entry into a root shell.
HELPER = "/usr/local/sbin/dt01-can-apply"


class CanError(RuntimeError):
    """Raised for anything the operator needs to see and act on."""


@dataclass
class Adapter:
    """One CAN interface, identified by the USB port it is plugged into."""

    iface: str
    bus: str = ""
    state: str = "DOWN"
    bitrate: int | None = None
    restart_ms: int | None = None
    driver: str = ""
    product: str = ""
    serial: str = ""
    role: str | None = None
    bus_kind: str = ""
    errors: dict[str, Any] = field(default_factory=dict)

    @property
    def assignable(self) -> bool:
        """Only USB adapters can be reassigned by unplugging and replugging.

        This host also exposes the Jetson's on-board MTTCAN controller and an
        SPI one. They are real CAN interfaces but they are soldered down, so a
        role assignment for them would be meaningless -- and their "bus path"
        (``c310000.mttcan``) is not a USB port at all.
        """
        return self.bus_kind == "usb" and bool(BUS_RE.match(self.bus))

    @property
    def usb_port(self) -> str:
        """The port alone (``3-2``), without the USB interface suffix."""
        return self.bus.split(":")[0]

    def as_dict(self) -> dict[str, Any]:
        """JSON shape consumed by the dashboard."""
        return {
            "iface": self.iface,
            "bus": self.bus,
            "usb_port": self.usb_port,
            "state": self.state,
            "bitrate": self.bitrate,
            "restart_ms": self.restart_ms,
            "driver": self.driver,
            "product": self.product,
            "serial": self.serial,
            "role": self.role,
            "bus_kind": self.bus_kind,
            "assignable": self.assignable,
            "errors": self.errors,
        }


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _bus_of(iface: str) -> str:
    """USB interface path for ``iface``, e.g. ``3-2:1.0``.

    ``/sys/class/net/<iface>/device`` links to the USB interface directory, and
    that directory's name is exactly what ``ethtool -i`` reports as ``bus-info``
    -- but readable without root, and without ethtool being installed.
    """
    try:
        return (SYS_NET / iface / "device").resolve().name
    except OSError:
        return ""


def _usb_meta(iface: str) -> tuple[str, str, str]:
    """``(driver, product, serial)`` for the adapter behind ``iface``."""
    dev = SYS_NET / iface / "device"
    driver = ""
    try:
        driver = (dev / "driver").resolve().name
    except OSError:
        pass
    # The USB *device* is the parent of the USB *interface* directory; that is
    # where the human-readable strings live.
    parent = dev.resolve().parent
    product = _read(parent / "product")
    serial = _read(parent / "serial")
    if not product:
        vid, pid = _read(parent / "idVendor"), _read(parent / "idProduct")
        if vid and pid:
            product = f"{vid}:{pid}"
    return driver, product, serial


def _ip_json(args: list[str]) -> list[dict[str, Any]]:
    """Run ``ip -json`` and return the parsed array, or [] if unsupported."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["ip", "-details", "-json", *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
        data = json.loads(out.stdout)
        return data if isinstance(data, list) else []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def discover(roles: dict[str, str] | None = None) -> list[Adapter]:
    """Every CAN interface on the host, newest kernel state included.

    ``roles`` maps a role name to the interface name the stack expects for it
    (``{"right": "can_right"}``), so the UI can say which adapter is currently
    filling which role.
    """
    by_name = {r: name for r, name in (roles or {}).items()}
    name_to_role = {name: role for role, name in by_name.items()}

    adapters: list[Adapter] = []
    for entry in _ip_json(["link", "show", "type", "can"]):
        iface = str(entry.get("ifname") or "")
        if not iface:
            continue
        info = (entry.get("linkinfo") or {}).get("info_data") or {}
        bittiming = info.get("bittiming") or {}
        driver, product, serial = _usb_meta(iface)
        # iproute2 reports the parent bus directly, which distinguishes a
        # plug-in USB adapter from a soldered-down controller far more reliably
        # than pattern-matching the path. Fall back to sysfs on older versions.
        bus = str(entry.get("parentdev") or "") or _bus_of(iface)
        bus_kind = str(entry.get("parentbus") or "")
        if not bus_kind:
            bus_kind = "usb" if BUS_RE.match(bus) else "platform"
        adapters.append(
            Adapter(
                iface=iface,
                bus=bus,
                bus_kind=bus_kind,
                # operstate for a CAN link is UNKNOWN even when up, so trust the
                # flags, which is also what `ip -br` renders.
                state="UP" if "UP" in (entry.get("flags") or []) else "DOWN",
                bitrate=bittiming.get("bitrate"),
                restart_ms=info.get("restart_ms"),
                driver=driver,
                product=product,
                serial=serial,
                role=name_to_role.get(iface),
                errors={
                    "state": info.get("state", ""),
                    "restarts": info.get("restart_count", 0),
                    "bus_errors": (info.get("berr_counter") or {}).get("tx", 0),
                },
            )
        )
    adapters.sort(key=lambda a: (not a.assignable, a.usb_port, a.iface))
    return adapters


def plan(
    adapters: list[Adapter], wanted: dict[str, str]
) -> tuple[list[str], list[dict[str, str]]]:
    """Work out how to reach ``wanted`` (bus -> target interface name).

    Returns ``(park, assign)``: interfaces whose current name is standing in the
    way and must be renamed out of the road first, then the assignments to
    apply. Doing this in two passes is what makes swapping the two arms work at
    all -- renaming ``can_left`` to ``can_right`` while a ``can_right`` still
    exists fails with EEXIST.
    """
    by_bus = {a.bus: a for a in adapters if a.bus}
    for bus, name in wanted.items():
        if not BUS_RE.match(bus):
            held = by_bus.get(bus)
            if held is not None and not held.assignable:
                raise CanError(
                    f"{held.iface} is an on-board {held.bus_kind} controller "
                    f"({bus}), not a pluggable USB adapter; it cannot be "
                    "assigned to an arm"
                )
            raise CanError(f"not a USB port path: {bus!r}")
        if not NAME_RE.match(name):
            raise CanError(f"not a valid interface name: {name!r}")
        if bus not in by_bus:
            raise CanError(f"no CAN adapter on USB port {bus}")
    if len(set(wanted.values())) != len(wanted):
        raise CanError("two adapters cannot share one interface name")

    targets = set(wanted.values())
    park = [
        a.iface
        for a in adapters
        # In the way: it holds a name we need, but is not the adapter we are
        # about to give that name to.
        if a.iface in targets and wanted.get(a.bus) != a.iface
    ]
    assign = [{"bus": bus, "name": name} for bus, name in sorted(wanted.items())]
    return park, assign


def apply(
    wanted: dict[str, str], bitrate: int, *, repo: Path, timeout: float = 120.0
) -> dict[str, Any]:
    """Park conflicting names, then configure each adapter, as root.

    All of it happens inside one privileged helper so that the dashboard needs
    exactly one sudo entry, and so a half-applied rename cannot be interleaved
    with anything else.
    """
    adapters = discover()
    park, assign = plan(adapters, wanted)
    if not Path(HELPER).exists():
        raise CanError(
            f"{HELPER} is not installed — run ops/install.sh on this host to "
            "set up CAN control"
        )
    if not 10_000 <= int(bitrate) <= 8_000_000:
        raise CanError(f"implausible bitrate: {bitrate}")

    argv = ["sudo", "-n", HELPER, str(int(bitrate))]
    argv += [f"{a['name']}={a['bus']}" for a in assign]
    try:
        out = subprocess.run(  # noqa: S603 - argv validated above
            argv, capture_output=True, text=True, timeout=timeout,
            check=False, cwd=str(repo),
        )
    except subprocess.SubprocessError as exc:
        raise CanError(f"could not run {HELPER}: {exc}") from exc

    log = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        if "password is required" in log or "sudo:" in log and "not allowed" in log:
            raise CanError(
                "the dashboard is not permitted to run the CAN helper without a "
                "password — re-run ops/install.sh to install the sudoers rule"
            )
        raise CanError(log.strip() or f"CAN helper exited {out.returncode}")
    return {"parked": park, "assigned": assign, "log": log, "adapters": [
        a.as_dict() for a in discover()
    ]}
