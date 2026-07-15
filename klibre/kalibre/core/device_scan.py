"""
Détection des périphériques audio USB réellement branchés.

Windows enregistre les anciennes cartes débranchées : on croise PortAudio
avec la liste PnP « PresentOnly » + un test d'ouverture de flux (check_*).
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

import sounddevice as sd


@dataclass(frozen=True)
class UsbPresentDevice:
    """Périphérique USB vu par Windows comme présent (câble branché)."""

    friendly_name: str
    instance_id: str

    @property
    def key(self) -> str:
        return _normalize_name(self.friendly_name)


def _normalize_name(name: str) -> str:
    """Clé de regroupement — même interface MME/WASAPI → une entrée."""
    n = name.lower().strip()
    n = re.sub(r"\s*\((wasapi|directsound|mme|wdm-ks|asio|windows (direct|wasapi))\)\s*", " ", n, flags=re.I)
    n = re.sub(r"192k", "", n, flags=re.I)
    n = re.sub(r"[^\w\s-]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _hostapi_priority(hostapi: str) -> int:
    h = hostapi.lower()
    if "wasapi" in h:
        return 0
    if h == "windows" or "windows multimedia" in h:
        return 1
    if "wdm" in h:
        return 2
    if "directsound" in h:
        return 3
    if "mme" in h:
        return 4
    if "asio" in h:
        return 5
    return 6


def parse_pnp_device_lines(lines: list[str] | tuple[str, ...]) -> tuple[UsbPresentDevice, ...]:
    """Parse la sortie PowerShell PnP en objets structurés, sans casser sur des lignes vides."""
    devices: list[UsbPresentDevice] = []
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line or "|" not in line:
            continue
        name, inst = line.split("|", 1)
        name = name.strip()
        inst = inst.strip()
        if not name or not inst:
            continue
        devices.append(UsbPresentDevice(friendly_name=name, instance_id=inst))
    return tuple(devices)


def _is_audio_pnp_usb(name: str) -> bool:
    """
    Filtre les faux positifs PnP (claviers, hubs, stockage…).
    Garde les interfaces audio USB reconnues.
    """
    n = name.lower()
    excluded = (
        "hub usb",
        "stockage",
        "storage",
        "scsi",
        "bluetooth",
        "composite",
        "contrôleur hôte",
        "controller",
        "périphérique d'entrée usb",
        "peripherique d'entree usb",
        "input device",
    )
    if any(x in n for x in excluded):
        return False
    hints = (
        "audio",
        "sound",
        "microphone",
        "mic ",
        "umc",
        "scarlett",
        "focusrite",
        "behringer",
        "motu",
        "rme",
        "presonus",
        "steinberg",
        "interface",
        "uac",
        "192k",
        "202hd",
        "umik",
        "line",
        "spdif",
        "usb audio",
        "audio interface",
        "solo",
        "2i2",
        "2i4",
        "4i4",
        "18i8",
    )
    return any(h in n for h in hints)


def filter_audio_usb_present(present_usb: tuple[UsbPresentDevice, ...]) -> tuple[UsbPresentDevice, ...]:
    return tuple(d for d in present_usb if _is_audio_pnp_usb(d.friendly_name))


def _token_set(name: str) -> set[str]:
    """Mots significatifs pour rapprocher noms PnP et noms PortAudio."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    stop = {"audio", "usb", "device", "driver", "interface", "sound", "digital", "analog"}
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _names_match(portaudio_name: str, pnp_name: str) -> bool:
    """Correspondance floue entre libellé PortAudio et FriendlyName Windows."""
    pa = _normalize_name(portaudio_name)
    pn = _normalize_name(pnp_name)
    if pa in pn or pn in pa:
        return True
    pa_t, pn_t = _token_set(pa), _token_set(pn)
    if not pa_t or not pn_t:
        return False
    common = pa_t & pn_t
    return len(common) >= 1 and len(common) >= min(len(pa_t), len(pn_t)) // 2


@lru_cache(maxsize=1)
def query_present_usb_devices() -> tuple[UsbPresentDevice, ...]:
    """
    Liste les périphériques USB **présents** (branchés) via PowerShell.

    Sur Linux/macOS : retourne vide — le filtre actif repose sur check_* seul.
    """
    if sys.platform != "win32":
        return tuple()

    script = (
        "Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | "
        "Where-Object { $_.InstanceId -like 'USB*' -and $_.Status -eq 'OK' } | "
        "ForEach-Object { \"$($_.FriendlyName)|$($_.InstanceId)\" }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return tuple()

    return parse_pnp_device_lines(proc.stdout.splitlines())


def clear_device_cache() -> None:
    """À appeler avant un refresh manuel (évite cache PnP obsolète)."""
    query_present_usb_devices.cache_clear()


def is_likely_usb(name: str, hostapi: str, present_usb: tuple[UsbPresentDevice, ...]) -> bool:
    blob = f"{name} {hostapi}".lower()
    if "usb" in blob or "uac" in blob:
        return True
    if any(token in blob for token in ("scarlett", "focusrite", "motu", "rme", "behringer", "presonus", "steinberg", "umc", "audio interface", "usb audio")):
        return True
    for dev in present_usb:
        if _names_match(name, dev.friendly_name):
            return True
    return False


def is_linked_to_present_usb(name: str, present_usb: tuple[UsbPresentDevice, ...]) -> bool:
    """Sur Windows : le device PortAudio correspond à une interface USB audio branchée."""
    if sys.platform != "win32":
        return True
    audio_usb = filter_audio_usb_present(present_usb)
    if not audio_usb:
        return False
    return any(_names_match(name, d.friendly_name) for d in audio_usb)


def is_input_active(device_index: int, sample_rate: int, max_channels: int = 1) -> bool:
    """True si PortAudio peut ouvrir une capture sur ce device (branché + dispo)."""
    if max_channels < 1:
        return False
    try:
        sd.check_input_settings(
            device=device_index,
            channels=min(max_channels, 2),
            samplerate=sample_rate,
        )
        return True
    except Exception:
        return False


def is_output_active(device_index: int, sample_rate: int, max_channels: int = 1) -> bool:
    if max_channels < 1:
        return False
    try:
        sd.check_output_settings(
            device=device_index,
            channels=min(max_channels, 2),
            samplerate=sample_rate,
        )
        return True
    except Exception:
        return False


def _is_full_duplex(dev) -> bool:
    return dev.max_input_channels > 0 and dev.max_output_channels > 0


def _dedupe_devices(devices: list) -> list:
    """
    Une seule entrée par interface physique.

    Préfère : full-duplex (in+out) puis WASAPI sur Windows.
    """
    best: dict[str, object] = {}
    for dev in devices:
        key = _normalize_name(dev.name)
        prev = best.get(key)
        if prev is None:
            best[key] = dev
            continue
        prev_duplex = _is_full_duplex(prev)
        dev_duplex = _is_full_duplex(dev)
        if dev_duplex and not prev_duplex:
            best[key] = dev
            continue
        if prev_duplex and not dev_duplex:
            continue
        if _hostapi_priority(dev.hostapi) < _hostapi_priority(prev.hostapi):
            best[key] = dev
    return list(best.values())


def find_full_duplex_index(input_index: int, output_index: int) -> int | tuple[int, int]:
    """
    Retourne un index unique in+out pour playrec (évite les paires in/out séparées Windows).

    Sur UMC / WASAPI, entrée et sortie peuvent avoir des index PortAudio différents ;
    le duplex sur un seul index est plus fiable.
    """
    if input_index == output_index:
        return input_index

    for idx in (input_index, output_index):
        dev = sd.query_devices(idx)
        if int(dev["max_input_channels"]) > 0 and int(dev["max_output_channels"]) > 0:
            return idx

    target = _normalize_name(str(sd.query_devices(input_index)["name"]))
    hostapis = sd.query_hostapis()
    candidates: list[tuple[int, int, bool]] = []

    for idx, dev in enumerate(sd.query_devices()):
        if _normalize_name(str(dev["name"])) != target:
            continue
        in_ch = int(dev["max_input_channels"])
        out_ch = int(dev["max_output_channels"])
        if in_ch < 1 or out_ch < 1:
            continue
        host = hostapis[dev["hostapi"]]["name"]
        candidates.append((idx, _hostapi_priority(host), True))

    if candidates:
        candidates.sort(key=lambda c: c[1])
        return candidates[0][0]

    return (input_index, output_index)
