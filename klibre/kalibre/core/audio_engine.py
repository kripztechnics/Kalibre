"""

Interaction avec les cartes son via sounddevice (PortAudio).



Mesure type Smaart/REW :

  - sortie → enceinte + boucle loopback (IN2)

  - micro sur IN1

  - le délai acoustique = corrélation(micro, loopback)

"""



from __future__ import annotations



import sys

from dataclasses import dataclass



import numpy as np

import sounddevice as sd

from numpy.typing import NDArray

from kalibre.core.device_scan import (

    _dedupe_devices,

    clear_device_cache,

    find_full_duplex_index,

    is_input_active,

    is_likely_usb,

    is_linked_to_present_usb,

    is_output_active,

    query_present_usb_devices,

)



class MeasurementAborted(RuntimeError):
    """Mesure ou lecture interrompue par l'utilisateur (sd.stop)."""




@dataclass(frozen=True)

class AudioDeviceInfo:

    index: int

    name: str

    hostapi: str

    max_input_channels: int

    max_output_channels: int

    default_sample_rate: float

    is_usb: bool

    is_active: bool = True



    @property

    def can_input(self) -> bool:

        return self.max_input_channels > 0



    @property

    def can_output(self) -> bool:

        return self.max_output_channels > 0



    def label(self) -> str:

        tags: list[str] = []

        if self.is_usb:

            tags.append("USB")

        tags.append(self.hostapi.split()[0])

        if self.can_input and self.can_output:

            tags.append("in/out")

        elif self.can_input:

            tags.append("in")

        elif self.can_output:

            tags.append("out")

        return f"{self.name} [{', '.join(tags)}]"





@dataclass

class CaptureResult:

    """Enregistrement duplex : référence électrique + micro."""



    loopback: NDArray[np.float64]

    mic: NDArray[np.float64]

    sample_rate: int

    mic_rms: float = 0.0

    loopback_rms: float = 0.0

    mic_peak: float = 0.0

    loopback_peak: float = 0.0

    duplex_device: int | tuple[int, int] | None = None





def _build_device_info(

    index: int, dev: dict, hostapi_name: str, present_usb: tuple

) -> AudioDeviceInfo:

    sr = int(dev["default_samplerate"]) if dev["default_samplerate"] else 48_000

    name = str(dev["name"])

    in_ch = int(dev["max_input_channels"])

    out_ch = int(dev["max_output_channels"])



    active = False

    if in_ch > 0 and is_input_active(index, sr, in_ch):

        active = True

    if out_ch > 0 and is_output_active(index, sr, out_ch):

        active = True



    return AudioDeviceInfo(

        index=index,

        name=name,

        hostapi=hostapi_name,

        max_input_channels=in_ch,

        max_output_channels=out_ch,

        default_sample_rate=float(dev["default_samplerate"]),

        is_usb=is_likely_usb(name, hostapi_name, present_usb),

        is_active=active,

    )





def enumerate_devices(*, usb_only: bool = False, active_only: bool = False) -> list[AudioDeviceInfo]:

    """Liste PortAudio, filtres USB / actifs optionnels."""

    clear_device_cache()

    present_usb = query_present_usb_devices()

    hostapis = sd.query_hostapis()

    raw = sd.query_devices()

    devices: list[AudioDeviceInfo] = []



    for index, dev in enumerate(raw):

        hostapi_name = hostapis[dev["hostapi"]]["name"]

        info = _build_device_info(index, dev, hostapi_name, present_usb)



        if usb_only and not info.is_usb:

            continue

        if usb_only and sys.platform == "win32" and present_usb:

            if not is_linked_to_present_usb(info.name, present_usb):

                continue

        if active_only and not info.is_active:

            continue



        devices.append(info)



    devices.sort(key=lambda d: d.name.lower())

    return devices





def enumerate_active_usb_devices() -> list[AudioDeviceInfo]:

    """USB branchés + utilisables, une entrée par interface (WASAPI prioritaire)."""

    return _dedupe_devices(enumerate_devices(usb_only=True, active_only=True))





def list_input_devices(*, active_usb_only: bool = True) -> list[AudioDeviceInfo]:

    source = enumerate_active_usb_devices() if active_usb_only else enumerate_devices()

    return [d for d in source if d.can_input]





def list_output_devices(*, active_usb_only: bool = True) -> list[AudioDeviceInfo]:

    source = enumerate_active_usb_devices() if active_usb_only else enumerate_devices()

    return [d for d in source if d.can_output]





def default_input_index() -> int | None:

    try:

        return int(sd.default.device[0])  # type: ignore[index]

    except (TypeError, ValueError, sd.PortAudioError):

        return None





def default_output_index() -> int | None:

    try:

        return int(sd.default.device[1])  # type: ignore[index]

    except (TypeError, ValueError, sd.PortAudioError):

        return None





def loopback_rms(signal: NDArray[np.float64]) -> float:

    if len(signal) == 0:

        return 0.0

    return float(np.sqrt(np.mean(signal**2)))





def rms_to_dbfs(rms: float) -> float:

    if rms <= 1e-12:

        return -120.0

    return float(20.0 * np.log10(rms))





class AudioEngine:

    """Émission + capture double entrée (loopback + micro)."""



    def __init__(

        self,

        sample_rate: int = 48_000,

        blocksize: int = 256,

        input_device: int | None = None,

        output_device: int | None = None,

        mic_channel: int = 1,

        loopback_channel: int = 2,

        output_channel: int = 1,

        *,

        stereo_output: bool = True,

    ) -> None:

        self.sample_rate = sample_rate

        self.blocksize = blocksize

        self.input_device = input_device

        self.output_device = output_device

        self.mic_channel = max(1, mic_channel)

        self.loopback_channel = max(1, loopback_channel)

        self.output_channel = max(1, output_channel)

        self.stereo_output = stereo_output



    def play_loop(self, signal: NDArray[np.float64]) -> None:

        """Lecture en boucle jusqu'à sd.stop() (bouton ARRÊTER)."""

        if len(signal) == 0:

            raise ValueError("Signal vide.")

        if self.output_device is None:

            raise sd.PortAudioError("Sortie audio non sélectionnée.")



        mono = signal.astype(np.float64, copy=False)

        out_idx = int(self.output_device)

        out_dev = self._device(out_idx)

        play_data = self._build_play_buffer(mono, out_dev.max_output_channels)



        sd.check_output_settings(

            device=out_idx,

            channels=play_data.shape[1],

            samplerate=self.sample_rate,

        )

        sd.stop()

        sd.play(

            play_data,

            samplerate=self.sample_rate,

            device=out_idx,

            loop=True,

            blocksize=self.blocksize,

        )



    def play_and_capture(

        self,

        signal: NDArray[np.float64],

        *,

        stop_check=None,

    ) -> CaptureResult:

        if len(signal) == 0:

            raise ValueError("Signal vide.")



        if self.input_device is None or self.output_device is None:

            raise sd.PortAudioError("Entrée ou sortie audio non sélectionnée.")



        mono = signal.astype(np.float64, copy=False)

        duplex = find_full_duplex_index(int(self.input_device), int(self.output_device))



        in_idx = self.input_device if isinstance(duplex, tuple) else duplex

        in_dev = self._device(int(in_idx))



        if self.mic_channel > in_dev.max_input_channels:

            raise ValueError(

                f"Canal micro {self.mic_channel} indisponible "

                f"(max {in_dev.max_input_channels})."

            )

        if self.loopback_channel > in_dev.max_input_channels:

            raise ValueError(

                f"Canal loopback {self.loopback_channel} indisponible "

                f"(max {in_dev.max_input_channels})."

            )



        out_idx = self.output_device if isinstance(duplex, tuple) else duplex

        out_dev = self._device(int(out_idx))

        play_data = self._build_play_buffer(mono, out_dev.max_output_channels)



        n_record = max(self.mic_channel, self.loopback_channel, 1)



        sd.check_input_settings(

            device=in_idx,

            channels=n_record,

            samplerate=self.sample_rate,

        )

        sd.check_output_settings(

            device=out_idx if isinstance(duplex, tuple) else duplex,

            channels=play_data.shape[1],

            samplerate=self.sample_rate,

        )



        sd.stop()

        recorded = sd.playrec(

            play_data,

            samplerate=self.sample_rate,

            device=duplex,

            channels=n_record,

            blocksize=self.blocksize,

            blocking=False,

            latency="high",

        )



        while True:

            if stop_check and stop_check():

                sd.stop()

                raise MeasurementAborted("Mesure interrompue.")

            stream = sd.get_stream()

            if stream is None or not stream.active:

                break

            sd.wait(0.05)



        if stop_check and stop_check():

            raise MeasurementAborted("Mesure interrompue.")



        if recorded.ndim == 1:

            recorded = recorded.reshape(-1, 1)



        mic = recorded[:, self.mic_channel - 1].astype(np.float64)

        loopback = recorded[:, self.loopback_channel - 1].astype(np.float64)

        return CaptureResult(

            loopback=loopback,

            mic=mic,

            sample_rate=self.sample_rate,

            mic_rms=loopback_rms(mic),

            loopback_rms=loopback_rms(loopback),

            mic_peak=float(np.max(np.abs(mic))) if len(mic) else 0.0,

            loopback_peak=float(np.max(np.abs(loopback))) if len(loopback) else 0.0,

            duplex_device=duplex,

        )



    def _build_play_buffer(self, mono: NDArray[np.float64], n_out: int) -> NDArray[np.float64]:

        """Matrice de sortie — par défaut le même signal sur toutes les sorties (L+R)."""

        if n_out <= 1:

            return mono.reshape(-1, 1)



        play_data = np.zeros((len(mono), n_out), dtype=np.float64)

        if self.stereo_output:

            for ch in range(n_out):

                play_data[:, ch] = mono

        else:

            play_data[:, min(self.output_channel - 1, n_out - 1)] = mono

        return play_data



    def test_output(self, duration_s: float = 0.3, frequency_hz: float = 440.0) -> CaptureResult:

        t = np.linspace(0, duration_s, int(duration_s * self.sample_rate), endpoint=False)

        tone = (0.2 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float64)

        return self.play_and_capture(tone)



    @staticmethod

    def _device(index: int | None) -> AudioDeviceInfo:

        if index is None:

            raise sd.PortAudioError("Aucun périphérique sélectionné.")

        present_usb = query_present_usb_devices()

        dev = sd.query_devices(index)

        hostapis = sd.query_hostapis()

        hostapi_name = hostapis[dev["hostapi"]]["name"]

        return _build_device_info(index, dev, hostapi_name, present_usb)


