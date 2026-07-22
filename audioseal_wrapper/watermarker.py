"""Sosie AudioSeal Wrapper Module.

Provides high-level helpers to embed and detect AudioSeal watermarks
on PyTorch audio tensors or numpy waveform arrays without external C++ binary deps.
"""
import io
import os
import wave

# Disable torch.compile in moshi/audioseal to run cleanly on Windows without MSVC cl.exe compiler
os.environ["NO_TORCH_COMPILE"] = "1"

import numpy as np
import torch

try:
    from audioseal import AudioSeal
except ImportError:
    AudioSeal = None


def tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int = 24000) -> io.BytesIO:
    """Converts a 1D/2D PyTorch audio tensor float [-1, 1] into a 16-bit PCM WAV BytesIO buffer."""
    audio_np = audio_tensor.detach().cpu().numpy().squeeze()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf


def wav_bytes_to_tensor(file_bytes_or_buf) -> tuple[torch.Tensor, int]:
    """Reads a WAV file BytesIO buffer or file object into a float32 PyTorch tensor shape (1, 1, samples)."""
    with wave.open(file_bytes_or_buf, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0
        tensor = torch.from_numpy(audio_float).unsqueeze(0).unsqueeze(0)
        return tensor, sr


class SosieWatermarker:
    """Encapsulates AudioSeal generator and detector models."""

    def __init__(self, generator_card: str = "audioseal_wm_16bits", detector_card: str = "audioseal_detector_16bits", device: str = None):
        if AudioSeal is None:
            raise RuntimeError("audioseal package is not installed.")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SosieWatermarker] Loading AudioSeal models on {self.device}...")
        
        self.generator = AudioSeal.load_generator(generator_card).to(self.device)
        self.generator.eval()
        
        self.detector = AudioSeal.load_detector(detector_card).to(self.device)
        self.detector.eval()
        print("[SosieWatermarker] AudioSeal loaded successfully.")

    def embed_watermark(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 24000,
        alpha: float = 1.2,
        message: torch.Tensor = None
    ) -> torch.Tensor:
        """Embeds localized AudioSeal watermark into audio_tensor.
        
        audio_tensor: shape (channels, samples) or (batch, channels, samples)
        Returns: watermarked_audio tensor matching input shape
        """
        orig_shape = audio_tensor.shape
        if audio_tensor.dim() == 1:
            wav = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            wav = audio_tensor.unsqueeze(0)
        else:
            wav = audio_tensor

        wav = wav.to(self.device)
        with torch.no_grad():
            watermark = self.generator.get_watermark(wav, message=message)
            watermarked_wav = wav + (alpha * watermark)

        watermarked_wav = watermarked_wav.cpu()
        if len(orig_shape) == 1:
            return watermarked_wav.squeeze(0).squeeze(0)
        elif len(orig_shape) == 2:
            return watermarked_wav.squeeze(0)
        return watermarked_wav

    def detect_watermark(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 16000
    ) -> dict:
        """Detects AudioSeal watermark presence in audio_tensor."""
        if audio_tensor.dim() == 1:
            wav = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            wav = audio_tensor.unsqueeze(0)
        else:
            wav = audio_tensor

        wav = wav.to(self.device)
        with torch.no_grad():
            result, message = self.detector.detect_watermark(wav)

        score = float(result) if isinstance(result, (float, int)) else float(result.max())
        is_wm = score > 0.5
        
        msg_list = None
        if message is not None:
            msg_list = message.cpu().squeeze().tolist() if hasattr(message, "cpu") else list(message)

        return {
            "is_watermarked": is_wm,
            "score": score,
            "message": msg_list
        }


# Singleton instance for simple module-level imports
_watermarker_instance = None


def get_watermarker() -> SosieWatermarker:
    global _watermarker_instance
    if _watermarker_instance is None:
        _watermarker_instance = SosieWatermarker()
    return _watermarker_instance
