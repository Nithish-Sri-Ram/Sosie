"""AudioSeal integration package for Sosie."""
from .watermarker import SosieWatermarker, get_watermarker, tensor_to_wav_bytes, wav_bytes_to_tensor

__all__ = ["SosieWatermarker", "get_watermarker", "tensor_to_wav_bytes", "wav_bytes_to_tensor"]
