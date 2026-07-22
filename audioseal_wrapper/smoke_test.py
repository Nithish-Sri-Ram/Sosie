"""Smoke test for Sosie AudioSeal Integration."""
import torch
from audioseal_wrapper import get_watermarker, tensor_to_wav_bytes, wav_bytes_to_tensor


def run_smoke_test():
    print("--- Running Sosie AudioSeal Smoke Test ---")
    watermarker = get_watermarker()

    # Generate 2 seconds of dummy audio sine wave (24kHz sample rate)
    sr = 24000
    duration_sec = 2.0
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))
    audio_orig = torch.sin(2 * 3.14159 * 440.0 * t).unsqueeze(0)  # shape (1, samples)

    print(f"Generated test sine wave audio shape: {audio_orig.shape}")

    # 1. Detect on raw unwatermarked audio
    res_before = watermarker.detect_watermark(audio_orig, sample_rate=sr)
    print(f"Detection BEFORE watermarking: {res_before}")
    assert not res_before["is_watermarked"], "Expected unwatermarked audio to yield is_watermarked=False"

    # 2. Embed watermark
    print("Embedding AudioSeal watermark...")
    audio_wm = watermarker.embed_watermark(audio_orig, sample_rate=sr, alpha=1.2)
    print(f"Watermarked audio shape: {audio_wm.shape}")

    # 3. Detect on watermarked audio
    res_after = watermarker.detect_watermark(audio_wm, sample_rate=sr)
    print(f"Detection AFTER watermarking: {res_after}")
    assert res_after["is_watermarked"], "Expected watermarked audio to yield is_watermarked=True"
    assert res_after["score"] > 0.5, f"Expected watermark score > 0.5, got {res_after['score']}"

    # 4. Test WAV bytes conversion buffer
    buf = tensor_to_wav_bytes(audio_wm, sample_rate=sr)
    tensor_loaded, sr_loaded = wav_bytes_to_tensor(buf)
    res_buffer = watermarker.detect_watermark(tensor_loaded, sample_rate=sr_loaded)
    print(f"Detection after WAV buffer round-trip: {res_buffer}")
    assert res_buffer["is_watermarked"], "Expected loaded WAV buffer to remain watermarked"

    print("\nOK - Sosie AudioSeal smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
