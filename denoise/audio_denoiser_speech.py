import math
import sys
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from audio_denoiser.AudioDenoiser import AudioDenoiser
from typing import Optional


def denoise_long_audio(
    in_path: str,
    out_path: str,
    segment_sec: float = 20.0,   # chiều dài một đoạn (giây)
    overlap_sec: float = 1.0,    # phần chồng lấn (giây)
    auto_scale: bool = True,
    device: Optional[str] = None,
):
    """Stream‑denoise a large WAV file."""
    info = torchaudio.info(in_path)
    sr = info.sample_rate
    total_frames = info.num_frames
    n_channels = info.num_channels

    seg_frames = int(segment_sec * sr)
    ovl_frames = int(overlap_sec * sr)

    # ── Chọn device ────────────────────────────────────────────
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    denoiser = AudioDenoiser(device=device)

    # ── Vùng đệm ghi kết quả cuối cùng (danh sách tensor) ─────
    cleaned_chunks: list[torch.Tensor] = []

    # ── Đọc tệp từng khúc với torchaudio.load(..., frame_offset, num_frames)
    offset = 0
    prev_tail: torch.Tensor | None = None

    while offset < total_frames:
        # đọc segment + overlap (trừ lần cuối có thể thiếu)
        num_frames = min(seg_frames + ovl_frames, total_frames - offset)
        wav, _ = torchaudio.load(in_path, frame_offset=offset, num_frames=num_frames)

        # ghép đuôi sạch của đoạn trước (nếu có) vào đầu đoạn này để tránh lỗ hổng
        if prev_tail is not None:
            wav = torch.cat([prev_tail, wav], dim=1)

        # denoise
        clean = denoiser.process_waveform(
            wav, sr, return_cpu_tensor=True, auto_scale=auto_scale
        )

        # Cắt phần **không** overlap (đầu) để ghi ngay; giữ đuôi để ghép đoạn sau
        write_len = clean.shape[1] - ovl_frames
        if write_len > 0:
            cleaned_chunks.append(clean[:, :write_len])

            # phần tail (kể cả raw lẫn clean) để nối liền mạch
            prev_tail = wav[:, -ovl_frames:]
        else:  # đoạn quá ngắn (cuối file)
            prev_tail = wav

        offset += seg_frames

    # Xử lý nốt phần đuôi cuối cùng
    if prev_tail is not None:
        tail_clean = denoiser.process_waveform(
            prev_tail, sr, return_cpu_tensor=True, auto_scale=auto_scale
        )
        cleaned_chunks.append(tail_clean)

    # ── Ghép & lưu ────────────────────────────────────────────
    result = torch.cat(cleaned_chunks, dim=1)                   # (C, T)
    # torchaudio/save yêu cầu (C, T) tensor float32 /-1‒1
    torchaudio.save(out_path, result, 16000)
    print(f"✅  Done. Clean file saved to {out_path}")


# if __name__ == "__main__":
#     in_wav, out_wav = "/data/cuongdd/ai_service/data/input/AmNhac.wav", "./AmNhac_clean.wav"

#     print("in :", torchaudio.info("/data/cuongdd/ai_service/data/input/Phim le.wav").sample_rate)
#     print("out:", torchaudio.info("./Phim le_clean.wav").sample_rate)

#     denoise_long_audio(
#         in_path=str(in_wav),
#         out_path=str(out_wav),
#         segment_sec=20.0,   # chỉnh mốc 10–30 s tuỳ RAM/GPU
#         overlap_sec=1.0,    # 0.5–2 s để tránh click/pop
#         auto_scale=True,
#     )
