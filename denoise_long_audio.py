import os
import torch
import torchaudio
import logging
from audio_denoiser import AudioDenoiser

logger = logging.getLogger(__name__)

def denoise_long_audio(
    input_path: str,
    output_path: str,
    segment_sec: float = 20.0,
    overlap_sec: float = 1.0
):
    """
    Khử nhiễu âm thanh WAV dài bằng cách chia thành các đoạn, dùng AudioDenoiser.

    Args:
        input_path: Đường dẫn file WAV đầu vào
        output_path: Đường dẫn file WAV đầu ra
        segment_sec: Độ dài mỗi đoạn (giây)
        overlap_sec: Độ chồng lấn giữa các đoạn (giây)
    """
    try:
        logger.info(f"Denoising audio: {input_path} -> {output_path}")

        # Khởi động AudioDenoiser với apply_noise_reduction = True
        denoiser = AudioDenoiser(sample_rate=16000)

        # Đọc audio
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Chuyển sang mono nếu cần
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        segment_samples = int(segment_sec * sample_rate)
        overlap_samples = int(overlap_sec * sample_rate)
        total_samples = waveform.shape[1]
        denoised_segments = []

        # Chia và khử nhiễu từng đoạn
        for start in range(0, total_samples, segment_samples - overlap_samples):
            end = min(start + segment_samples, total_samples)
            segment = waveform[:, start:end]
            if segment.shape[1] == 0:
                continue
            denoised_segment = denoiser.process(segment.numpy())  # Numpy -> xử lý
            denoised_segments.append(torch.tensor(denoised_segment))

        # Ghép lại
        if not denoised_segments:
            raise Exception("No valid segments after denoising")
        denoised_waveform = torch.cat(denoised_segments, dim=1)
        if denoised_waveform.shape[1] > total_samples:
            denoised_waveform = denoised_waveform[:, :total_samples]

        # Lưu file
        torchaudio.save(output_path, denoised_waveform, sample_rate)
        logger.info(f"Denoised audio saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error denoising audio: {str(e)}")
        raise Exception(f"Error denoising audio: {str(e)}")
