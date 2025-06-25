import os
import torch
import torchaudio
import torchaudio.transforms as T
import logging
import numpy as np
from scipy.io import wavfile
from config import SILERO_VAD_DIR

logger = logging.getLogger(__name__)

class SileroVADProcessor:
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        try:
            # Kiểm tra thư mục lưu mô hình SileroVAD
            os.makedirs(SILERO_VAD_DIR, exist_ok=True)
            model_path = os.path.join(SILERO_VAD_DIR, "silero_vad.jit")

            if not os.path.exists(model_path):
                logger.info(f"SileroVAD model not found at {model_path}. Downloading...")
                # Tải mô hình từ torch.hub và lưu vào SILERO_VAD_DIR
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=True,  # Tải mới nếu chưa có
                    trust_repo=True
                )
                # Lưu mô hình vào thư mục cục bộ
                torch.jit.save(model, model_path)
                logger.info(f"SileroVAD model saved to {model_path}")
            else:
                logger.info(f"Loading SileroVAD model from {model_path}")
                model = torch.jit.load(model_path)

            self.model = model
            self.get_speech_timestamps = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                skip_validation=True
            )[1][0]  # Lấy hàm get_speech_timestamps từ utils
            logger.info(f"Using torchaudio version: {torchaudio.__version__}")
            logger.info(f"Loaded silero-vad model successfully from {SILERO_VAD_DIR}")
        except Exception as e:
            logger.error(f"Error loading silero-vad model: {str(e)}")
            raise

    def process(self, input_path, output_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        try:
            logger.info(f"Loading input audio: {input_path}")
            waveform, sample_rate = torchaudio.load(input_path)
            logger.info(f"Input waveform shape: {waveform.shape}, sample rate: {sample_rate}")
            if sample_rate != self.sampling_rate:
                logger.info(f"Resampling audio from {sample_rate} Hz to {self.sampling_rate} Hz")
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
                waveform = resampler(waveform)
            logger.info(f"Resampled waveform shape: {waveform.shape}")

            logger.info("Running VAD on audio")
            if waveform.shape[0] > 1:
                logger.info("Converting stereo to mono for VAD")
                waveform_vad = waveform.mean(dim=0, keepdim=True)
            else:
                waveform_vad = waveform

            speech_timestamps = self.get_speech_timestamps(
                waveform_vad,
                self.model,
                sampling_rate=self.sampling_rate,
                threshold=0.2,
                min_speech_duration_ms=200,
                min_silence_duration_ms=500
            )
            if not speech_timestamps:
                logger.warning("No speech detected by VAD. Using original audio.")
                audio = waveform.mean(dim=0, keepdim=True).numpy().T
                if audio.dtype != np.int16:
                    audio = (audio * 32767).astype(np.int16)
                wavfile.write(output_path, self.sampling_rate, audio)
                logger.info(f"Saved original audio to: {output_path}")
                return speech_timestamps

            logger.info(f"Found {len(speech_timestamps)} speech segments")

            chunks = [waveform[:, int(ts["start"]):int(ts["end"])] for ts in speech_timestamps]
            if not chunks:
                raise Exception("No valid audio chunks after VAD processing.")
            merged = torch.cat(chunks, dim=1)
            if merged.ndim == 1:
                logger.info("Converting mono waveform to 2D")
                merged = merged.unsqueeze(0)
            logger.info(f"Merged waveform shape: {merged.shape}")

            logger.info(f"Saving processed audio to: {output_path}")
            waveform_np = merged.numpy().T
            if waveform_np.dtype != np.int16:
                waveform_np = (waveform_np * 32767).astype(np.int16)
            wavfile.write(output_path, self.sampling_rate, waveform_np)
            logger.info(f"Audio saved successfully to: {output_path}")
            return speech_timestamps
        except Exception as e:
            logger.error(f"Error in VAD processing: {str(e)}")
            raise
        