import os
from pydub import AudioSegment
import numpy as np
import logging
from vad_speeched import SileroVADProcessor
from audio_denoiser import AudioDenoiser

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def process_audio(input_path: str, output_path: str, apply_noise_reduction: bool = False) -> str:
        try:
            logger.info(f"Processing audio with VAD: {input_path} -> {output_path}")
            vad_output_path = output_path.replace(".wav", "_vad.wav")
            vad_processor = SileroVADProcessor()
            speech_timestamps = vad_processor.process(input_path, vad_output_path)

            # Lưu timestamp gốc
            original_audio = AudioSegment.from_file(input_path)
            original_audio = original_audio.set_frame_rate(16000).set_channels(1)
            original_output_path = output_path.replace(".wav", "_original.wav")
            original_audio.export(original_output_path, format="wav")
            logger.info(f"Saved original (normalized) audio to: {original_output_path}")

            if not speech_timestamps:
                logger.warning("No speech segments detected. Using original audio instead.")
                original_audio.export(output_path, format="wav")
                return output_path

            if not os.path.exists(vad_output_path):
                raise Exception("VAD output not created")

            # Log speech timestamps
            for i, segment in enumerate(speech_timestamps):
                start_time = segment['start'] / 16000
                end_time = segment['end'] / 16000
                logger.info(f"Speech segment {i+1}: {start_time:.3f}s --> {end_time:.3f}s")

            # Verify audio file integrity
            if os.path.getsize(vad_output_path) == 0:
                raise Exception("VAD output file is empty")

            audio = AudioSegment.from_file(vad_output_path)
            if apply_noise_reduction:
                logger.info("Applying noise reduction")
                denoiser = AudioDenoiser(sample_rate=16000, prop_reduce=0.8)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32767.0
                samples = np.expand_dims(samples, axis=0)
                # Lấy mẫu nhiễu từ 0.8s đầu
                noise_sample = samples[:, :int(0.8 * 16000)]
                denoised_samples = denoiser.process(samples)
                # Kiểm tra SNR
                signal_power = np.mean(denoised_samples ** 2)
                noise_power = np.mean(noise_sample ** 2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                if snr < 20:
                    logger.warning(f"SNR {snr:.2f} dB below threshold, applying stronger noise reduction")
                    denoised_samples = denoiser.process(samples, prop_reduce=1.0)
                audio = AudioSegment(
                    (denoised_samples[0] * 32767).astype(np.int16).tobytes(),
                    frame_rate=16000,
                    sample_width=2,
                    channels=1
                )
            else:
                logger.info("Skipping noise reduction")

            audio_normalized = audio.normalize()
            audio_normalized.export(output_path, format="wav")
            logger.info(f"Audio processed and saved to: {output_path}")
            os.remove(vad_output_path)
            return output_path

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            if os.path.exists(vad_output_path):
                os.remove(vad_output_path)
            raise Exception(f"Error processing audio: {str(e)}")