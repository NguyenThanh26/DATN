import whisper
import logging
import torch
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning, module='whisper')

logger = logging.getLogger(__name__)

class WhisperOpenAITranscriber:
    def __init__(self, model_name='large-v3.pt', directory='/data/datn/models'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model {model_name} from directory {directory} on {self.device}")
        model_file = os.path.join(directory, model_name)

        if not os.path.exists(model_file):
            logger.error(f"Model file not found at {model_file}")
            raise FileNotFoundError(f"Model weights not found at {model_file}")

        self.model = whisper.load_model(model_file, device=self.device)
        logger.info(f"Whisper model successfully loaded from {model_file}")

    def transcribe_audio(self, audio_path: str, language: str = "vi") -> list:
        try:
            logger.info(f"Transcribing audio: {audio_path} with language {language}")
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                temperature=0.0
            )
            segments = result["segments"]
            transcribed_text = [
                {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                }
                for segment in segments
            ]
            logger.info(f"Transcription completed with {len(transcribed_text)} segments")
            return transcribed_text
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise Exception(f"Error transcribing audio: {str(e)}")