import os

# Input and output directories
AUDIO_FOLDER = os.path.join("data", "input")
OUTPUT_FOLDER = os.path.join("data", "output")

# Whisper model configuration
WHISPER_MODEL = "large-v3.pt"
WHISPER_MODEL_DIR = "/data/datn/models"

# NLLB model configuration
NLLB_MODEL = "/data/datn/models/nllb-200-distilled-600M"

# T5 model configuration
T5_MODEL = "/data/datn/models/t5-small"  # Sử dụng t5-small

# SileroVAD model configuration
SILERO_VAD_DIR = "/data/datn/models/silero_vad"

# Language code mapping
LANGUAGE_MAP = {
    'vi': 'vietnamese',
    'en': 'english',
    'ko': 'korean',
    'zh': 'chinese'
}