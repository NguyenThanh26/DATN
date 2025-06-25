# download_model.py
import whisper
import logging

logger = logging.getLogger(__name__)

def download_model(model_name='large-v3', save_dir='/data/datn/models'):
    logger.info(f"Downloading Whisper model {model_name} to {save_dir}")
    model = whisper.load_model(model_name, download_root=save_dir)
    logger.info(f"Model {model_name} downloaded successfully.")
    return model


if __name__ == "__main__":
    # Tải mô hình nếu chưa có
    download_model("large-v3", "/data/datn/models")
