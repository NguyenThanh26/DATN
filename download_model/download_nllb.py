from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = "facebook/nllb-200-distilled-600M"
save_dir = "/data/datn/models/nllb-200-distilled-600M"
token = "your_token"  # Thay bằng token của bạn

try:
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Tạo thư mục {save_dir}")

    # Tải mô hình
    logger.info(f"Tải mô hình từ {model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype="float16",
        low_cpu_mem_usage=True,
        token=token
    )
    model.save_pretrained(save_dir)
    logger.info(f"Đã lưu mô hình vào {save_dir}")

    # Tải tokenizer
    logger.info(f"Tải tokenizer từ {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Đã lưu tokenizer vào {save_dir}")

    print(f"Mô hình NLLB-200-distilled-600M đã được lưu vào {save_dir}")
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình: {str(e)}")
    raise