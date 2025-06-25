import logging
import re
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
import torch
from config import NLLB_MODEL, T5_MODEL, LANGUAGE_MAP

logger = logging.getLogger(__name__)

class LLMTextService:
    def __init__(self, model_name=NLLB_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading NLLB model from {model_name} on {self.device}...")
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"NLLB model directory not found at {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(self.device)
            logger.info(f"NLLB model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {str(e)}")
            raise RuntimeError(f"Failed to load NLLB model: {str(e)}")

        logger.info(f"Loading T5 model from {T5_MODEL}...")
        if not os.path.exists(T5_MODEL):
            logger.info(f"T5 model not found at {T5_MODEL}. Downloading t5-small...")
            try:
                os.makedirs(T5_MODEL, exist_ok=True)
                t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
                t5_tokenizer.save_pretrained(T5_MODEL)
                t5_model.save_pretrained(T5_MODEL)
                logger.info(f"T5 model downloaded and saved to {T5_MODEL}")
            except Exception as e:
                logger.error(f"Failed to download and save T5 model: {str(e)}")
                raise RuntimeError(f"Failed to download and save T5 model: {str(e)}")
        
        try:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL, local_files_only=True, legacy=False)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL, local_files_only=True).to(self.device)
            logger.info(f"T5 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load T5 model: {str(e)}")
            raise RuntimeError(f"Failed to load T5 model: {str(e)}")

        self.language_to_nllb_code = {
            "vietnamese": "vie_Latn",
            "english": "eng_Latn",
            "korean": "kor_Hang",
            "chinese": "zho_Hans",
        }

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        try:
            src_lang_name = LANGUAGE_MAP.get(src_lang, src_lang)
            tgt_lang_name = LANGUAGE_MAP.get(tgt_lang, tgt_lang)

            src_code = self.language_to_nllb_code.get(src_lang_name, "eng_Latn")
            tgt_code = self.language_to_nllb_code.get(tgt_lang_name, "vie_Latn")

            logger.info(f"Translating text from {src_code} to {tgt_code}")

            inputs = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=200,
                num_beams=5
            )
            translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            logger.info(f"Translation completed: {translated_text}")

            return translated_text
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise Exception(f"Error translating text: {str(e)}")

    def correct_text(self, text: str, language: str) -> str:
        try:
            logger.info(f"Correcting text with T5 for language: {language}")
            input_text = f"correct: {text}"
            inputs = self.t5_tokenizer(input_text, return_tensors="pt").to(self.device)
            corrected = self.t5_model.generate(**inputs, max_length=200, num_beams=4)
            corrected_text = self.t5_tokenizer.decode(corrected[0], skip_special_tokens=True)
            logger.info(f"Text correction completed: {corrected_text}")
            return corrected_text.strip()
        except Exception as e:
            logger.error(f"Error correcting text with T5: {str(e)}")
            raise Exception(f"Error correcting text: {str(e)}")

    def translate_vtt(self, input_path: str, output_path: str, source_language: str, target_language: str) -> str:
        try:
            logger.info(f"Translating VTT file from {input_path} to {output_path}")

            with open(input_path, 'r', encoding='utf-8') as f:
                vtt_content = f.readlines()

            translated_lines = ['WEBVTT\n']

            current_segment = []
            is_timing_line = False

            for line in vtt_content[1:]:
                line = line.strip()
                if not line:
                    if current_segment:
                        text_to_translate = ' '.join(current_segment)
                        corrected_text = self.correct_text(text_to_translate, source_language)
                        translated_text = self.translate_text(corrected_text, source_language, target_language)
                        translated_lines.append(translated_text)
                        current_segment = []
                        translated_lines.append('')
                    is_timing_line = False
                elif re.match(r'\d\d:\d\d:\d\d\.\d\d\d --> \d\d:\d\d:\d\d\.\d\d\d', line):
                    if current_segment:
                        text_to_translate = ' '.join(current_segment)
                        corrected_text = self.correct_text(text_to_translate, source_language)
                        translated_text = self.translate_text(corrected_text, source_language, target_language)
                        translated_lines.append(translated_text)
                        current_segment = []
                    translated_lines.append(line)
                    is_timing_line = True
                else:
                    if is_timing_line:
                        current_segment.append(line)
                    else:
                        translated_lines.append(line)

            if current_segment:
                text_to_translate = ' '.join(current_segment)
                corrected_text = self.correct_text(text_to_translate, source_language)
                translated_text = self.translate_text(corrected_text, source_language, target_language)
                translated_lines.append(translated_text)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translated_lines) + '\n')

            logger.info(f"Translated VTT saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error translating VTT: {str(e)}")
            raise Exception(f"Error translating VTT: {str(e)}")