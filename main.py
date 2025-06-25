import os
import logging
import time
import json
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import VideoFileClip
import torchaudio

from performance_monitor import PerformanceMonitor
from database import DatabaseHandler
from process_audio import AudioProcessor
from whisper_model_openai import WhisperOpenAITranscriber
from llm_text_service import LLMTextService
from subtitle_converter import SubtitleConverter
from subtitle_embedder import SubtitleEmbedder
from video_splitter import VideoSplitter
from evaluate_metrics import calculate_wer, calculate_bleu
from config import AUDIO_FOLDER, OUTPUT_FOLDER, WHISPER_MODEL, NLLB_MODEL, LANGUAGE_MAP, WHISPER_MODEL_DIR
from video_concatenator import concatenate_videos_ffmpeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ %(levelname)s ] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", f"process_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioTranscriptionService:
    def __init__(self):
        self.perf_monitor = PerformanceMonitor()
        self.db = DatabaseHandler()
        self.transcriber = WhisperOpenAITranscriber(model_name='large-v3.pt', directory=WHISPER_MODEL_DIR)
        self.text_service = LLMTextService(model_name=NLLB_MODEL)
        self.embedder = SubtitleEmbedder()

    def process_segment(self, audio_path: str, origin_language: str, translate_language: str, use_correction: bool, reference_vtt_orig: Optional[str] = None, reference_vtt_trans: Optional[str] = None, timestamp_offset: float = 0.0) -> Dict:
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        result = {}

        self.perf_monitor.start_measurement(f"transcription_{file_name}")
        transcription_result = self.transcriber.transcribe_audio(audio_path, language=origin_language)
        # Ánh xạ timestamp với offset
        for segment in transcription_result:
            segment['start'] += timestamp_offset
            segment['end'] += timestamp_offset
        self.perf_monitor.end_measurement(f"transcription_{file_name}")

        if use_correction:
            self.perf_monitor.start_measurement(f"text_correction_{file_name}")
            lang_code = LANGUAGE_MAP.get(origin_language, 'vietnamese')
            corrected_segments = []
            for segment in transcription_result:
                corrected_text = self.text_service.correct_text(segment['text'], language=lang_code)
                corrected_segments.append({
                    'text': corrected_text,
                    'start': segment['start'],
                    'end': segment['end']
                })
            transcription_result = corrected_segments
            self.perf_monitor.end_measurement(f"text_correction_{file_name}")

        output_vtt = os.path.join(OUTPUT_FOLDER, f"{file_name}_{origin_language}_{origin_language}.vtt")
        vtt_content = SubtitleConverter.generate_vtt_content(transcription_result, timestamp_offset=timestamp_offset)
        SubtitleConverter.save_vtt_file(vtt_content, output_vtt)
        result["subtitle_path"] = output_vtt
        logger.info(f"Generated subtitle file: {output_vtt}")

        if reference_vtt_orig and os.path.exists(reference_vtt_orig):
            self.perf_monitor.start_measurement(f"wer_evaluation_{file_name}")
            lang_code = LANGUAGE_MAP.get(origin_language, 'vietnamese')
            wer_score = calculate_wer(reference_vtt_orig, output_vtt, language=lang_code)
            self.perf_monitor.end_measurement(f"wer_evaluation_{file_name}")
            result["wer_score"] = wer_score
            logger.info(f"WER Score: {wer_score:.4f} for {output_vtt} vs {reference_vtt_orig}")

            wer_output = os.path.join(OUTPUT_FOLDER, f"{file_name}_wer.txt")
            with open(wer_output, "w", encoding='utf-8') as f:
                f.write(f"WER Evaluation ({lang_code.capitalize()})\n")
                f.write(f"WER: {wer_score:.4f}\n")
            logger.info(f"Saved WER results to: {wer_output}")

        if origin_language == translate_language:
            if use_correction:
                self.perf_monitor.start_measurement(f"text_correction_for_same_language_{file_name}")
                lang_code = LANGUAGE_MAP.get(origin_language, 'vietnamese')
                corrected_segments = []
                for segment in transcription_result:
                    corrected_text = self.text_service.correct_text(segment['text'], language=lang_code)
                    corrected_segments.append({
                        'text': corrected_text,
                        'start': segment['start'],
                        'end': segment['end']
                    })
                corrected_vtt_content = SubtitleConverter.generate_vtt_content(corrected_segments, timestamp_offset=timestamp_offset)
                corrected_vtt_path = os.path.join(OUTPUT_FOLDER, f"{file_name}_{origin_language}_{origin_language}_corrected.vtt")
                SubtitleConverter.save_vtt_file(corrected_vtt_content, corrected_vtt_path)
                result["translated_subtitle_path"] = corrected_vtt_path
                self.perf_monitor.end_measurement(f"text_correction_for_same_language_{file_name}")
            else:
                result["translated_subtitle_path"] = output_vtt
                logger.info(f"Using original subtitle as translated file: {output_vtt}")
        else:
            self.perf_monitor.start_measurement(f"translation_{file_name}")
            translated_vtt = self.text_service.translate_vtt(
                input_path=output_vtt,
                output_path=os.path.join(OUTPUT_FOLDER, f"{file_name}_{origin_language}_{translate_language}.vtt"),
                source_language=LANGUAGE_MAP.get(origin_language, 'vietnamese'),
                target_language=LANGUAGE_MAP.get(translate_language, 'english')
            )
            result["translated_subtitle_path"] = translated_vtt
            self.perf_monitor.end_measurement(f"translation_{file_name}")
            logger.info(f"Generated translated subtitle file: {translated_vtt}")

            if reference_vtt_trans and os.path.exists(reference_vtt_trans):
                self.perf_monitor.start_measurement(f"bleu_evaluation_{file_name}")
                lang_code = LANGUAGE_MAP.get(translate_language, 'english')
                bleu_score = calculate_bleu(reference_vtt_trans, translated_vtt, language=lang_code)
                self.perf_monitor.end_measurement(f"bleu_evaluation_{file_name}")
                result["bleu_score"] = bleu_score
                logger.info(f"BLEU-1 Score: {bleu_score:.4f} for {translated_vtt} vs {reference_vtt_trans}")

                bleu_output = os.path.join(OUTPUT_FOLDER, f"{file_name}_bleu.txt")
                with open(bleu_output, "w", encoding='utf-8') as f:
                    f.write(f"BLEU-1 Evaluation ({lang_code.capitalize()})\n")
                    f.write(f"BLEU-1: {bleu_score:.4f}\n")
                logger.info(f"Saved BLEU results to: {bleu_output}")

        return result

    def process_single_file(
        self,
        audio_path: str,
        origin_language: str = 'vi',
        translate_language: str = 'en',
        use_correction: bool = True,
        embed_subtitle: str = 'none',
        reference_vtt_orig: Optional[str] = None,
        reference_vtt_trans: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[Dict]:
        try:
            logger.info(f"Processing file: {audio_path}")
            self.perf_monitor.start_measurement("total_processing")
            file_name = os.path.splitext(os.path.basename(audio_path))[0]
            result = {}

            # Kiểm tra metadata
            file_info = metadata or {}
            input_audio = audio_path

            if file_info.get("is_video"):
                logger.info(f"Converting video to audio: {audio_path}")
                temp_audio = os.path.join(OUTPUT_FOLDER, f"{file_name}_temp.wav")
                video = VideoFileClip(audio_path)
                video.audio.write_audiofile(temp_audio, codec='pcm_s16le')
                video.close()
                input_audio = temp_audio

            # Xử lý âm thanh
            processed_audio = AudioProcessor.process_audio(
                input_audio,
                os.path.join(OUTPUT_FOLDER, f"{file_name}_processed.wav"),
                apply_noise_reduction=False
            )
            original_audio = processed_audio.replace("_processed.wav", "_original.wav")
            if not os.path.exists(original_audio):
                logger.warning(f"Original normalized audio not found: {original_audio}. Fallback to processed.")
                original_audio = processed_audio

            # Kiểm tra độ dài và phân đoạn
            duration = file_info.get("duration", 0) or (VideoFileClip(audio_path).duration if file_info.get("is_video") else torchaudio.info(original_audio).num_frames / 16000)
            audio_paths = [original_audio]
            map_path = None
            if duration > 600:
                logger.info(f"Video duration {duration}s > 10 minutes, splitting...")
                audio_paths, map_path = VideoSplitter.split_video(audio_path if file_info.get("is_video") else original_audio, OUTPUT_FOLDER)

            # Khử nhiễu
            denoised_paths = []
            for audio_path in audio_paths:
                denoised_path = os.path.join(OUTPUT_FOLDER, f"{os.path.basename(audio_path).replace('.wav', '_denoised.wav')}")
                AudioProcessor.process_audio(audio_path, denoised_path, apply_noise_reduction=True)
                denoised_paths.append(denoised_path)

            segment_results = {}
            with ThreadPoolExecutor() as executor:
                futures = []
                if map_path and os.path.exists(map_path):
                    with open(map_path, 'r', encoding='utf-8') as f:
                        timestamp_map = json.load(f)
                else:
                    timestamp_map = {}
                
                for audio_path in denoised_paths:
                    segment_name = os.path.basename(audio_path)
                    timestamp_offset = timestamp_map.get(segment_name, {}).get("start", 0.0)
                    futures.append(
                        executor.submit(
                            self.process_segment,
                            audio_path,
                            origin_language,
                            translate_language,
                            use_correction,
                            reference_vtt_orig,
                            reference_vtt_trans,
                            timestamp_offset
                        )
                    )
                
                for future in as_completed(futures):
                    segment_result = future.result()
                    segment_results.update(segment_result)

            # Kết hợp kết quả
            result["subtitle_paths"] = [segment_results.get("subtitle_path", "")]
            result["translated_subtitle_paths"] = [segment_results.get("translated_subtitle_path", result["subtitle_paths"][0])]
            result["wer_scores"] = [segment_results.get("wer_score")] if "wer_score" in segment_results else []
            result["bleu_scores"] = [segment_results.get("bleu_score")] if "bleu_score" in segment_results else []
            result["map_path"] = map_path

            if embed_subtitle in ['soft', 'hard'] and file_info.get("is_video"):
                self.perf_monitor.start_measurement("subtitle_embedding")
                logger.info(f"Embedding subtitle into video.")
                output_video = os.path.join(OUTPUT_FOLDER, f"{file_name}_subtitled.mp4")
                subtitle_to_embed = result["translated_subtitle_paths"][-1]
                map_path = result.get("map_path")

                embedded_video = self.embedder.embed_subtitle(
                    video_path=audio_path,
                    subtitle_path=subtitle_to_embed,
                    output_path=output_video,
                    embed_type=embed_subtitle,
                    timestamp_map_path=map_path
                )
                if os.path.exists(embedded_video) and os.path.getsize(embedded_video) > 0:
                    result["video_url"] = embedded_video
                    logger.info(f"Successfully embedded subtitle into video: {embedded_video}")
                else:
                    logger.error(f"Video file not created or empty: {embedded_video}")
                    raise RuntimeError("Video file not created or empty")
                self.perf_monitor.end_measurement("subtitle_embedding")
            elif embed_subtitle in ['soft', 'hard'] and not file_info.get("is_video"):
                logger.warning("Subtitle embedding requested but input file is not a video")

            if duration > 600 and embed_subtitle in ['soft', 'hard']:
                segment_video_paths = [os.path.join(OUTPUT_FOLDER, f"{file_name}_segment_{i}_subtitled.mp4") for i in range(len(denoised_paths))]
                final_video_path = os.path.join(OUTPUT_FOLDER, f"{file_name}_subtitled.mp4")
                concatenate_videos_ffmpeg(segment_video_paths, final_video_path)
                result["video_url"] = final_video_path

            self.perf_monitor.end_measurement("total_processing")
            self.perf_monitor.print_summary()
            return result
        except FileNotFoundError as e:
            logger.error(f"File error: {str(e)}")
            return None
        except RuntimeError as e:
            logger.error(f"Embedding error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None

    def process_batch(self):
        audio_files = self.db.get_pending_files()
        logger.info(f"Processing {len(audio_files)} audio files...")
        for audio in audio_files:
            try:
                self.db.update_status(audio['id'], 'PROCESSING')
                audio_path = os.path.join(AUDIO_FOLDER, audio['file_name'])

                reference_vtt_orig = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(audio['file_name'])[0]}_ref_{audio['origin_language']}.vtt")
                reference_vtt_trans = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(audio['file_name'])[0]}_ref_{audio['translate_language']}.vtt")

                result = self.process_single_file(
                    audio_path=audio_path,
                    origin_language=audio['origin_language'],
                    translate_language=audio['translate_language'],
                    use_correction=audio['use_correction'],
                    embed_subtitle=audio['embed_subtitle'],
                    reference_vtt_orig=reference_vtt_orig if os.path.exists(reference_vtt_orig) else None,
                    reference_vtt_trans=reference_vtt_trans if os.path.exists(reference_vtt_trans) else None
                )
                if result:
                    self.db.update_status(audio['id'], 'COMPLETED')
                    self.db.update_result(audio['id'], result)
                else:
                    self.db.update_status(audio['id'], 'FAILED')
            except Exception as e:
                logger.error(f"Error processing {audio['file_name']}: {str(e)}")
                self.db.update_status(audio['id'], 'FAILED')

def main():
    service = AudioTranscriptionService()
    audio_path = "/data/datn/DATN_chat/data/input/Mylove_ch_ch.mp4"
    reference_vtt_orig = "/data/datn/DATN_chat/data/original_subtitle/Mylove_ch_ch.vtt"
    reference_vtt_trans = "/data/datn/DATN_chat/data/original_translate/Mylove_ch_en.vtt"

    result = service.process_single_file(
        audio_path=audio_path,
        origin_language='zh',
        translate_language='en',
        use_correction=True,
        embed_subtitle='soft',
        reference_vtt_orig=reference_vtt_orig if os.path.exists(reference_vtt_orig) else None,
        reference_vtt_trans=reference_vtt_trans if os.path.exists(reference_vtt_trans) else None
    )
    if result:
        logger.info(f"Processed successfully: {result}")

if __name__ == "__main__":
    main()