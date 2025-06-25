import os
import subprocess
import logging
import re
import json
from typing import Optional

logger = logging.getLogger(__name__)

class SubtitleEmbedder:
    @staticmethod
    def embed_subtitle(video_path: str, subtitle_path: str, output_path: str, embed_type: str = "soft", timestamp_map_path: Optional[str] = None) -> str:
        try:
            video_file_path = os.path.abspath(video_path)
            subtitle_file_path = os.path.abspath(subtitle_path)
            output_file_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            if not os.path.exists(video_file_path):
                logger.error(f"Video file not found: {video_file_path}")
                raise FileNotFoundError(f"Video file not found: {video_file_path}")
            if not os.path.exists(subtitle_file_path):
                logger.error(f"Subtitle file not found: {subtitle_file_path}")
                raise FileNotFoundError(f"Subtitle file not found: {subtitle_file_path}")

            temp_srt_path = None
            if subtitle_file_path.endswith('.vtt') and embed_type == "hard":
                temp_srt_path = subtitle_file_path.replace('.vtt', '_temp.srt')
                with open(subtitle_file_path, 'r', encoding='utf-8') as vtt_file:
                    lines = vtt_file.readlines()

                srt_content = []
                index = 1
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line == "WEBVTT" or not line:
                        i += 1
                        continue
                    if '-->' in line:
                        time_line = line.strip()
                        start_time, end_time = re.split(r'\s*-->\s*', time_line)
                        start_time = start_time.replace('.', ',')
                        end_time = end_time.replace('.', ',')
                        srt_content.append(str(index))
                        srt_content.append(f"{start_time} --> {end_time}")
                        i += 1
                        text_lines = []
                        while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                            text_lines.append(lines[i].strip())
                            i += 1
                        srt_content.append('\n'.join(text_lines))
                        srt_content.append('')
                        index += 1
                    else:
                        i += 1

                with open(temp_srt_path, 'w', encoding='utf-8') as srt_file:
                    srt_file.write('\n'.join(srt_content))
                subtitle_file_path = temp_srt_path
                logger.info(f"Converted .vtt to .srt: {temp_srt_path}")

            subtitle_file_path = subtitle_file_path.replace("'", "'\\''")
            if embed_type == "soft":
                cmd = [
                    "ffmpeg", "-i", video_file_path, "-i", subtitle_file_path,
                    "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng", "-y", output_file_path
                ]
            else:  # hard
                cmd = [
                    "ffmpeg", "-i", video_file_path,
                    "-vf", f"subtitles='{subtitle_file_path}':force_style='Fontsize=24,PrimaryColour=&HFFFFFF&,FontName=DejaVu Sans'",
                    "-c:v", "libx264", "-c:a", "aac", "-y", output_file_path
                ]

            logger.info(f"Embedding subtitle with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding="utf-8")
            logger.info(f"FFmpeg stdout: {result.stdout}")
            if result.stderr:
                logger.error(f"FFmpeg stderr: {result.stderr}")

            if result.returncode != 0 or not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
                logger.error(f"Embedding failed. Return code: {result.returncode}, Output file not found or empty: {output_file_path}")
                raise RuntimeError(f"FFmpeg failed to embed subtitle. Error: {result.stderr}")

            # Kiểm tra tính toàn vẹn
            check_cmd = ['ffprobe', '-v', 'error', '-show_streams', '-show_format', output_file_path]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            if check_result.returncode != 0:
                logger.error("Video integrity check failed")
                raise RuntimeError("Output video integrity check failed")

            logger.info(f"Successfully embedded subtitle to: {output_file_path}")

            if temp_srt_path and os.path.exists(temp_srt_path):
                try:
                    os.unlink(temp_srt_path)
                    logger.info(f"Deleted temporary SRT file: {temp_srt_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary SRT file {temp_srt_path}: {str(e)}")

            return output_file_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"FFmpeg failed to embed subtitle: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error in embed_subtitle: {str(e)}")
            raise RuntimeError(f"Error embedding subtitle: {str(e)}")