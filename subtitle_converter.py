import logging
import textwrap
import json
import os
from typing import Optional
import pysubs2

logger = logging.getLogger(__name__)

class SubtitleConverter:
    @staticmethod
    def generate_vtt_content(transcription_result: list, min_duration: float = 1.5, max_duration: float = 6.0, appear_offset: float = 0.5, linger_offset: float = 1.0, timestamp_offset: float = 0.0) -> str:
        try:
            vtt_content = "WEBVTT\n\n"
            for i, segment in enumerate(transcription_result):
                start = float(segment.get("start", 0.0)) + appear_offset + timestamp_offset
                end = float(segment.get("end", 0.0)) + linger_offset + timestamp_offset
                text = segment.get("text", "").strip()

                if not text:
                    continue

                start = max(0.0, start)
                duration = end - start
                if duration < min_duration:
                    end = start + min_duration
                elif duration > max_duration:
                    end = start + max_duration

                if i + 1 < len(transcription_result):
                    next_start = float(transcription_result[i + 1]['start']) + appear_offset + timestamp_offset
                    if end > next_start:
                        end = next_start - 0.1

                if end <= start:
                    continue

                timestamp = f"{SubtitleConverter._seconds_to_timestamp(start)} --> {SubtitleConverter._seconds_to_timestamp(end)}"
                wrapped_lines = textwrap.wrap(text, width=40)
                wrapped_text = '\n'.join(wrapped_lines)
                vtt_content += f"{timestamp}\n{wrapped_text}\n\n"

            logger.info(f"Generated VTT content with {len(transcription_result)} segments")
            return vtt_content
        except Exception as e:
            logger.error(f"Error generating VTT content: {str(e)}")
            raise Exception(f"Error generating VTT content: {str(e)}")

    @staticmethod
    def generate_srt_content(transcription_result: list, min_duration: float = 1.5, max_duration: float = 6.0, appear_offset: float = 0.5, linger_offset: float = 1.0, timestamp_offset: float = 0.0) -> str:
        try:
            srt_content = ""
            for i, segment in enumerate(transcription_result):
                start = float(segment.get("start", 0.0)) + appear_offset + timestamp_offset
                end = float(segment.get("end", 0.0)) + linger_offset + timestamp_offset
                text = segment.get("text", "").strip()

                if not text:
                    continue

                start = max(0.0, start)
                duration = end - start
                if duration < min_duration:
                    end = start + min_duration
                elif duration > max_duration:
                    end = start + max_duration

                if i + 1 < len(transcription_result):
                    next_start = float(transcription_result[i + 1]['start']) + appear_offset + timestamp_offset
                    if end > next_start:
                        end = next_start - 0.1

                if end <= start:
                    continue

                start_ts = SubtitleConverter._seconds_to_timestamp(start).replace('.', ',')
                end_ts = SubtitleConverter._seconds_to_timestamp(end).replace('.', ',')
                wrapped_lines = textwrap.wrap(text, width=40)
                wrapped_text = '\n'.join(wrapped_lines)
                srt_content += f"{i+1}\n{start_ts} --> {end_ts}\n{wrapped_text}\n\n"

            logger.info(f"Generated SRT content with {len(transcription_result)} segments")
            return srt_content
        except Exception as e:
            logger.error(f"Error generating SRT content: {str(e)}")
            raise Exception(f"Error generating SRT content: {str(e)}")

    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        parts = timestamp.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def save_vtt_file(content: str, output_path: str, timestamp_map_path: Optional[str] = None, segment_name: Optional[str] = None) -> None:
        try:
            if timestamp_map_path and segment_name and os.path.exists(timestamp_map_path):
                with open(timestamp_map_path, "r", encoding="utf-8") as f:
                    timestamp_map = json.load(f)
                offset = timestamp_map.get(segment_name, {}).get("start", 0.0)
            else:
                offset = 0.0

            subs = pysubs2.load(output_path, format='vtt') if os.path.exists(output_path) else pysubs2.SSAFile.from_string(content, format='vtt')
            for event in subs.events:
                event.start -= 0.5
                event.end += 1.0
                event.start += offset
                event.end += offset

            subs.sort()
            for i in range(len(subs.events) - 1):
                if subs.events[i].end > subs.events[i + 1].start:
                    logger.warning(f"Overlap detected between {subs.events[i].start} and {subs.events[i + 1].start}")
                    subs.events[i].end = subs.events[i + 1].start - 0.1

            with open(output_path, "w", encoding="utf-8") as f:
                subs.save(output_path, format='vtt')
            logger.info(f"VTT file saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving VTT file: {str(e)}")
            raise Exception(f"Error saving VTT file: {str(e)}")

    @staticmethod
    def save_srt_file(content: str, output_path: str, timestamp_map_path: Optional[str] = None, segment_name: Optional[str] = None) -> None:
        try:
            if timestamp_map_path and segment_name and os.path.exists(timestamp_map_path):
                with open(timestamp_map_path, "r", encoding="utf-8") as f:
                    timestamp_map = json.load(f)
                offset = timestamp_map.get(segment_name, {}).get("start", 0.0)
            else:
                offset = 0.0

            subs = pysubs2.load(output_path, format='srt') if os.path.exists(output_path) else pysubs2.SSAFile.from_string(content, format='srt')
            for event in subs.events:
                event.start -= 0.5
                event.end += 1.0
                event.start += offset
                event.end += offset

            subs.sort()
            for i in range(len(subs.events) - 1):
                if subs.events[i].end > subs.events[i + 1].start:
                    logger.warning(f"Overlap detected between {subs.events[i].start} and {subs.events[i + 1].start}")
                    subs.events[i].end = subs.events[i + 1].start - 0.1

            with open(output_path, "w", encoding="utf-8") as f:
                subs.save(output_path, format='srt')
            logger.info(f"SRT file saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving SRT file: {str(e)}")
            raise Exception(f"Error saving SRT file: {str(e)}")