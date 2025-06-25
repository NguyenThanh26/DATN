import os
import json
import logging
from moviepy.editor import VideoFileClip
from vad_speeched import SileroVADProcessor

logger = logging.getLogger(__name__)

class VideoSplitter:
    MIN_SEGMENT_DURATION = 300  # 5 phút
    MAX_SEGMENT_DURATION = 600  # 10 phút
    MIN_SILENCE_DURATION_MS = 500  # 500ms

    @staticmethod
    def split_video(video_path: str, output_dir: str) -> tuple[list, str]:
        try:
            logger.info(f"Analyzing video: {video_path}")
            os.makedirs(output_dir, exist_ok=True)
            video = VideoFileClip(video_path)
            duration = video.duration

            if duration <= VideoSplitter.MAX_SEGMENT_DURATION:
                logger.info(f"Video duration {duration}s is less than 5 minutes, no splitting needed")
                output_path = os.path.join(output_dir, f"full_{os.path.basename(video_path)}.wav")
                video.audio.write_audiofile(output_path, codec='pcm_s16le')
                video.close()
                return [output_path], None

            # Phát hiện khoảng im lặng
            temp_audio = os.path.join(output_dir, "temp_audio.wav")
            video.audio.write_audiofile(temp_audio, codec='pcm_s16le')
            vad_processor = SileroVADProcessor()
            speech_timestamps = vad_processor.process(temp_audio, temp_audio.replace(".wav", "_vad.wav"))
            os.remove(temp_audio)

            silence_intervals = []
            last_end = 0
            for ts in speech_timestamps:
                if ts['start'] - last_end >= VideoSplitter.MIN_SILENCE_DURATION_MS:
                    silence_intervals.append((last_end / 1000, ts['start'] / 1000))
                last_end = ts['end']
            if last_end / 1000 < duration:
                silence_intervals.append((last_end / 1000, duration))

            # Tìm điểm cắt
            cut_points = [0]
            current_time = 0
            for silence_start, silence_end in silence_intervals:
                if silence_end - current_time >= VideoSplitter.MIN_SEGMENT_DURATION and silence_end <= VideoSplitter.MAX_SEGMENT_DURATION:
                    cut_points.append(silence_end)
                    current_time = silence_end
                elif silence_end - current_time > VideoSplitter.MAX_SEGMENT_DURATION:
                    cut_points.append(current_time + VideoSplitter.MAX_SEGMENT_DURATION)
                    current_time += VideoSplitter.MAX_SEGMENT_DURATION
            if current_time < duration:
                cut_points.append(duration)

            # Cắt video và lưu timestamp
            audio_paths = []
            timestamp_map = {}
            for i in range(len(cut_points) - 1):
                start_time = cut_points[i]
                end_time = min(cut_points[i + 1], duration)
                output_path = os.path.join(output_dir, f"segment_{i}_{start_time}_{end_time}.wav")
                video_segment = video.subclip(start_time, end_time)
                video_segment.audio.write_audiofile(output_path, codec='pcm_s16le')
                audio_paths.append(output_path)
                timestamp_map[os.path.basename(output_path)] = {"start": start_time, "end": end_time}
                logger.info(f"Extracted audio segment: {output_path}")

            video.close()

            # Lưu timestamp map
            map_path = os.path.join(output_dir, "map.json")
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(timestamp_map, f, indent=2)
            logger.info(f"Saved timestamp map to: {map_path}")

            return audio_paths, map_path
        except Exception as e:
            logger.error(f"Error splitting video: {str(e)}")
            raise Exception(f"Error splitting video: {str(e)}")