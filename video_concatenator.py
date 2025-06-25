import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def concatenate_videos_ffmpeg(video_paths: list, output_path: str) -> str:
    try:
        logger.info(f"Concatenating {len(video_paths)} video segments to {output_path}")
        if not video_paths:
            raise ValueError("No video paths provided")

        # Tạo danh sách file cho ffmpeg
        playlist_path = os.path.join(os.path.dirname(output_path), "playlist.txt")
        with open(playlist_path, "w") as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")

        # Nối video bằng ffmpeg
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', playlist_path,
            '-c', 'copy', '-y', output_path
        ]
        subprocess.run(cmd, check=True)

        # Kiểm tra tính toàn vẹn bằng ffprobe
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', output_path], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Video integrity check failed")
            raise Exception("Output video integrity check failed")

        logger.info(f"Concatenated video saved to: {output_path}")
        os.remove(playlist_path)
        return output_path
    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        raise RuntimeError(f"Error concatenating videos: {str(e)}")