import os
import logging
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from main import AudioTranscriptionService
from config import OUTPUT_FOLDER, LANGUAGE_MAP
import asyncio
import subprocess
import magic

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='[ %(levelname)s ] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", f"api_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory='static'), name='static')
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name='output')

# Khởi tạo service
service = AudioTranscriptionService()

def validate_file(file_path: str) -> dict:
    """Kiểm tra định dạng file và metadata bằng ffprobe."""
    valid_video_formats = (".mp4", ".mkv", ".avi", ".mov")
    valid_audio_formats = (".wav",)
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    
    if not (file_type.startswith("video/") or file_type.startswith("audio/")):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    ext = os.path.splitext(file_path)[1].lower()
    if not (ext in valid_video_formats or ext in valid_audio_formats):
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")
    
    # Kiểm tra stream âm thanh bằng ffprobe
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-select_streams', 'a', '-show_format', file_path],
        capture_output=True, text=True
    )
    if result.returncode != 0 or "stream" not in result.stdout.lower():
        raise HTTPException(status_code=400, detail="No audio stream found in file")
    
    # Trích xuất metadata
    metadata = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    duration = float(metadata.get("duration", 0))
    sample_rate = int(metadata.get("sample_rate", 0))
    
    return {
        "is_video": file_type.startswith("video/"),
        "is_audio": file_type.startswith("audio/"),
        "file_type": file_type,
        "duration": duration,
        "sample_rate": sample_rate
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    logger.info("Serving root page")
    with open("index.html", "r", encoding='utf-8') as f:
        return f.read()

@app.post("/process")
async def process_video(
    file: UploadFile = File(...),
    origin_language: str = Query("vi"),
    translate_language: str = Query("en"),
    embed_subtitle: str = Query("none")
):
    logger.info(f"Processing file: {file.filename}, origin_language={origin_language}, translate_language={translate_language}, embed_subtitle={embed_subtitle}")
    try:
        file_path = os.path.join("/tmp", file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Kiểm tra file
        metadata = validate_file(file_path)
        
        # Chạy xử lý với timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                service.process_single_file,
                audio_path=file_path,
                origin_language=origin_language,
                translate_language=translate_language,
                use_correction=True,  # Luôn sử dụng hiệu chỉnh theo yêu cầu
                embed_subtitle=embed_subtitle,
                metadata=metadata
            ),
            timeout=600
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to process video")

        if "video_url" in result:
            max_attempts = 5
            for attempt in range(max_attempts):
                if os.path.exists(result["video_url"]) and os.path.getsize(result["video_url"]) > 0:
                    logger.info(f"Video file ready after {attempt + 1} attempts: {result['video_url']}")
                    break
                logger.warning(f"Video file not ready (attempt {attempt + 1}/{max_attempts}): {result['video_url']}")
                time.sleep(1)
            else:
                logger.error(f"Video file not ready after {max_attempts} attempts: {result['video_url']}")
                raise HTTPException(status_code=500, detail='Video file not ready')

        response_data = {
            "message": "Processed successfully",
            "subtitle_path": os.path.basename(result["subtitle_path"]),
            "target_language": LANGUAGE_MAP.get(translate_language, translate_language).capitalize(),
            "metadata": metadata
        }
        if "translated_subtitle_path" in result and result["translated_subtitle_path"]:
            response_data["translated_subtitle_path"] = os.path.basename(result["translated_subtitle_path"])

        if "video_url" in result and os.path.exists(result["video_url"]):
            if os.path.getsize(result["video_url"]) > 0:
                response_data["video_with_subtitle"] = True
                response_data["video_url"] = f"/output/{os.path.basename(result['video_url'])}"
                logger.info(f"Video file found and valid at: {result['video_url']} (Size: {os.path.getsize(result['video_url'])} bytes)")
            else:
                logger.warning(f"Video file is empty: {result['video_url']}")
                response_data["video_with_subtitle"] = False
                response_data["message"] += " (Video file created but empty)"
        else:
            response_data["video_with_subtitle"] = False
            if embed_subtitle == "none":
                response_data["message"] += " (No video with subtitle created because embed_subtitle is set to 'none')"
                logger.info("No video with subtitle created because embed_subtitle is set to 'none'")
            else:
                logger.warning("Video with subtitle not created")

        return JSONResponse(content=response_data)

    except asyncio.TimeoutError:
        logger.error(f"Processing timeout for file: {file.filename}")
        raise HTTPException(status_code=504, detail="Processing took too long")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/list_processed_videos")
async def list_processed_videos():
    logger.info("Listing processed videos")
    try:
        videos = []
        video_extensions = (".mp4", ".mkv", ".avi", ".mov")

        for fname in os.listdir(OUTPUT_FOLDER):
            if fname.lower().endswith(video_extensions):
                file_path = os.path.join(OUTPUT_FOLDER, fname)
                base_name = os.path.splitext(fname)[0]
                subtitle_files = [
                    f for f in os.listdir(OUTPUT_FOLDER)
                    if f.startswith(base_name) and f.endswith((".vtt", ".srt"))
                ]

                language = translate_language if translate_language else "unknown"
                subtitle_path = None
                if subtitle_files:
                    subtitle_file = subtitle_files[0]
                    subtitle_path = subtitle_file
                    parts = os.path.splitext(subtitle_file)[0].split('_')
                    if len(parts) >= 3:
                        origin = parts[-2]
                        translate = parts[-1]
                        language = f"origin:{origin},translate:{translate}"

                video_info = {
                    "filename": fname,
                    "path": f"/output/{fname}",
                    "size": os.path.getsize(file_path),
                    "created_at": time.ctime(os.path.getctime(file_path)),
                    "language": language,
                    "subtitle_path": f"/output/{subtitle_path}" if subtitle_path else "none"
                }
                videos.append(video_info)

        if not videos:
            return {"message": "No processed videos found", "videos": []}

        return {"videos": videos}

    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    if filename == "undefined":
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)