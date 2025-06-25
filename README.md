# Automatic-Video-Subtitle-Generation-System
A lightweight and efficient system for automatically generating subtitles for videos using state-of-the-art models like Whisper v3 for speech-to-text transcription and NLLB-200 for multilingual translation.

📌 Overview
This project aims to automate the process of subtitle generation for videos by transcribing audio to text using Whisper v3 and translating subtitles into multiple languages with Meta's NLLB-200 model. The system supports quick, accurate, and customizable subtitle creation, suitable for various video content.

🚀 Features
🎙️ Automatic speech-to-text transcription using Whisper v3
🌍 Subtitle translation into 200+ languages with NLLB-200
🎬 Supports common video formats (mp4, mkv, etc.)
🗂️ Generates .srt or .vtt subtitle files

🛠️ Tech Stack

Python 3.10

OpenAI Whisper v3

Meta NLLB-200

FFmpeg

VAD

FastAPI (optional for building API)

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/video-subtitle-generator.git  
cd video-subtitle-generator  
Create virtual environment (recommended)


▶️ Usage

python api.py --input your_video.mp4 --output subtitles.vtt --lang vi  
--lang specifies the target language for translation (e.g., vi for Vietnamese, en for English).

Subtitles will be generated in the specified language.

🤝 Contributing
Contributions are welcome! Please open an issue or pull request to suggest improvements or new features.

🙋‍♂️ Author
Thanh Nguyen — 2025

