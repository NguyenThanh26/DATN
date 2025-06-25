import os
from pydub import AudioSegment
try:
    from rnnoise_wrapper import RNNoise
except ModuleNotFoundError:
    print("Lỗi: rnnoise không được cài đặt. Chạy 'pip install rnnoise' trong môi trường denoise.")
    exit(1)
import soundfile as sf

# Cấu hình ffmpeg cho pydub
AudioSegment.converter = "/usr/bin/ffmpeg"  # Thay bằng đường dẫn ffmpeg, nếu cần

def preprocess_audio(input_file, output_file):
    """Lọc nhiễu âm thanh bằng RNNoise và lưu vào output_file."""
    # Chuẩn hóa đường dẫn
    input_file = os.path.normpath(input_file)
    output_file = os.path.normpath(output_file)
    
    # Kiểm tra tệp tồn tại
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Tệp {input_file} không tồn tại")
    
    # Chuyển đổi MP3 sang WAV nếu cần
    if input_file.lower().endswith('.mp3'):
        try:
            audio = AudioSegment.from_mp3(input_file)
            temp_wav = os.path.splitext(input_file)[0] + "_temp.wav"
            audio.export(temp_wav, format="wav")
            input_file = temp_wav
        except Exception as e:
            raise RuntimeError(f"Lỗi khi chuyển đổi MP3 sang WAV: {e}")
    
    # Đọc tệp âm thanh
    try:
        audio_data, sample_rate = sf.read(input_file)
    except Exception as e:
        raise RuntimeError(f"Lỗi khi đọc tệp âm thanh {input_file}: {e}")
    
    # Khởi tạo RNNoise
    try:
        denoiser = RNNoise()
    except Exception as e:
        raise RuntimeError(f"Lỗi khi khởi tạo RNNoise: {e}")
    
    # Lọc nhiễu
    try:
        clean_audio = denoiser.denoise(audio_data)
    except Exception as e:
        raise RuntimeError(f"Lỗi khi lọc nhiễu: {e}")
    
    # Lưu âm thanh sạch
    try:
        sf.write(output_file, clean_audio, sample_rate)
        print(f"Đã lưu âm thanh sạch tại: {output_file}")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi lưu tệp {output_file}: {e}")
    
    # Xóa tệp tạm nếu có
    if input_file.lower().endswith('_temp.wav'):
        try:
            os.remove(input_file)
        except OSError as e:
            print(f"Lỗi khi xóa tệp tạm {input_file}: {e}")

# # Sử dụng
# input_audio = "/data/cuongdd/ai_service/data/input/VOD Phim tieng Anh.wav"  # Thay bằng đường dẫn tệp
# output_clean_audio = "./clean_vi_audio.wav"

# try:
#     preprocess_audio(input_audio, output_clean_audio)
# except FileNotFoundError as e:
#     print(e)
# except Exception as e:
#     print(f"Lỗi: {type(e).__name__}, {e}")