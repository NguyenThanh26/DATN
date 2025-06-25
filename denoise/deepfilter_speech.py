import os
import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment

def split_and_enhance(input_file, output_file, segment_length_ms=10000, device="cpu"):
    """Cắt tệp âm thanh thành đoạn, lọc nhiễu, và gộp lại."""
    try:
        # Chuẩn hóa đường dẫn
        input_file = os.path.normpath(input_file)
        output_file = os.path.normpath(output_file)
        
        # Kiểm tra tệp tồn tại
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Tệp {input_file} không tồn tại")
        
        # Load model
        model, df_state, _ = init_df()
        print("Model loaded successfully")

        # Đọc tệp âm thanh bằng pydub
        audio = AudioSegment.from_file(input_file)
        print(f"Total duration: {len(audio)/1000} seconds")

        # Tạo thư mục tạm
        temp_dir = os.path.join(os.path.dirname(output_file), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)

        # Cắt thành các đoạn
        segment_files = []
        for i in range(0, len(audio), segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            segment_file = os.path.join(temp_dir, f"segment_{i//1000}.wav")
            segment.export(segment_file, format="wav")
            segment_files.append(segment_file)

        # Lọc nhiễu từng đoạn
        enhanced_segments = []
        for segment_file in segment_files:
            try:
                # Load đoạn âm thanh
                audio_segment, _ = load_audio(segment_file, sr=df_state.sr())
                audio_segment = audio_segment.contiguous().to(device)
                
                # Lọc nhiễu
                enhanced = enhance(model, df_state, audio_segment)
                enhanced = enhanced.contiguous()
                
                # Lưu đoạn sạch tạm
                enhanced_file = os.path.join(temp_dir, f"enhanced_{os.path.basename(segment_file)}")
                save_audio(enhanced_file, enhanced, df_state.sr())
                enhanced_segments.append(enhanced_file)
                
                print(f"Processed segment: {segment_file}")
            except Exception as e:
                print(f"Lỗi khi xử lý {segment_file}: {e}")
                continue

        # Gộp các đoạn sạch
        final_audio = AudioSegment.empty()
        for enhanced_file in enhanced_segments:
            segment_audio = AudioSegment.from_wav(enhanced_file)
            final_audio += segment_audio

        # Lưu tệp cuối cùng
        final_audio.export(output_file, format="wav")
        print(f"Đã lưu âm thanh sạch tại: {output_file}")

        # Xóa tệp tạm
        for file in segment_files + enhanced_segments:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Lỗi khi xóa {file}: {e}")
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"Lỗi khi xóa thư mục {temp_dir}: {e}")

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except RuntimeError as e:
        print(f"Lỗi khi xử lý âm thanh: {e}")
    except Exception as e:
        print(f"Lỗi khác: {type(e).__name__}, {e}")

# # Sử dụng
# input_audio = "/data/cuongdd/ai_service/data/input/mytv_VOD_tiengviet2_chuanhoa.wav"
# output_clean_audio = "/data/cuongdd/ai_service_copy/model/denoise/mytv_VOD_tiengviet2_chuanhoa_deepfilter.wav"

# split_and_enhance(input_audio, output_clean_audio, segment_length_ms=300000, device="cpu")