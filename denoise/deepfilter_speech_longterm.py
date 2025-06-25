import os
import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment

def split_and_enhance(input_file, output_file, segment_length_ms=10000, overlap_ms=500, device="cpu"):
    """Cắt tệp âm thanh thành đoạn có chồng lấn, lọc nhiễu, và gộp lại mượt mà."""
    try:
        input_file = os.path.normpath(input_file)
        output_file = os.path.normpath(output_file)

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Tệp {input_file} không tồn tại")

        # Load model
        model, df_state, _ = init_df()
        print("Model loaded successfully")

        # Load full audio
        audio = AudioSegment.from_file(input_file)
        print(f"Total duration: {len(audio) / 1000:.1f} seconds")

        temp_dir = os.path.join(os.path.dirname(output_file), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)

        segment_files = []
        step = segment_length_ms - overlap_ms
        for i in range(0, len(audio), step):
            segment = audio[i:i + segment_length_ms]
            segment_file = os.path.join(temp_dir, f"segment_{i//1000}.wav")
            segment.export(segment_file, format="wav")
            segment_files.append((segment_file, i))

        enhanced_segments = []
        for segment_file, start_ms in segment_files:
            try:
                audio_segment, _ = load_audio(segment_file, sr=df_state.sr())
                audio_segment = audio_segment.contiguous().to(device)
                enhanced = enhance(model, df_state, audio_segment)
                enhanced = enhanced.contiguous()
                enhanced_file = os.path.join(temp_dir, f"enhanced_{os.path.basename(segment_file)}")
                save_audio(enhanced_file, enhanced, df_state.sr())
                enhanced_segments.append((enhanced_file, start_ms))
                print(f"Processed: {segment_file}")
            except Exception as e:
                print(f"Lỗi xử lý {segment_file}: {e}")
                continue

        # Ghép với xử lý overlap
        final_audio = AudioSegment.empty()
        prev_segment = None
        for idx, (enhanced_file, start_ms) in enumerate(enhanced_segments):
            segment_audio = AudioSegment.from_wav(enhanced_file)

            if idx == 0:
                final_audio += segment_audio
                prev_segment = segment_audio
                continue

            # Cross-fade phần chồng lấn
            overlap_region = segment_audio[:overlap_ms].fade_in(overlap_ms)
            prev_overlap = prev_segment[-overlap_ms:].fade_out(overlap_ms)

            mixed_overlap = prev_overlap.overlay(overlap_region)

            # Ghép phần trước (không chồng lấn) + phần trộn
            final_audio = final_audio[:-overlap_ms] + mixed_overlap
            final_audio += segment_audio[overlap_ms:]

            prev_segment = segment_audio

        final_audio.export(output_file, format="wav")
        print(f"Đã lưu âm thanh sạch tại: {output_file}")

        # Cleanup
        for file, _ in enhanced_segments + segment_files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Lỗi khi xóa {file}: {e}")
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"Lỗi khi xóa thư mục {temp_dir}: {e}")

    except Exception as e:
        print(f"Lỗi: {type(e).__name__}, {e}")

# Dùng thử
# /data/cuongdd/ai_service/data/input/VOD Phim tieng Anh.wav
# /data/cuongdd/ai_service/data/input/Clip the thao - VOD tieng Viet.wav
# /data/cuongdd/ai_service/data/input/mytv_VOD_tiengviet2_chuanhoa.wav
# /data/cuongdd/ai_service/data/input/mytv_VOD_tienganh2.wav

# input_audio = "/data/cuongdd/ai_service/data/input/mytv_VOD_tienganh2.wav"
# output_clean_audio = "/data/cuongdd/ai_service_copy/model/denoise/mytv_VOD_tienganh2_deppfilter_longterm.wav"
# split_and_enhance(input_audio, output_clean_audio, segment_length_ms=300000, overlap_ms=1000, device="cpu")
