import os
from pprint import pprint
import torch
from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    collect_chunks
)
import torchaudio
import torchaudio.transforms as T


class SileroVADProcessor:
    def __init__(self, sampling_rate: int = 16000, use_onnx: bool = False):
        self.sampling_rate = sampling_rate
        self.model = load_silero_vad(onnx=use_onnx)
        torch.set_num_threads(1)
    
    def covert_to_clip_timestamp_str(self, speech_timestamps: list) -> str:
        """
        Chuyển đổi danh sách các timestamp thành chuỗi định dạng "start1,end1,start2,end2,..."
        """
        flat_list = []
        for seg in speech_timestamps:
            flat_list.extend([round(seg["start"], 2), round(seg["end"], 2)])
        
        timestamps_in_seconds = [round(s / self.sampling_rate, 2) for s in flat_list]

        return ",".join(str(x) for x in timestamps_in_seconds)
    
    def save_output(self, input_path: str, output_path: str, speech_timestamps: str):
        """
        Lưu kết quả vào file
        """
        # Đọc toàn bộ audio
        waveform, sr = torchaudio.load(input_path)
        # Chuyển đổi nếu khác
        if sr != self.sampling_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # Cắt các đoạn và ghép lại
        chunks = []
        for seg in speech_timestamps:
            start, end = seg["start"], seg["end"]
            chunk = waveform[:, (int(start) - 1):(int(end) + 1)]
            chunks.append(chunk)

        # Ghép lại thành 1 đoạn duy nhất
        merged_waveform = torch.cat(chunks, dim=1)  # ghép theo chiều thời gian

        # Lưu file đã ghép
        torchaudio.save(output_path, merged_waveform, sample_rate=self.sampling_rate)
        print(f"✅ Saved merged audio to: {output_path}")

    def process(self, input_path: str, output_path: str, show_timestamps: bool = True) -> list:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"📥 Reading audio: {input_path}")
        wav = read_audio(input_path, sampling_rate=self.sampling_rate)

        print("🎯 Detecting speech timestamps...")
        speech_timestamps = get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate)

        if not speech_timestamps:
            print("⚠️ No speech detected.")
            return []
        
        self.save_output(input_path, output_path, speech_timestamps)

        return self.covert_to_clip_timestamp_str(speech_timestamps)

# if __name__ == "__main__":
#     input_file = "/data/cuongdd/ai_service_copy/model/denoise/mytv_VOD_tiengviet2_chuanhoa_deepfilter.wav"
#     output_file = "/data/cuongdd/ai_service_copy/model/vad_segments/mytv_VOD_tiengviet2_chuanhoa_deepfilter_silero.wav"

#     vad_processor = SileroVADProcessor(sampling_rate=16000, use_onnx=False)
#     result = vad_processor.process(input_file, output_file)

#     print("\n✅ clip_timestamps dùng cho Whisper:")
#     print(result)
