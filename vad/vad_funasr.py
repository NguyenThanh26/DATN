from funasr import AutoModel
import os
import torchaudio
import torchaudio.transforms as T
import torch

class FunasrVADProcessor:
    def __init__(self):
        """
        Khởi tạo model Funasr với kích thước và ngôn ngữ cấu hình trước.
        """
        self.sampling_rate = 16000
        self.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", sampling_rate=self.sampling_rate)
    
    def covert_to_clip_timestamp_str(self, speech_timestamps: list) -> str:
        """
        Chuyển đổi danh sách các timestamp thành chuỗi định dạng "start1,end1,start2,end2,..."
        """
        sampling_rate = self.sampling_rate

        # Chuyển từng cặp [start, end] sang đơn vị giây (2 chữ số sau dấu phẩy)
        converted = [
            [round(start, 2), round(end, 2)]
            for start, end in speech_timestamps[0]['value']
        ]

        # b. Dạng chuỗi "s1,e1,s2,e2,..."
        flat_list = [str(x) for pair in converted for x in pair]
        clip_timestamp_str = ",".join(flat_list)

        return clip_timestamp_str

    def process(self, input_path: str, output_path: str, show_timestamps: bool = True) -> list:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        speech_timestamps = self.model.generate(input=input_path, sampling_rate=self.sampling_rate)
        
        if show_timestamps:
            print("Speech timestamps:", speech_timestamps)

        if not speech_timestamps:
            print("⚠️ No speech detected.")
            return []
        
        self.save_output(input_path, output_path, speech_timestamps)

        return self.covert_to_clip_timestamp_str(speech_timestamps)
    
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
        for start, end in speech_timestamps[0]['value']:
            chunk = waveform[:, int(start):int(end)]
            chunks.append(chunk)

        # Ghép lại thành 1 đoạn duy nhất
        merged_waveform = torch.cat(chunks, dim=1)  # ghép theo chiều thời gian

        # Lưu file đã ghép
        torchaudio.save(output_path, merged_waveform, sample_rate=self.sampling_rate)
        print(f"✅ Saved merged audio to: {output_path}")


# if __name__ == "__main__":
#     input_file = "/data/cuongdd/ai_service_copy/model/denoise/mytv_VOD_tienganh2_deepfilter.wav"
#     output_file = "/data/cuongdd/ai_service_copy/model/vad_segments/mytv_VOD_tienganh2_deepfilter_fsmn.wav"

#     vad_processor = FunasrVADProcessor()
#     result = vad_processor.process(input_file, output_file)

#     print("\n✅ clip_timestamps dùng cho Whisper:")
#     print(result)
