import os
import torchaudio
import torchaudio.transforms as T
import torch
import logging
from funasr import AutoModel

logger = logging.getLogger(__name__)

class FunasrVADProcessor:
    def __init__(self, sampling_rate=16000):
        """
        Khởi tạo FunASR VAD với cấu hình.
        """
        self.sampling_rate = sampling_rate
        try:
            self.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", sampling_rate=sampling_rate)
            logger.info("Loaded FunASR VAD model successfully")
        except Exception as e:
            logger.error(f"Error loading FunASR VAD model: {str(e)}")
            raise

    def convert_to_clip_timestamp_str(self, speech_timestamps: list) -> str:
        """
        Chuyển đổi danh sách timestamp thành chuỗi "start1,end1,start2,end2,...".
        """
        converted = [
            [round(start, 2), round(end, 2)]
            for start, end in speech_timestamps
        ]
        flat_list = [str(x) for pair in converted for x in pair]
        return ",".join(flat_list)

    def save_output(self, input_path: str, output_path: str, speech_timestamps: list):
        """
        Lưu những đoạn âm thanh được phát hiện từ VAD vào một file âm thanh hoàn chỉnh.
        """
        waveform, sr = torchaudio.load(input_path)
        if sr != self.sampling_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=sampling_rate)
            waveform = resampler(waveform)

        chunks = []
        for start, end in speech_timestamps:
            chunk = waveform[:, int(start):int(end)]
            if chunk.shape[1] == 0:
                continue
            chunks.append(chunk)

        if not chunks:
            raise Exception("No valid audio chunks after VAD processing")

        merged_waveform = torch.cat(chunks, dim=1)
        torchaudio.save(output_path, merged_waveform, sample_rate=sr)
        logger.info(f"Saved merged audio to: {output_path}")

    def process(self, input_path: str, output_path: str, show_timestamps: bool = True) -> tuple:
        """
        Thực hiện VAD, ghép các đoạn tiếng nói vào một file hoàn chỉnh, 
        đồng thời trả về timestamp của những đoạn tiếng nói.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            # Dùng mô hình VAD để phát hiện thời gian tiếng nói
            speech_timestamps = self.model.generate(input=input_path, sampling_rate=self.sampling_rate)
            if not speech_timestamps:
                logger.warning("No speech detected.")
                return "", []

            if show_timestamps:
                logger.info(f"Speech timestamps: {speech_timestamps}")

            # Chuyển sang thời gian thực (giây)
            timestamps_absolute = [
                {"start": start / self.sampling_rate, "end": end / self.sampling_rate}
                for start, end in speech_timestamps[0]['value']
            ]

            self.save_output(input_path, output_path, speech_timestamps[0]['value'])

            clip_timestamp_str = self.convert_to_clip_timestamp_str(speech_timestamps[0]['value'])

            return clip_timestamp_str, timestamps_absolute

        except Exception as e:
            logger.error(f"Error in FunASR VAD processing: {str(e)}")
            raise
