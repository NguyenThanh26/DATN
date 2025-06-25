import numpy as np
import noisereduce as nr

class AudioDenoiser:
    """
    AudioDenoiser xử lý khử nhiễu từ một waveform.
    """
    def __init__(self, sample_rate=16000, prop_reduce=1.0, n_fft=2048, win_length=2048, hop_length=512):
        """
        Args:
            sample_rate (int): Tốc độ mẫu của âm thanh.
            prop_reduce (float): Tỉ lệ khử nhiễu (1.0 = hoàn toàn).
            n_fft (int): Kích thước biến đổi Fourirer.
            win_length (int): Kích thước khung phân tích.
            hop_length (int): Khoảng dịch khung.
        """
        self.sample_rate = sample_rate
        self.prop_reduce = prop_reduce
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def process(self, waveform_np):
        """
        Khử nhiễu waveform.
        
        Args:
            waveform_np (np.ndarray): ndarray có shape [1, time]
                                
        Returns:
            np.ndarray: waveform sau khi khử nhiễu với cùng shape [1, time]
        """
        if not isinstance(waveform_np, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if waveform_np.ndim == 1:
            waveform_np = np.expand_dims(waveform_np, axis=0)

        if waveform_np.ndim != 2 or waveform_np.shape[0] != 1:
            raise ValueError("Expected waveform of shape [1, time]")

        # Lấy kênh âm thanh duy nhất (mono)
        mono = waveform_np[0]

        # Khử nhiễu với noisereduce
        denoised = nr.reduce_noise(
            y=mono,
            sr=self.sample_rate,
            prop_decrease = self.prop_reduce,
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length
        )

        # Trả lại shape [1, time]
        return np.expand_dims(denoised, axis=0)
