import soundfile as sf
import numpy as np
import noisereduce as nr
from tqdm import tqdm

# 1. Đọc file mono, float32
data, sr = sf.read('/data/cuongdd/ai_service/data/input/AmNhac.wav',
                   dtype='float32')
if data.ndim == 2:                # stereo -> mono
    data = data.mean(axis=1)

noise_sample = data[: int(0.8 * sr)]        # 0.5 s noise
seg_len  = 5 * sr             # 5 s/đoạn
overlap  = 1 * sr             # 1 s chồng lấn

with sf.SoundFile('./output_clean.wav', 'w', sr, 1, subtype='PCM_16') as out_f:
    for start in tqdm(range(0, len(data), seg_len)):
        end = min(start + seg_len + overlap, len(data))
        segment = data[start:end]

        seg_clean = nr.reduce_noise(
            y=segment,
            sr=sr,
            y_noise=noise_sample,
            stationary=True,      # đổi False nếu ồn dao động
            prop_decrease=1,    # 90 % là đủ, tránh artefact
            use_torch=True        # cần noisereduce>=3.0
        )

        # cắt bỏ phần overlap rồi ghi thẳng ra file
        clean_part = seg_clean[: min(seg_len, len(data) - start)]
        out_f.write(clean_part)
