import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from konlpy.tag import Okt
import jieba
from underthesea import word_tokenize

# ---------------------- TIỀN XỬ LÝ VÀ TÁCH TỪ ----------------------
def preprocess_text(text, language='english'):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    if language == 'chinese':
        from zhconv import convert
        text = convert(text, 'zh-cn')
    return text

def tokenize_english(text): return text.split()
def tokenize_vietnamese(text): return word_tokenize(text, format="text").split()
def tokenize_korean(text): return Okt().morphs(text)
def tokenize_chinese(text): return list(jieba.cut(text))

def tokenize_text(text, language='english'):
    if language == 'english': return tokenize_english(text)
    elif language == 'vietnamese': return tokenize_vietnamese(text)
    elif language == 'korean': return tokenize_korean(text)
    elif language == 'chinese': return tokenize_chinese(text)
    else: raise ValueError("Ngôn ngữ không được hỗ trợ!")

# ---------------------- ĐỌC FILE VTT ----------------------
def read_vtt_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    content = []
    for line in lines:
        line = line.strip()
        if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} -->", line): continue
        if line.isdigit() or not line: continue
        content.append(line)
    return ' '.join(content)

# ---------------------- CẤU HÌNH ----------------------
reference_file = '/data/datn/DATN_chat/data/original_translate/translate_TheThaoTV.vtt'  # File tham chiếu
candidate_file = '/data/datn/DATN_chat/data/output/TheThao_vi_en.vtt'             # File sinh ra
output_file = '/data/datn/DATN_chat/data/output_evaluate/bleu_bert_TheThao.txt'
language = 'english'  # Chọn 'english', 'vietnamese', 'korean', 'chinese'

# ---------------------- ĐỌC FILE & TIỀN XỬ LÝ ----------------------
try:
    ref_text = read_vtt_text(reference_file)
    cand_text = read_vtt_text(candidate_file)
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file đầu vào.")
    exit(1)

ref_text_clean = preprocess_text(ref_text, language)
cand_text_clean = preprocess_text(cand_text, language)

# ---------------------- BLEU-1 ----------------------
ref_tokens = tokenize_text(ref_text_clean, language)
cand_tokens = tokenize_text(cand_text_clean, language)
smoothie = SmoothingFunction().method4
bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)

# ---------------------- BERTScore ----------------------
bert_P, bert_R, bert_F1 = bert_score([cand_text], [ref_text], lang=language, verbose=True)
bert_f1 = bert_F1[0].item()

# ---------------------- GHI KẾT QUẢ ----------------------
with open(output_file, 'w', encoding='utf-8') as out:
    out.write(f"===== Evaluation ({language.capitalize()}) =====\n")
    out.write(f"BLEU-1: {bleu_1:.4f}\n")
    out.write(f"BERTScore F1: {bert_f1:.4f}\n")

print(f"✅ Đã lưu BLEU-1 & BERTScore (ngôn ngữ: {language}) vào: {output_file}")
