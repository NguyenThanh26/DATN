import re
import os
from konlpy.tag import Okt  # Tiếng Hàn
import jieba  # Tiếng Trung
from underthesea import word_tokenize  # Tiếng Việt

def preprocess_text(text, language='english'):
    """Tiền xử lý văn bản: chuyển về chữ thường, loại bỏ dấu câu."""
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    if language == 'chinese':
        try:
            from zhconv import convert
            text = convert(text, 'zh-cn')  # Chuẩn hóa tiếng Trung (giản thể)
        except ImportError:
            print("Cảnh báo: Thư viện zhconv không được cài đặt. Bỏ qua chuẩn hóa tiếng Trung.")
    return text

def tokenize_english(text):
    """Phân tách văn bản tiếng Anh."""
    return text.split()

def tokenize_vietnamese(text):
    """Phân tách văn bản tiếng Việt."""
    return word_tokenize(text, format="text").split()

def tokenize_korean(text):
    """Phân tách văn bản tiếng Hàn."""
    okt = Okt()
    return okt.morphs(text)

def tokenize_chinese(text):
    """Phân tách văn bản tiếng Trung."""
    return list(jieba.cut(text))

def tokenize_text(text, language='english'):
    """Phân tách văn bản theo ngôn ngữ."""
    if language == 'english':
        return tokenize_english(text)
    elif language == 'vietnamese':
        return tokenize_vietnamese(text)
    elif language == 'korean':
        return tokenize_korean(text)
    elif language == 'chinese':
        return tokenize_chinese(text)
    else:
        raise ValueError("Ngôn ngữ không được hỗ trợ! Chọn 'english', 'vietnamese', 'korean', hoặc 'chinese'.")

def read_vtt_text(path):
    """Đọc nội dung văn bản từ tệp VTT, bỏ qua dòng thời gian và số thứ tự."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tệp không tồn tại: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    content = []
    for line in lines:
        line = line.strip()
        # Bỏ qua dòng WEBVTT, dòng trống, số thứ tự, và dòng thời gian
        if line == "WEBVTT" or not line or line.isdigit():
            continue
        if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', line):
            continue
        content.append(line)
    
    return ' '.join(content)

def levenshtein_distance(ref_tokens, cand_tokens):
    """Tính Levenshtein Distance giữa hai danh sách token."""
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i  # Số lần xóa
    for j in range(n + 1):
        dp[0][j] = j  # Số lần thêm
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1,  # Thay thế
                               dp[i-1][j] + 1,    # Xóa
                               dp[i][j-1] + 1)    # Thêm
    
    return dp[m][n]

def calculate_wer(ref_tokens, cand_tokens):
    """Tính WER dựa trên Levenshtein Distance."""
    if not ref_tokens and not cand_tokens:
        return 0.0  # Cả hai rỗng, WER = 0
    if not ref_tokens:
        return 1.0  # Tham chiếu rỗng, WER = 1
    if not cand_tokens:
        return 1.0  # Ứng viên rỗng, WER = 1
    
    errors = levenshtein_distance(ref_tokens, cand_tokens)
    ref_length = len(ref_tokens)
    return errors / ref_length

# Điền đường dẫn file
reference_file = '/data/datn/ai_service/data/output_vtt_demo/AmNhac_vi_vi.vtt'
candidate_file = '/data/datn/ai_service/data/output_vtt/AmNhac_vi_vi.vtt'
output_file = '/data/datn/DATN_chat/data/output_evaluate/WER_Amnhac.txt'
language = 'vietnamese'  # Thay đổi thành 'english', 'korean', hoặc 'chinese'

try:
    # Kiểm tra và tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Đọc và xử lý văn bản
    ref_text = read_vtt_text(reference_file)
    cand_text = read_vtt_text(candidate_file)
    
    # Tiền xử lý và phân tách
    ref_text = preprocess_text(ref_text, language=language)
    cand_text = preprocess_text(cand_text, language=language)
    
    ref_tokens = tokenize_text(ref_text, language=language)
    cand_tokens = tokenize_text(cand_text, language=language)
    
    # Tính WER
    wer_score = calculate_wer(ref_tokens, cand_tokens)
    
    # Lưu kết quả
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"===== WER Evaluation ({language.capitalize()}) =====\n")
        out.write(f"Reference File: {reference_file}\n")
        out.write(f"Candidate File: {candidate_file}\n")
        out.write(f"WER: {wer_score:.4f}\n")
    
    print(f"Đã lưu WER (ngôn ngữ: {language}) vào file '{output_file}'")

except FileNotFoundError as e:
    print(f"Lỗi: {str(e)}")
except Exception as e:
    print(f"Lỗi không mong muốn: {str(e)}")