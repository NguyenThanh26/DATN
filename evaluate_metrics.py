import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from underthesea import word_tokenize
import logging

logger = logging.getLogger(__name__)

def preprocess_text(text, language='english'):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_english(text):
    return text.split()

def tokenize_vietnamese(text):
    return word_tokenize(text, format="text").split()

def tokenize_text(text, language='english'):
    if language == 'english':
        return tokenize_english(text)
    elif language == 'vietnamese':
        return tokenize_vietnamese(text)
    else:
        logger.warning(f"Language {language} not fully supported, using default tokenizer.")
        return tokenize_english(text)

def read_vtt_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        content = []
        for line in lines:
            line = line.strip()
            if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} -->", line): continue
            if line.isdigit() or not line: continue
            content.append(line)
        return ' '.join(content)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

def levenshtein_distance(ref_tokens, cand_tokens):
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)
    return dp[m][n]

def calculate_wer(reference_path, hypothesis_path, language='english'):
    ref_text = read_vtt_text(reference_path)
    cand_text = read_vtt_text(hypothesis_path)
    ref_text = preprocess_text(ref_text, language=language)
    cand_text = preprocess_text(cand_text, language=language)
    ref_tokens = tokenize_text(ref_text, language=language)
    cand_tokens = tokenize_text(cand_text, language=language)
    errors = levenshtein_distance(ref_tokens, cand_tokens)
    ref_length = len(ref_tokens)
    wer_score = 0.0 if ref_length == 0 else errors / ref_length
    logger.info(f"WER: {wer_score:.4f} for {reference_path} vs {hypothesis_path}")
    return wer_score

def calculate_bleu(reference_path, hypothesis_path, language='english'):
    ref_text = read_vtt_text(reference_path)
    cand_text = read_vtt_text(hypothesis_path)
    ref_text = preprocess_text(ref_text, language=language)
    cand_text = preprocess_text(cand_text, language=language)
    ref_tokens = tokenize_text(ref_text, language=language)
    cand_tokens = tokenize_text(cand_text, language=language)
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([ref_tokens], cand_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    logger.info(f"BLEU-1: {bleu_score:.4f} for {reference_path} vs {hypothesis_path}")
    return bleu_score