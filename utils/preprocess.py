import re
import nltk
from collections import Counter

# Nếu chưa có NLTK tokenizer, tải về
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

SPECIAL_TOKENS = {'PAD': '<PAD>', 'SOS': '<SOS>', 'EOS': '<EOS>', 'UNK': '<UNK>'}

# Chuẩn hóa văn bản

def normalize_text(text, lang='en'):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\.!?]", "", text) if lang == 'en' else re.sub(r"[^a-zA-ZÀ-ỹà-ỹ0-9\s\.!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tách từ/tokenization

def tokenize(text, lang='en'):
    if lang == 'en':
        return nltk.word_tokenize(text)
    else:
        # Tiếng Việt: tách theo khoảng trắng, ghi chú hạn chế
        return text.split()

# Tạo từ điển (vocab)
def build_vocab(sentences, max_vocab_size=5000):
    counter = Counter()
    for sent in sentences:
        # Bỏ qua các token đặc biệt khi đếm
        counter.update([w for w in sent if w not in SPECIAL_TOKENS.values()])
    vocab = [SPECIAL_TOKENS['PAD'], SPECIAL_TOKENS['SOS'], SPECIAL_TOKENS['EOS'], SPECIAL_TOKENS['UNK']] + [w for w, _ in counter.most_common(max_vocab_size-4)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

# Chuyển câu sang chuỗi chỉ số

def sentence_to_indices(sentence, word2idx):
    return [word2idx.get(w, word2idx[SPECIAL_TOKENS['UNK']]) for w in sentence]

# Thêm token đặc biệt

def add_special_tokens(tokens):
    return [SPECIAL_TOKENS['SOS']] + tokens + [SPECIAL_TOKENS['EOS']]

# Padding

def pad_sequence(seq, max_len, pad_value):
    return seq + [pad_value] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
