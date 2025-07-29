import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/eng_vie_pairs.txt')

def read_data(file_path=DATA_PATH):
    """Đọc dữ liệu song ngữ từ file, trả về list các cặp câu."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line.split('\t')
            pairs.append((src, tgt))
    return pairs

def analyze_data(pairs):
    """Phân tích thống kê cơ bản về độ dài câu, số từ, phân bố độ dài."""
    src_lens = [len(s.split()) for s, _ in pairs]
    tgt_lens = [len(t.split()) for _, t in pairs]
    stats = {
        'num_pairs': len(pairs),
        'src_avg_len': np.mean(src_lens),
        'src_max_len': np.max(src_lens),
        'src_min_len': np.min(src_lens),
        'tgt_avg_len': np.mean(tgt_lens),
        'tgt_max_len': np.max(tgt_lens),
        'tgt_min_len': np.min(tgt_lens),
    }
    return stats

def print_stats(stats):
    print("Số lượng cặp câu:", stats['num_pairs'])
    print("Độ dài trung bình câu nguồn:", stats['src_avg_len'])
    print("Độ dài lớn nhất câu nguồn:", stats['src_max_len'])
    print("Độ dài nhỏ nhất câu nguồn:", stats['src_min_len'])
    print("Độ dài trung bình câu đích:", stats['tgt_avg_len'])
    print("Độ dài lớn nhất câu đích:", stats['tgt_max_len'])
    print("Độ dài nhỏ nhất câu đích:", stats['tgt_min_len'])
