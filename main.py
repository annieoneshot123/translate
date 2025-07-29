# main entry point for NMT Seq2Seq project
from utils.data_utils import read_data, analyze_data, print_stats
from utils.preprocess import normalize_text, tokenize, build_vocab, sentence_to_indices, add_special_tokens, pad_sequence, SPECIAL_TOKENS
from models.seq2seq import Encoder, Decoder, Seq2Seq
from models.seq2seq_attention import Encoder as AttnEncoder, Decoder as AttnDecoder, Seq2SeqAttention
from models.attention import Attention
from utils.train_utils import train_epoch, evaluate
from utils.bleu_utils import compute_bleu
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    pairs = read_data()
    stats = analyze_data(pairs)
    print_stats(stats)
    print("Ví dụ một số cặp câu:")
    for i in range(min(3, len(pairs))):
        print(f"EN: {pairs[i][0]} | VI: {pairs[i][1]}")

    # Tiền xử lý dữ liệu
    src_lang = 'en'
    tgt_lang = 'vi'
    src_sentences = [tokenize(normalize_text(src, src_lang), src_lang) for src, _ in pairs]
    tgt_sentences = [tokenize(normalize_text(tgt, tgt_lang), tgt_lang) for _, tgt in pairs]
    src_sentences = [add_special_tokens(s) for s in src_sentences]
    tgt_sentences = [add_special_tokens(s) for s in tgt_sentences]

    src_word2idx, src_idx2word = build_vocab(src_sentences, max_vocab_size=5000)
    tgt_word2idx, tgt_idx2word = build_vocab(tgt_sentences, max_vocab_size=5000)

    src_indices = [sentence_to_indices(s, src_word2idx) for s in src_sentences]
    tgt_indices = [sentence_to_indices(s, tgt_word2idx) for s in tgt_sentences]

    max_src_len = max(len(s) for s in src_indices)
    max_tgt_len = max(len(s) for s in tgt_indices)
    src_padded = [pad_sequence(s, max_src_len, src_word2idx[SPECIAL_TOKENS['PAD']]) for s in src_indices]
    tgt_padded = [pad_sequence(s, max_tgt_len, tgt_word2idx[SPECIAL_TOKENS['PAD']]) for s in tgt_indices]

    print("\nVí dụ tiền xử lý:")
    for i in range(min(2, len(src_padded))):
        print(f"SRC tokens: {src_sentences[i]}")
        print(f"SRC indices: {src_padded[i]}")
        print(f"TGT tokens: {tgt_sentences[i]}")
        print(f"TGT indices: {tgt_padded[i]}")

    # Kiểm tra giá trị chỉ số lớn nhất
    print(f"Max src index: {max(max(s) for s in src_indices)} | Vocab size: {len(src_word2idx)}")
    print(f"Max tgt index: {max(max(t) for t in tgt_indices)} | Vocab size: {len(tgt_word2idx)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tham số mô hình
    EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.3
    encoder = Encoder(len(src_word2idx), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
    decoder = Decoder(len(tgt_word2idx), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print(f"\nEncoder: input_dim={len(src_word2idx)}, emb_dim={EMB_DIM}, hid_dim={HID_DIM}, n_layers={N_LAYERS}, dropout={DROPOUT}")
    print(f"Decoder: output_dim={len(tgt_word2idx)}, emb_dim={EMB_DIM}, hid_dim={HID_DIM}, n_layers={N_LAYERS}, dropout={DROPOUT}")
    print(f"Model device: {device}")
    # Minh họa shape tensor qua từng lớp
    src_tensor = torch.tensor(src_padded, dtype=torch.long).to(device)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long).to(device)
    hidden, cell = encoder(src_tensor)
    print(f"\nEncoder output hidden shape: {hidden.shape}, cell shape: {cell.shape}")
    output, hidden, cell = decoder(tgt_tensor[:,0], hidden, cell)
    print(f"Decoder output shape (1 step): {output.shape}")
    outputs = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.75)
    print(f"Seq2Seq outputs shape: {outputs.shape}")

    # Chia dữ liệu train/test
    split_idx = int(len(src_padded) * 0.8)
    train_data = list(zip(src_padded[:split_idx], tgt_padded[:split_idx]))
    test_data = list(zip(src_padded[split_idx:], tgt_padded[split_idx:]))
    criterion = nn.CrossEntropyLoss(ignore_index=src_word2idx[SPECIAL_TOKENS['PAD']])
    optimizer = optim.Adam(model.parameters())
    N_EPOCHS = 10
    CLIP = 1.0
    print(f"\nBắt đầu huấn luyện...")
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model, train_data, optimizer, criterion, CLIP, device)
        val_loss = evaluate(model, test_data, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Khởi tạo mô hình Attention
    attention = Attention(HID_DIM)
    attn_encoder = AttnEncoder(len(src_word2idx), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
    attn_decoder = AttnDecoder(len(tgt_word2idx), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attention).to(device)
    attn_model = Seq2SeqAttention(attn_encoder, attn_decoder, device).to(device)
    attn_optimizer = optim.Adam(attn_model.parameters())
    attn_criterion = nn.CrossEntropyLoss(ignore_index=src_word2idx[SPECIAL_TOKENS['PAD']])
    print("\nBắt đầu huấn luyện mô hình Attention...")
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(attn_model, train_data, attn_optimizer, attn_criterion, CLIP, device)
        val_loss = evaluate(attn_model, test_data, attn_criterion, device)
        print(f"[Attention] Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Hàm dịch một câu (inference)
    def translate_sentence(model, src_seq, src_word2idx, tgt_idx2word, device, max_len=20, use_attention=False):
        model.eval()
        src = torch.tensor(src_seq, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            if use_attention:
                encoder_outputs, hidden, cell = model.encoder(src)
                input = torch.tensor([src_word2idx['<SOS>']], dtype=torch.long).to(device)
                outputs = []
                for _ in range(max_len):
                    output, hidden, cell, _ = model.decoder(input, hidden, cell, encoder_outputs)
                    top1 = output.argmax(1)
                    word = tgt_idx2word[top1.item()]
                    if word == '<EOS>':
                        break
                    outputs.append(word)
                    input = top1
                return outputs
            else:
                hidden, cell = model.encoder(src)
                input = torch.tensor([src_word2idx['<SOS>']], dtype=torch.long).to(device)
                outputs = []
                for _ in range(max_len):
                    output, hidden, cell = model.decoder(input, hidden, cell)
                    top1 = output.argmax(1)
                    word = tgt_idx2word[top1.item()]
                    if word == '<EOS>':
                        break
                    outputs.append(word)
                    input = top1
                return outputs

    # Đánh giá BLEU trên tập test
    print("\nĐánh giá BLEU trên tập test:")
    bleu_scores = []
    for src, tgt in test_data:
        pred = translate_sentence(model, src, src_word2idx, tgt_idx2word, device, max_len=len(tgt))
        tgt_tokens = [w for w in tgt if w != tgt_word2idx['<PAD>']]
        ref = [tgt_idx2word[idx] for idx in tgt_tokens if tgt_idx2word[idx] not in ['<PAD>', '<SOS>']]
        bleu = compute_bleu(ref, pred)
        bleu_scores.append(bleu)
        print(f"SRC: {[src_idx2word[idx] for idx in src if src_idx2word[idx] not in ['<PAD>', '<SOS>']]}")
        print(f"REF: {ref}")
        print(f"PRED: {pred}")
        print(f"BLEU: {bleu:.4f}\n")
    print(f"Trung bình BLEU score: {sum(bleu_scores)/len(bleu_scores):.4f}")

    # Đánh giá BLEU cho mô hình Attention
    print("\nĐánh giá BLEU trên tập test (Attention):")
    bleu_scores = []
    for src, tgt in test_data:
        pred = translate_sentence(attn_model, src, src_word2idx, tgt_idx2word, device, max_len=len(tgt), use_attention=True)
        tgt_tokens = [w for w in tgt if w != tgt_word2idx['<PAD>']]
        ref = [tgt_idx2word[idx] for idx in tgt_tokens if tgt_idx2word[idx] not in ['<PAD>', '<SOS>']]
        bleu = compute_bleu(ref, pred)
        bleu_scores.append(bleu)
        print(f"SRC: {[src_idx2word[idx] for idx in src if src_idx2word[idx] not in ['<PAD>', '<SOS>']]}")
        print(f"REF: {ref}")
        print(f"PRED: {pred}")
        print(f"BLEU: {bleu:.4f}\n")
    print(f"Trung bình BLEU score (Attention): {sum(bleu_scores)/len(bleu_scores):.4f}")
    print("\nSo sánh hiệu quả Attention vs không Attention: Kiểm tra BLEU score và chất lượng dịch từng câu.")
    print("Vai trò Attention: giúp mô hình tập trung vào các phần quan trọng của câu nguồn khi dịch từng token, cải thiện chất lượng dịch, đặc biệt với câu dài và phức tạp.")
    # Báo cáo và phân tích lỗi
    print("\nBáo cáo và phân tích lỗi dịch:")
    for i, (src, tgt) in enumerate(test_data):
        pred = translate_sentence(model, src, src_word2idx, tgt_idx2word, device, max_len=len(tgt))
        tgt_tokens = [w for w in tgt if w != tgt_word2idx['<PAD>']]
        ref = [tgt_idx2word[idx] for idx in tgt_tokens if tgt_idx2word[idx] not in ['<PAD>', '<SOS>']]
        src_text = ' '.join([src_idx2word[idx] for idx in src if src_idx2word[idx] not in ['<PAD>', '<SOS>']])
        ref_text = ' '.join(ref)
        pred_text = ' '.join(pred)
        print(f"Câu gốc: {src_text}")
        print(f"Dịch đúng: {ref_text}")
        print(f"Mô hình dịch: {pred_text}")
        # Phân tích lỗi
        missing = set(ref) - set(pred)
        wrong = set(pred) - set(ref)
        print(f"Thiếu từ: {missing}")
        print(f"Sai từ: {wrong}")
        print("---")
    print("\nHướng dẫn chuyển đổi giữa Anh-Việt và Việt-Anh:")
    print("Chỉ cần đổi src_lang, tgt_lang, src_word2idx, tgt_word2idx, src_idx2word, tgt_idx2word và dữ liệu đầu vào tương ứng.")

if __name__ == "__main__":
    main()
