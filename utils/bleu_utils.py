import nltk

def compute_bleu(reference, hypothesis):
    # reference: list of token lists (ground truth)
    # hypothesis: list of tokens (predicted)
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
