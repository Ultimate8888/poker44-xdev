"""
Own action n-gram features for xdev-trainer-v7 (independent implementation).

Captures action-SEQUENCE micro-patterns as a projection-robust bag-of-n-grams,
complementing the aggregate features in features.py. Each action becomes a token
<street_initial><action_code><pot-ratio bucket>; we count unigrams, bigrams and
trigrams of consecutive tokens across the chunk, normalized per hand. The fixed
vocabulary below was selected from our own training data (benchmark + real-human
sessions, projected through prepare_hand_for_miner) by bot/human discrimination.
Standard n-gram method; implementation and vocabulary are our own.
"""
from collections import Counter
import numpy as np

_ACT = {"check": "K", "call": "C", "bet": "B", "raise": "R", "fold": "F"}


def _bucket(amount: float, pot_before: float) -> str:
    if amount <= 0:
        return "z"
    if pot_before <= 0:
        return "u"
    r = amount / pot_before
    if r < 0.5:
        return "s"
    if r < 1.0:
        return "m"
    if r < 2.0:
        return "l"
    return "h"


def _hand_tokens(hand: dict) -> list:
    toks = []
    for a in (hand.get("actions") or []):
        street = str(a.get("street") or "x")[:1]
        code = _ACT.get(str(a.get("action_type") or "").lower(), "X")
        amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
        pb = float(a.get("pot_before", 0.0) or 0.0) / 0.02
        toks.append(street + code + _bucket(amt, pb))
    return toks


def _chunk_gram_counts(chunk: list) -> Counter:
    tot = Counter()
    for hand in chunk:
        t = _hand_tokens(hand)
        for x in t:
            tot[x] += 1
        for i in range(len(t) - 1):
            tot[t[i] + ">" + t[i + 1]] += 1
            if i + 2 < len(t):
                tot[t[i] + ">" + t[i + 1] + ">" + t[i + 2]] += 1
    return tot


# fixed vocabulary selected from our training data (48 tokens)
NGRAM_VOCAB = [
    'pRh', 'pRh>pFz', 'pRm', 'pBm', 
    'pFz>pBm', 'pBs', 'pFz>pBs', 'pFz>pRh', 
    'pFz>pFz>pFz', 'pFz>pFz>pBs', 'pRh>pFz>pFz', 'fBs', 
    'pCs', 'pRm>pFz', 'pFz>pRh>pFz', 'pFz>pFz>pBm', 
    'pFz>pRm', 'pFz>pRl', 'pFz>pFz>pRh', 'pRm>pFz>pFz', 
    'fFz', 'pFz>pFz', 'pFz>pRm>pFz', 'tBs', 
    'pRl>pFz', 'pFz>pRl>pFz', 'pRl', 'pFz>pCs', 
    'pFz>pFz>pRm', 'fFz>fBs', 'rBs', 'pFz>pFz>pRl', 
    'pRl>pFz>pFz', 'fKz', 'fBm>fFz', 'pFz>pCm', 
    'pCs>fKz', 'pRl>pFz>pBm', 'rFz', 'fCs', 
    'pCs>pFz', 'tFz>tBs', 'pKz', 'fBl', 
    'tFz', 'fKz>fBm', 'fBm', 'fKz>fBm>fFz', 
]

N_NGRAM_FEATURES = len(NGRAM_VOCAB)
assert N_NGRAM_FEATURES == 48


def extract_ngram_features(chunk) -> np.ndarray:
    """Return per-hand-normalized counts for the fixed n-gram vocabulary."""
    n = max(len(chunk or []), 1)
    g = _chunk_gram_counts(chunk or [])
    return np.array([g.get(t, 0.0) / n for t in NGRAM_VOCAB], dtype=np.float32)
