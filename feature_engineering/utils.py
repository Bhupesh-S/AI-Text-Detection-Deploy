"""
Optimised Handcrafted Feature Extraction Utilities
===================================================
Computes 27 syntactic, stylometric, statistical, and readability features
from raw and tokenised text splits to differentiate human from AI writing.
"""

import re
import string
import math
import logging
from typing import List, Dict, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stopwords set (NLTK-aligned)
STOPWORDS: frozenset = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "you're", "you've", "you'll", "you'd", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who",
    "whom", "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
    "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't",
    "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma",
    "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't",
    "wouldn", "wouldn't"
})

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_PUNCT_CHARS = set(string.punctuation)
_CITATION_PATTERN = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]|\([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?,\s*\d{4}\)")
_MATH_CHARS = set("=+$#<>^_*\\%/")

def _count_syllables_word(word: str) -> int:
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)

def _shannon_entropy(items) -> float:
    if not items:
        return 0.0
    from collections import Counter
    counts = Counter(items)
    total = len(items)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 6)

def extract_features(raw_text: str) -> Dict[str, Union[int, float]]:
    """Extract 27 production-ready features from raw original text."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "word_count": 0, "char_count": 0, "sentence_count": 0,
            "avg_word_length": 0.0, "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0, "sentence_length_burstiness": 0.0,
            "vocab_size": 0, "ttr": 0.0, "yules_k": 0.0, "hapax_ratio": 0.0,
            "stopword_ratio": 0.0, "punctuation_ratio": 0.0, "digit_ratio": 0.0,
            "uppercase_ratio": 0.0, "parentheses_ratio": 0.0, "quotation_ratio": 0.0,
            "citation_density": 0.0, "latex_formula_density": 0.0,
            "char_entropy": 0.0, "word_entropy": 0.0,
            "flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0, "smog_grade": 0.0, "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0
        }

    # 1. Base Tokenisation
    words_raw = raw_text.split()
    words_clean = [w.lower().translate(str.maketrans("", "", string.punctuation)) for w in words_raw]
    words_clean = [w for w in words_clean if w]

    word_count = len(words_clean)
    char_count = len(raw_text)

    # 2. Sentences
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(raw_text) if s.strip()]
    sentence_count = len(sentences)

    # Sentence Length Stats
    sentence_word_lengths = []
    for s in sentences:
        s_words = [w for w in s.split() if w.translate(str.maketrans("", "", string.punctuation))]
        if s_words:
            sentence_word_lengths.append(len(s_words))

    if sentence_word_lengths:
        mean_sent_len = np.mean(sentence_word_lengths)
        var_sent_len = np.var(sentence_word_lengths)
        std_sent_len = np.std(sentence_word_lengths)
        burstiness = (std_sent_len / mean_sent_len) if mean_sent_len > 0 else 0.0
    else:
        mean_sent_len = 0.0
        var_sent_len = 0.0
        burstiness = 0.0

    # 3. Lexical Diversity
    vocab = set(words_clean)
    vocab_size = len(vocab)
    ttr = (vocab_size / word_count) if word_count > 0 else 0.0

    # Yule's K
    if word_count > 1:
        from collections import Counter
        counts = Counter(words_clean)
        m1 = word_count
        m2 = sum(c * c for c in counts.values())
        yules_k = 10000.0 * (m2 - m1) / (m1 * m1)
        hapax_count = sum(1 for c in counts.values() if c == 1)
        hapax_ratio = hapax_count / word_count
    else:
        yules_k = 0.0
        hapax_ratio = 0.0

    # 4. Syntactic / Character ratios
    sw_count = sum(1 for w in words_clean if w in STOPWORDS)
    stopword_ratio = (sw_count / word_count) if word_count > 0 else 0.0

    punctuation_ratio = sum(1 for c in raw_text if c in _PUNCT_CHARS) / char_count if char_count > 0 else 0.0
    digit_ratio = sum(1 for c in raw_text if c.isdigit()) / char_count if char_count > 0 else 0.0
    uppercase_ratio = sum(1 for c in raw_text if c.isupper()) / char_count if char_count > 0 else 0.0
    parentheses_ratio = sum(1 for c in raw_text if c in "()[]{}") / char_count if char_count > 0 else 0.0
    quotation_ratio = sum(1 for c in raw_text if c in "\"'`“”‘’") / char_count if char_count > 0 else 0.0

    # 5. Academic Citations and Math
    citation_count = len(_CITATION_PATTERN.findall(raw_text))
    citation_density = (citation_count / word_count) if word_count > 0 else 0.0
    latex_count = sum(1 for c in raw_text if c in _MATH_CHARS)
    latex_formula_density = latex_count / char_count if char_count > 0 else 0.0

    # 6. Entropies
    char_entropy = _shannon_entropy(list(raw_text))
    word_entropy = _shannon_entropy(words_clean)

    # 7. Readability Elements
    avg_word_len = sum(len(w) for w in words_clean) / word_count if word_count > 0 else 0.0

    syllables = [_count_syllables_word(w) for w in words_clean]
    total_syllables = sum(syllables)
    complex_words_count = sum(1 for syl in syllables if syl >= 3)

    # Formulas
    words_per_sent = (word_count / sentence_count) if sentence_count > 0 else 0.0
    syl_per_word = (total_syllables / word_count) if word_count > 0 else 0.0
    complex_ratio = (complex_words_count / word_count) if word_count > 0 else 0.0

    flesch_reading_ease = 206.835 - 1.015 * words_per_sent - 84.6 * syl_per_word
    flesch_kincaid_grade = 0.39 * words_per_sent + 11.8 * syl_per_word - 15.59
    gunning_fog = 0.4 * (words_per_sent + 100.0 * complex_ratio)

    if sentence_count > 0:
        smog_grade = 1.0430 * math.sqrt(complex_words_count * (30.0 / sentence_count)) + 3.1291
    else:
        smog_grade = 0.0

    letters_per_100 = (sum(len(w) for w in words_clean) / word_count * 100) if word_count > 0 else 0.0
    sentences_per_100 = (sentence_count / word_count * 100) if word_count > 0 else 0.0
    coleman_liau_index = 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8

    automated_readability_index = 4.71 * (char_count / word_count) + 0.5 * words_per_sent - 21.43 if word_count > 0 else 0.0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_len, 6),
        "avg_sentence_length": round(mean_sent_len, 6),
        "sentence_length_variance": round(var_sent_len, 6),
        "sentence_length_burstiness": round(burstiness, 6),
        "vocab_size": vocab_size,
        "ttr": round(ttr, 6),
        "yules_k": round(yules_k, 6),
        "hapax_ratio": round(hapax_ratio, 6),
        "stopword_ratio": round(stopword_ratio, 6),
        "punctuation_ratio": round(punctuation_ratio, 6),
        "digit_ratio": round(digit_ratio, 6),
        "uppercase_ratio": round(uppercase_ratio, 6),
        "parentheses_ratio": round(parentheses_ratio, 6),
        "quotation_ratio": round(quotation_ratio, 6),
        "citation_density": round(citation_density, 6),
        "latex_formula_density": round(latex_formula_density, 6),
        "char_entropy": round(char_entropy, 6),
        "word_entropy": round(word_entropy, 6),
        "flesch_reading_ease": round(flesch_reading_ease, 6),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 6),
        "gunning_fog": round(gunning_fog, 6),
        "smog_grade": round(smog_grade, 6),
        "coleman_liau_index": round(coleman_liau_index, 6),
        "automated_readability_index": round(automated_readability_index, 6)
    }

def extract_features_batch(texts: List[str]) -> List[Dict[str, Union[int, float]]]:
    return [extract_features(t) for t in texts]

def features_to_dataframe(features_list: List[Dict[str, Union[int, float]]]) -> pd.DataFrame:
    return pd.DataFrame(features_list)
