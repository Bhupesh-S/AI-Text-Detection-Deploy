# Feature Engineering Report

## Dataset Information
- **Total Samples:** 12
- **Train Size:** 8 (70%)
- **Validation Size:** 2 (15%)
- **Test Size:** 2 (15%)

## Cleaning Summary
Text was converted to lowercase, URLs/HTML tags were removed, Unicode was normalized, extra whitespace/newlines/tabs were removed, and empty texts were dropped. Special characters were kept for NLP tokenization.

## TF-IDF Configuration
- **Max Features:** 10000
- **N-gram Range:** (1, 2)
- **Min DF:** 1
- **Max DF:** 1.0
- **Sublinear TF:** True
- **Strip Accents:** unicode

### TF-IDF Summary
- **Vocabulary Size:** 20
- **Top 20 Important Terms (by lowest IDF):** this is, this, is, machine, human, is human, is machine, one, three, two, machine four, machine one, machine three, machine two, human three, six, human six, human one, human two, four

## Feature List (Handcrafted NLP Features)
The following features were extracted and saved as CSV:
char_count, text_length, word_count, sentence_count, unique_words, type_token_ratio, uppercase_ratio, digit_ratio, punctuation_ratio, stopword_ratio, avg_word_length, avg_sentence_length, lexical_diversity

## Summary Statistics (Training Set)
```text
                     count       mean       std        min        25%        50%        75%        max
char_count             8.0  18.625000  1.505941  17.000000  17.000000  19.000000  19.250000  21.000000
text_length            8.0  18.625000  1.505941  17.000000  17.000000  19.000000  19.250000  21.000000
word_count             8.0   4.000000  0.000000   4.000000   4.000000   4.000000   4.000000   4.000000
sentence_count         8.0   1.000000  0.000000   1.000000   1.000000   1.000000   1.000000   1.000000
unique_words           8.0   4.000000  0.000000   4.000000   4.000000   4.000000   4.000000   4.000000
type_token_ratio       8.0   1.000000  0.000000   1.000000   1.000000   1.000000   1.000000   1.000000
uppercase_ratio        8.0   0.053998  0.004345   0.047619   0.051974   0.052632   0.058824   0.058824
digit_ratio            8.0   0.000000  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
punctuation_ratio      8.0   0.000000  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
stopword_ratio         8.0   0.500000  0.000000   0.500000   0.500000   0.500000   0.500000   0.500000
avg_word_length        8.0   3.906250  0.376485   3.500000   3.500000   4.000000   4.062500   4.500000
avg_sentence_length    8.0   4.000000  0.000000   4.000000   4.000000   4.000000   4.000000   4.000000
lexical_diversity      8.0   1.000000  0.000000   1.000000   1.000000   1.000000   1.000000   1.000000
```
