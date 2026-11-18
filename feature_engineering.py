"""
Feature Engineering for AI-generated vs Human-written Text Detection
----------------------------------------------------------------------
Extracts stylometric and statistical features from text samples that
help distinguish AI-generated text from human-written text, without
relying purely on word content.

Expected input: a CSV with at least two columns:
    text  - the text sample
    label - 0 for human, 1 for AI-generated  (adjust as per your dataset)

Usage:
    python feature_engineering.py --input data.csv --output features.csv
"""

import argparse #Reading command-line options like --input, --tune
import re    #The regular expression that finds contractions like don't, it's
import string    #string.punctuation, the list of punctuation characters

import pandas as pd
import nltk   #Splitting text into sentences and words (sent_tokenize, word_tokenize) and downloading its punkt data
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data (only runs once, then cached)
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Common English function words (articles, prepositions, conjunctions, etc.)
# AI-generated text often uses these more uniformly than human writing.
FUNCTION_WORDS = set("""
a an the and or but if because as until while of at by for with about
against between into through during before after above below to from
up down in out on off over under again further then once here there
when where why how all any both each few more most other some such
no nor not only own same so than too very is am are was were be been
being have has had do does did will would shall should may might must
can could
""".split())


def get_sentences(text: str):
    """Split text into sentences, guard against empty/malformed input."""
    if not isinstance(text, str) or not text.strip():
        return []
    return sent_tokenize(text)


def get_words(text: str):
    """Split text into lowercase word tokens, punctuation stripped."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = word_tokenize(text)
    return [t.lower() for t in tokens if t.isalpha()]


def avg_sentence_length(sentences):
    """Average number of words per sentence."""
    if not sentences:
        return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def sentence_length_variance(sentences):
    """
    'Burstiness' — how much sentence length fluctuates.
    Human writing tends to have higher variance (mixing short punchy
    sentences with long ones); AI text is often more uniform.
    """
    if len(sentences) < 2:
        return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    return variance


def type_token_ratio(words):
    """
    Lexical diversity: unique words / total words.
    Lower values can indicate repetitive, formulaic phrasing.
    """
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def fixed_window_ttr(words, window=50):
    """
    Length-controlled lexical diversity: TTR computed on only the first
    `window` words. Plain TTR falls as a text gets longer (long texts repeat
    more words), so it can secretly measure length. Using the same window for
    every text removes that effect. Texts shorter than `window` use all their
    words (prepare_dataset.py keeps only texts of at least 50 words, so this
    only matters for other datasets).
    """
    if not words:
        return 0.0
    chunk = words[:window]
    return len(set(chunk)) / len(chunk)


def function_word_ratio(words):
    """
    Fraction of words that are function words (the, of, and, etc.)
    AI text often shows a more 'balanced'/predictable function-word
    distribution than human text.
    """
    if not words:
        return 0.0
    fw_count = sum(1 for w in words if w in FUNCTION_WORDS)
    return fw_count / len(words)


def punctuation_frequency(text: str):
    """
    Punctuation marks per 100 characters.
    Captures stylistic habits (heavy comma use, exclamation marks,
    em-dashes, etc.) that differ between AI and human writing styles.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    punct_count = sum(1 for ch in text if ch in string.punctuation)
    return (punct_count / len(text)) * 100


def avg_word_length(words):
    """Average number of characters per word."""
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


# ---------------------------------------------------------------------
# Extra stylometric features (v2)
# ---------------------------------------------------------------------
# Classic stylometry looks at how often an author uses individual common
# words (not just the total share of function words), plus punctuation habits.
TOP_FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "it", "for", "as", "with",
    "was", "on", "be", "are", "this", "by", "or", "not", "but", "have", "from",
    "an", "at", "they", "which", "you", "can", "will", "would", "there", "their",
    "if", "so", "more", "also", "about", "one", "we",
]
PUNCT_MARKS = {
    "comma": ",", "period": ".", "question": "?", "exclaim": "!", "colon": ":",
    "semicolon": ";", "paren": "(", "quote": '"', "apostrophe": "'", "hyphen": "-",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our"}
CONTRACTION_RE = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


def per_function_word_freq(words):
    """Relative frequency of each of the most common function words."""
    n = len(words) or 1
    counts = {w: 0 for w in TOP_FUNCTION_WORDS}
    for w in words:
        if w in counts:
            counts[w] += 1
    return {f"fw_{w}": c / n for w, c in counts.items()}


def punctuation_breakdown(text):
    """Rate of each punctuation mark per 100 characters."""
    n = len(text) or 1
    return {f"punct_{name}": text.count(ch) / n * 100 for name, ch in PUNCT_MARKS.items()}


def misc_style_features(text, words):
    """Capitalisation, digits, line breaks, contractions, first-person words, long words."""
    n_chars = len(text) or 1
    n_words = len(words) or 1
    letters = [c for c in text if c.isalpha()]
    lengths = [len(w) for w in words] or [0]
    mean_len = sum(lengths) / len(lengths)
    return {
        "uppercase_ratio": sum(1 for c in letters if c.isupper()) / (len(letters) or 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
        "newline_count": text.count("\n"),
        "contraction_ratio": len(CONTRACTION_RE.findall(text)) / n_words,
        "first_person_ratio": sum(1 for w in words if w in FIRST_PERSON) / n_words,
        "long_word_ratio": sum(1 for l in lengths if l >= 7) / n_words,
        "word_length_std": (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5,
    }


def extract_features(text: str) -> dict:
    """Combine all feature functions into a single feature dict for one text sample."""
    sentences = get_sentences(text)
    words = get_words(text)

    return {
        "avg_sentence_length": avg_sentence_length(sentences),
        "sentence_length_variance": sentence_length_variance(sentences),
        "type_token_ratio": type_token_ratio(words),
        "ttr_fixed_window": fixed_window_ttr(words),
        "function_word_ratio": function_word_ratio(words),
        "punctuation_frequency": punctuation_frequency(text),
        "avg_word_length": avg_word_length(words),
        "num_sentences": len(sentences),
        "num_words": len(words),
        **per_function_word_freq(words),
        **punctuation_breakdown(text if isinstance(text, str) else ""),
        **misc_style_features(text if isinstance(text, str) else "", words),
    }


def build_feature_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply extract_features to every row and return a new feature DataFrame."""
    feature_rows = df[text_col].apply(extract_features)
    features_df = pd.DataFrame(list(feature_rows))
    return features_df


def main():
    parser = argparse.ArgumentParser(description="Extract stylometric features from text data.")
    parser.add_argument("--input", required=True, help="Path to input CSV (must have a 'text' column)")
    parser.add_argument("--output", required=True, help="Path to save the output feature CSV")
    parser.add_argument("--text_col", default="text", help="Name of the text column (default: 'text')")
    parser.add_argument("--label_col", default="label", help="Name of the label column, if present")
    parser.add_argument("--keep_cols", nargs="*", default=["qid"], help="Extra columns to copy through (default: qid, the question id)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    features_df = build_feature_dataframe(df, text_col=args.text_col)

    # Carry the label column over if it exists, so this file is ready for modeling
    if args.label_col in df.columns:
        features_df[args.label_col] = df[args.label_col].values
    for col in args.keep_cols:
        if col in df.columns:
            features_df[col] = df[col].values

    features_df.to_csv(args.output, index=False)
    print(f"Saved {len(features_df)} rows of features to {args.output}")
    print("\nPreview:")
    print(features_df.head())


if __name__ == "__main__":
    main()