"""
Prepare Dataset — HC3 (Human ChatGPT Comparison Corpus)
---------------------------------------------------------
Downloads HC3's all.jsonl (or reads a local copy) and converts it into a
simple CSV with two columns:
    text  - an answer
    label - 0 = human-written, 1 = AI-generated (ChatGPT)
    qid   - which question the answer belongs to (used so that answers to the
            same question never end up in both train and test)

The CSV can be passed straight into feature_engineering.py.

Usage:
    python prepare_dataset.py --output dataset.csv
    python prepare_dataset.py --input all.jsonl --output dataset.csv   # if you downloaded it manually
"""

import argparse
import json
import os
import urllib.request

import pandas as pd

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"


def download(url: str, dest: str):
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise SystemExit(
            f"\nDownload failed: {e}\n\n"
            "Fix: open this link in your browser, let it download, then re-run with --input:\n"
            f"  {url}\n"
            "  python prepare_dataset.py --input all.jsonl --output dataset.csv\n"
            "(If the error mentions SSL/CERTIFICATE, that's the same macOS certificate issue as NLTK.)"
        )


def load_rows(path: str):
    """Flatten HC3: each question has a list of human answers and a list of ChatGPT answers."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for qid, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for ans in item.get("human_answers", []):
                rows.append({"text": ans, "label": 0, "qid": qid})
            for ans in item.get("chatgpt_answers", []):
                rows.append({"text": ans, "label": 1, "qid": qid})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a labeled human-vs-AI CSV from HC3.")
    parser.add_argument("--input", default=None, help="Path to a local all.jsonl (downloads it if omitted)")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--min_words", type=int, default=50, help="Drop answers shorter than this (default: 50, needed for the fixed-window TTR)")
    parser.add_argument("--max_per_class", type=int, default=3000, help="Samples kept per class (default: 3000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = args.input
    if path is None:
        path = "all.jsonl"
        if not os.path.exists(path):
            download(HC3_URL, path)

    df = load_rows(path)
    print(f"Loaded {len(df)} answers (human: {(df.label == 0).sum()}, AI: {(df.label == 1).sum()})")

    # Clean: drop empty / very short answers (sentence-based features need some length)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() >= args.min_words]
    df = df.drop_duplicates(subset="text")

    # Balance the classes so accuracy isn't inflated by one class dominating
    n = min(args.max_per_class, df.label.value_counts().min())
    df = pd.concat([
        df[df.label == 0].sample(n=n, random_state=args.seed),
        df[df.label == 1].sample(n=n, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    df[["text", "label", "qid"]].to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows ({n} per class) to {args.output}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()