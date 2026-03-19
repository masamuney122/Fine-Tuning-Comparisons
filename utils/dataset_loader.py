import os
import re
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "IMDB Dataset.csv")


def _clean_html(text):
    """Strip HTML tags like <br /> from review text."""
    return re.sub(r"<[^>]+>", " ", text).strip()


def load_imdb_dataset():
    """
    Load the IMDB Movie Reviews dataset from a local CSV file.

    Dataset: IMDB 50K Movie Reviews
    - Source: Maas et al., 2011 — "Learning Word Vectors for Sentiment Analysis"
              https://ai.stanford.edu/~amaas/data/sentiment/
    - Labels: "positive" / "negative" → mapped to 1 / 0
    - Total: 50,000 reviews (balanced: 25K positive, 25K negative)
    - Split: 40,000 train / 5,000 validation / 5,000 test
    """
    df = pd.read_csv(DATA_PATH)

    df["text"] = df["review"].apply(_clean_html)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df[["text", "label"]]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    print("=" * 60)
    print("DATASET: IMDB Movie Reviews (50K)")
    print("=" * 60)
    print(f"  Source      : Maas et al. 2011 / Kaggle IMDB 50K")
    print(f"  Labels      : 0 (negative), 1 (positive)")
    print(f"  Train       : {len(train_dataset)} samples")
    print(f"  Validation  : {len(val_dataset)} samples")
    print(f"  Test        : {len(test_dataset)} samples")
    print("-" * 60)
    print("Example records:")
    for i in range(min(3, len(train_dataset))):
        label_str = "positive" if train_dataset[i]["label"] == 1 else "negative"
        print(f"  [{label_str}] {train_dataset[i]['text'][:120]}...")
    print("=" * 60)

    return train_dataset, val_dataset, test_dataset


def tokenize_for_bert(dataset, tokenizer, max_length=256):
    """Tokenize a dataset for BERT."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def tokenize_for_gpt(dataset, tokenizer, max_length=256):
    """
    Tokenize a dataset for GPT-1 (openai-gpt).
    GPT-1 uses the last non-pad token representation for classification.
    Input structure: <review text> [CLS]
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    cls_token = tokenizer.cls_token

    def tokenize_fn(examples):
        texts = [t + " " + cls_token for t in examples["text"]]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return encoded

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized
