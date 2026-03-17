import os
from datasets import load_dataset


def load_sst2_dataset():
    """
    Load the SST-2 (Stanford Sentiment Treebank, binary) dataset.

    SST-2 is a binary sentiment classification dataset from the GLUE benchmark.
    - Source: Socher et al., 2013 — "Recursive Deep Models for Semantic Compositionality
      Over a Sentiment Treebank" (https://nlp.stanford.edu/sentiment/)
    - Available via Hugging Face: glue/sst2
    - Labels: 0 = negative, 1 = positive
    - Train: 67,349 samples | Validation: 872 samples
    - The official test set has hidden labels, so we split validation into val + test.
    """
    dataset = load_dataset("glue", "sst2")

    train_dataset = dataset["train"]

    val_test = dataset["validation"].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    print("=" * 60)
    print("DATASET: SST-2 (Stanford Sentiment Treebank — Binary)")
    print("=" * 60)
    print(f"  Source      : GLUE benchmark / Socher et al. 2013")
    print(f"  Labels      : 0 (negative), 1 (positive)")
    print(f"  Train       : {len(train_dataset)} samples")
    print(f"  Validation  : {len(val_dataset)} samples")
    print(f"  Test        : {len(test_dataset)} samples")
    print("-" * 60)
    print("Example records:")
    for i in range(min(3, len(train_dataset))):
        label_str = "positive" if train_dataset[i]["label"] == 1 else "negative"
        print(f"  [{label_str}] {train_dataset[i]['sentence'][:100]}")
    print("=" * 60)

    return train_dataset, val_dataset, test_dataset


def tokenize_for_bert(dataset, tokenizer, max_length=128):
    """Tokenize a dataset for BERT."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def tokenize_for_gpt(dataset, tokenizer, max_length=128):
    """
    Tokenize a dataset for GPT-1 (openai-gpt).
    GPT-1 uses the last non-pad token representation for classification.
    We structure the input as: <sentence> <cls_token>
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    cls_token = tokenizer.cls_token

    def tokenize_fn(examples):
        texts = [sent + " " + cls_token for sent in examples["sentence"]]
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
