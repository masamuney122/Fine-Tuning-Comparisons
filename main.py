"""
Homework 1: Sentiment Analysis with Transformer Models
=======================================================
This script fine-tunes BERT (encoder) and GPT-1 (decoder) on the SST-2 dataset,
evaluates both, and prints a side-by-side comparison.
"""

from training.train_bert import train_bert
from training.train_gpt import train_gpt


def compare_models(bert_metrics, gpt_metrics):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 62)
    print("  MODEL COMPARISON — BERT vs GPT-1 on SST-2")
    print("=" * 62)
    header = f"  {'Metric':<12} {'BERT':>10} {'GPT-1':>10} {'Δ (BERT-GPT)':>14}"
    print(header)
    print("  " + "-" * 48)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        b = bert_metrics[metric]
        g = gpt_metrics[metric]
        diff = b - g
        sign = "+" if diff >= 0 else ""
        print(f"  {metric.capitalize():<12} {b:>10.4f} {g:>10.4f} {sign}{diff:>13.4f}")

    print("=" * 62)

    if bert_metrics["f1"] > gpt_metrics["f1"]:
        winner = "BERT"
    elif gpt_metrics["f1"] > bert_metrics["f1"]:
        winner = "GPT-1"
    else:
        winner = "Tie"
    print(f"\n  Winner (by F1): {winner}\n")

    print("  Discussion:")
    print("  - BERT is an encoder-only model with bidirectional self-attention.")
    print("    It sees the full context (left + right) for every token, making it")
    print("    naturally suited for understanding/classification tasks.")
    print("  - GPT-1 is a decoder-only model with masked (causal) self-attention.")
    print("    It processes tokens left-to-right, which is ideal for generation")
    print("    but less optimal for classification where full context helps.")
    print("  - BERT's bidirectional attention typically yields better sentence-level")
    print("    representations, giving it an edge on sentiment analysis.")
    print("  - GPT-1's strength lies in generative and few-shot tasks rather than")
    print("    fixed classification benchmarks.")
    print()


def main():
    print("=" * 62)
    print("  PHASE 1: Fine-tuning BERT on SST-2")
    print("=" * 62)
    bert_metrics = train_bert()

    print("\n" + "=" * 62)
    print("  PHASE 2: Fine-tuning GPT-1 on SST-2")
    print("=" * 62)
    gpt_metrics = train_gpt()

    compare_models(bert_metrics, gpt_metrics)


if __name__ == "__main__":
    main()
