import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from models.gpt_model import get_gpt_model_and_tokenizer
from utils.dataset_loader import load_sst2_dataset, tokenize_for_gpt
from utils.metrics import compute_metrics, print_metrics

# ── Training Configuration ──────────────────────────────────────────
LEARNING_RATE = 6.25e-5
BATCH_SIZE = 32
EPOCHS = 3
MAX_LENGTH = 128
WARMUP_RATIO = 0.1
# ────────────────────────────────────────────────────────────────────


def train_gpt():
    """
    Fine-tune GPT-1 for sentiment classification.

    How GPT-1 is adapted for classification:
      The original GPT-1 paper (Radford et al., 2018) describes a two-stage approach:
      1) Unsupervised pre-training with autoregressive language modeling.
      2) Supervised fine-tuning where a linear classification head is added on top
         of the transformer's final hidden state at a designated token position.

    Input structure:
      For classification, the input is formatted as:
        <sentence tokens> [CLS]
      The hidden state at the [CLS] position is used for prediction.

    Training objective:
      The fine-tuning loss combines classification cross-entropy with an auxiliary
      language modeling loss:  L = L_cls + λ * L_lm  (λ = 0.5 in the original paper).
      This auxiliary objective improves generalization and accelerates convergence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GPT-1] Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────
    train_data, val_data, test_data = load_sst2_dataset()
    model, tokenizer = get_gpt_model_and_tokenizer(num_labels=2)

    train_dataset = tokenize_for_gpt(train_data, tokenizer, MAX_LENGTH)
    val_dataset = tokenize_for_gpt(val_data, tokenizer, MAX_LENGTH)
    test_dataset = tokenize_for_gpt(test_data, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ── Optimizer & Scheduler ───────────────────────────────────────
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n[GPT-1] Training config:")
    print(f"  Optimizer     : AdamW")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Max seq length: {MAX_LENGTH}")
    print(f"  Warmup steps  : {warmup_steps}")
    print()

    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} — Avg training loss: {avg_loss:.4f}")

        # ── Validation ──────────────────────────────────────────────
        val_metrics = _evaluate(model, val_loader, device)
        print_metrics(val_metrics, f"GPT-1 Epoch {epoch+1} Validation")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Restore best and evaluate on test set ───────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    test_metrics = _evaluate(model, test_loader, device)
    print_metrics(test_metrics, "GPT-1 — Final Test")

    return test_metrics


def _evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return compute_metrics(all_preds, all_labels)


if __name__ == "__main__":
    train_gpt()
