from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification

MODEL_NAME = "openai-gpt"


def get_gpt_model_and_tokenizer(num_labels=2):
    """
    Load the pretrained GPT-1 model with a classification head.

    Architecture:
      GPT-1 (decoder-only transformer) processes tokens left-to-right.
      For classification, a special [CLS] token is appended to the input.
      The hidden state at the position of [CLS] is fed into a linear classification
      head that produces `num_labels` logits.

      Input + [CLS] -> GPT-1 Decoder (12 layers, masked self-attention) -> [CLS] hidden -> Linear -> logits

    Training objective during fine-tuning:
      GPT-1 uses an auxiliary language modeling loss alongside the classification
      cross-entropy loss. The combined objective is:
        L = L_cls + λ * L_lm
      where λ is a weighting coefficient (typically 0.5).
    """
    tokenizer = OpenAIGPTTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    model = OpenAIGPTForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
