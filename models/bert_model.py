from transformers import BertForSequenceClassification, BertTokenizer

MODEL_NAME = "bert-base-uncased"


def get_bert_model_and_tokenizer(num_labels=2):
    """
    Load a pretrained BERT model with a sequence classification head.

    Architecture:
      BERT (encoder-only transformer) produces contextual embeddings for every token.
      For classification, the [CLS] token representation is passed through a linear
      layer that projects it to `num_labels` logits.

      Input  -> BERT Encoder (12 layers) -> [CLS] hidden state -> Linear -> logits
    """
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    return model, tokenizer
