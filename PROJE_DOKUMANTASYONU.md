# Transformer Modelleri ile Duygu Analizi — Proje Dokumantasyonu

## Icindekiler

1. [Projeye Genel Bakis](#1-projeye-genel-bakis)
2. [Proje Klasor Yapisi](#2-proje-klasor-yapisi)
3. [Kullanilan Teknolojiler ve Kutuphaneler](#3-kullanilan-teknolojiler-ve-kutuphaneler)
4. [Part 1 — Veri Seti Secimi ve Yuklenmesi](#4-part-1--veri-seti-secimi-ve-yuklenmesi)
5. [Part 2 — BERT Modelinin Fine-Tuning Sureci](#5-part-2--bert-modelinin-fine-tuning-sureci)
6. [Part 3 — GPT-1 Modelinin Fine-Tuning Sureci](#6-part-3--gpt-1-modelinin-fine-tuning-sureci)
7. [Part 4 — Model Karsilastirmasi](#7-part-4--model-karsilastirmasi)
8. [Part 5 — Kavramsal Sorular](#8-part-5--kavramsal-sorular)
9. [Kodun Calistirilmasi](#9-kodun-calistirilmasi)

---

## 1. Projeye Genel Bakis

Bu proje, **LLM-Driven Software Development** dersi kapsaminda hazirlanan bir odevdir. Amac, transformer tabanli dil modellerinin downstream (alt gorev) gorevlere nasil ince ayar (fine-tuning) yapilarak uyarlandigini anlamak ve iki farkli mimariyi karsilastirmaktir:

- **BERT** (Bidirectional Encoder Representations from Transformers) — Encoder tabanli model
- **GPT-1** (Generative Pre-trained Transformer 1) — Decoder tabanli model

Her iki model de **duygu analizi (sentiment analysis)** gorevi icin ayni veri seti uzerinde fine-tune edilmis ve performanslari karsilastirilmistir.

### Projenin Ana Hedefleri

1. Halka acik bir Ingilizce duygu analizi veri seti secmek ve analiz etmek
2. BERT modelini duygu siniflandirmasi icin fine-tune etmek
3. GPT-1 modelini ayni veri seti uzerinde fine-tune etmek
4. Iki modelin performansini karsilastirmak
5. Transformer mimarisine dair kavramsal sorulari cevaplamak

---

## 2. Proje Klasor Yapisi

Proje modüler bir yapiyla organize edilmistir. Her bir dosyanin ne is yaptigi asagida aciklanmistir:

```
Fine-Tuning/
│
├── main.py                      # Ana calistirma dosyasi (orkestrator)
├── requirements.txt             # Bagimliliklarin listesi
├── report.md                    # Ingilizce akademik rapor (Part 1-5 cevaplari)
├── PROJE_DOKUMANTASYONU.md      # Bu dosya (Turkce detayli anlatim)
│
├── models/                      # Model tanimlari
│   ├── __init__.py              # Python paketi olarak tanimlar
│   ├── bert_model.py            # BERT modelini ve tokenizer'ini yukler
│   └── gpt_model.py             # GPT-1 modelini ve tokenizer'ini yukler
│
├── training/                    # Egitim pipeline'lari
│   ├── __init__.py              # Python paketi olarak tanimlar
│   ├── train_bert.py            # BERT egitim dongusu ve degerlendirme
│   └── train_gpt.py             # GPT-1 egitim dongusu ve degerlendirme
│
├── utils/                       # Yardimci moduller
│   ├── __init__.py              # Python paketi olarak tanimlar
│   ├── dataset_loader.py        # Veri seti yukleme ve tokenizasyon
│   └── metrics.py               # Degerlendirme metrikleri hesaplama
│
└── venv/                        # Python sanal ortami (virtual environment)
```

### Neden Bu Yapi?

- **Modülerlik**: Her dosya tek bir sorumluluga sahiptir. Model tanimlari `models/` klasorunde, egitim mantigi `training/` klasorunde, yardimci fonksiyonlar `utils/` klasorundedir.
- **Tekrar Kullanilabilirlik**: `dataset_loader.py` hem BERT hem GPT-1 icin tokenizasyon fonksiyonlari icerir. `metrics.py` her iki model icin ortak degerlendirme fonksiyonlari sunar.
- **`__init__.py` Dosyalari**: Her alt klasorde bulunan bos `__init__.py` dosyalari, Python'un bu klasorleri **paket (package)** olarak tanimasi icin gereklidir. Bu sayede `from models.bert_model import ...` gibi importlar calisir.

---

## 3. Kullanilan Teknolojiler ve Kutuphaneler

### `requirements.txt` Dosyasi

```
torch
transformers
datasets
scikit-learn
numpy
accelerate
```

| Kutuphane | Amaci |
|---|---|
| **PyTorch (`torch`)** | Derin ogrenme framework'u. Modellerin egitimi, gradyan hesaplama, GPU/CPU yonetimi icin kullanilir. |
| **Hugging Face Transformers (`transformers`)** | Onceden egitilmis BERT ve GPT-1 modellerini, tokenizer'lari ve siniflandirma basliklarini saglar. |
| **Hugging Face Datasets (`datasets`)** | SST-2 veri setini kolayca indirip yuklemek icin kullanilir. Hugging Face Hub uzerinden veri setlerine erisim saglar. |
| **scikit-learn (`scikit-learn`)** | Accuracy, Precision, Recall ve F1 Score metriklerini hesaplamak icin kullanilir. |
| **NumPy (`numpy`)** | Sayisal diziler uzerinde islem yapmak icin kullanilir (tahminleri ve etiketleri birlestirme vb.). |
| **Accelerate (`accelerate`)** | Hugging Face'in cihaz yonetimini (CPU/GPU) kolaylastiran kutuphanesidir. |

---

## 4. Part 1 — Veri Seti Secimi ve Yuklenmesi

### Secilen Veri Seti: IMDB Movie Reviews

**IMDB Movie Reviews (50K)** veri seti secilmistir. Bu veri seti, NLP alanindaki en yaygin duygu analizi benchmark'larindan biridir.

| Ozellik | Deger |
|---|---|
| **Ad** | IMDB Movie Reviews (50K) |
| **Kaynak** | Maas et al., 2011 — *"Learning Word Vectors for Sentiment Analysis"* |
| **Erisim** | Lokal CSV dosyasi (`IMDB Dataset.csv`) |
| **Toplam Ornek** | 50,000 (dengeli: 25K pozitif, 25K negatif) |
| **Etiketler** | Binary — `0` (negatif / "negative"), `1` (pozitif / "positive") |
| **Bolme** | Train: 40,000 / Validation: 5,000 / Test: 5,000 (stratified %80/%10/%10) |

### Neden IMDB?

1. NLP alaninda en bilinen duygu analizi veri setlerinden biridir
2. Dengeli dagılım (25K pozitif, 25K negatif) adil degerlendirme saglar
3. Uzun review'lar modelin anlama kapasitesini test eder
4. Binary siniflandirma oldugu icin model karsilastirmasi kolaydir

### Kod Aciklamasi: `utils/dataset_loader.py`

Bu dosya uc ana fonksiyon icerir:

#### 1. `load_imdb_dataset()` — Veri Setini Yukleme

```python
df = pd.read_csv(DATA_PATH)
df["text"] = df["review"].apply(_clean_html)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
```

Lokal `IMDB Dataset.csv` dosyasi pandas ile okunur. Ardindan:
1. **HTML temizligi**: Review'lardaki `<br />` gibi HTML tag'leri `_clean_html()` fonksiyonu ile kaldirilir
2. **Label donusumu**: "positive"/"negative" string degerleri 1/0 tamsayilara cevirilir

Veri seti stratified (katmanli) olarak bolunur:

```python
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
```

- `stratify=df["label"]`: Her bolumde pozitif/negatif oraninin ayni kalmasini saglar
- `random_state=42`: Tekrarlanabilirlik icin sabit seed
- Sonuc: %80 train (40K), %10 val (5K), %10 test (5K)

#### 2. `tokenize_for_bert()` — BERT icin Tokenizasyon

```python
def tokenize_for_bert(dataset, tokenizer, max_length=256):
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
```

Bu fonksiyon ne yapar:

1. **`tokenizer()`**: Her review'u BERT'in anlayacagi token ID'lerine donusturur. Ornegin `"this movie is great"` → `[101, 2023, 3185, 2003, 2307, 102, 0, 0, ...]`
   - `101` = `[CLS]` token'i (cumle basinda)
   - `102` = `[SEP]` token'i (cumle sonunda)
   - `0` = `[PAD]` token'i (max uzunluga kadar doldurma)
2. **`padding="max_length"`**: Tum review'lari 256 token uzunluguna kadar pad'ler (sifirlarla doldurur)
3. **`truncation=True`**: 256'dan uzun review'lari keser
4. **`rename_column("label", "labels")`**: Hugging Face modelleri `labels` sutun adini bekler
5. **`set_format("torch")`**: Verileri PyTorch tensor formatina cevirir

#### 3. `tokenize_for_gpt()` — GPT-1 icin Tokenizasyon

```python
def tokenize_for_gpt(dataset, tokenizer, max_length=128):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    cls_token = tokenizer.cls_token

    def tokenize_fn(examples):
        texts = [sent + " " + cls_token for sent in examples["sentence"]]
        ...
```

GPT-1'in tokenizasyonu BERT'ten farklidir:

1. **GPT-1'in orijinal tokenizer'inda `[PAD]` ve `[CLS]` tokenlari yoktur**, bu yuzden oncelikle bunlari ekliyoruz
2. **Girdi yapisi**: Her cumlenin sonuna `[CLS]` token'i eklenir: `<cumle> [CLS]`. GPT-1 soldan saga isledigi icin, sondaki `[CLS]` token'i tum onceki tokenlara attend edebilir
3. Geri kalan islemler BERT ile aynidir (padding, truncation, format donusumu)

---

## 5. Part 2 — BERT Modelinin Fine-Tuning Sureci

### BERT Nasil Siniflandirmaya Uyarlanir?

BERT, **encoder-only** bir transformer modelidir. Onceden egitimde (pre-training) iki gorev kullanir:

1. **Masked Language Modeling (MLM)** — Rastgele maskelenmis tokenlari tahmin etme
2. **Next Sentence Prediction (NSP)** — Iki cumlenin ardisik olup olmadigini tahmin etme

Siniflandirma icin uyarlama sureci:

```
Girdi: [CLS] this movie is great [SEP] [PAD] [PAD] ...
                    ↓
         BERT Encoder (12 katman)
                    ↓
         [CLS] token'inin hidden state'i (768 boyutlu vektor)
                    ↓
         Linear katman (768 → 2)
                    ↓
         Logitler → Softmax → Tahmin (pozitif/negatif)
```

`[CLS]` token'i cumlenin **toplu temsilini** (aggregate representation) tasir. Ust tarafa eklenen linear katman, bu temsili sinif sayisina (2) projekte eder.

### Kod Aciklamasi: `models/bert_model.py`

```python
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_NAME = "bert-base-uncased"

def get_bert_model_and_tokenizer(num_labels=2):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    return model, tokenizer
```

- **`bert-base-uncased`**: 12 katmanli, 768 gizli boyutlu, 110M parametreli onceden egitilmis BERT modeli. `uncased` = buyuk/kucuk harf ayrimi yapilmaz.
- **`BertForSequenceClassification`**: Hugging Face'in hazir siniflandirma sinifi. BERT'in uzerine otomatik olarak bir linear siniflandirma basi (classification head) ekler.
- **`num_labels=2`**: Binary siniflandirma (pozitif/negatif)

### Kod Aciklamasi: `training/train_bert.py`

Bu dosya egitim pipeline'inin tamamini icerir.

#### Egitim Konfigurasyonu

```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 3
MAX_LENGTH = 128
WARMUP_RATIO = 0.1
```

| Parametre | Deger | Aciklama |
|---|---|---|
| **Learning Rate** | 2×10⁻⁵ | BERT fine-tuning icin standart oran. Cok yuksek olursa onceden ogrenilen bilgi kaybolur (catastrophic forgetting). |
| **Batch Size** | 16 | Her adimda 16 ornek islenir. IMDB review'lari uzun oldugu icin bellek tasarrufu amaciyla 16 secilmistir. |
| **Epochs** | 3 | Tum veri seti 3 kez gezilir. BERT fine-tuning icin 2-4 epoch yeterlidir. |
| **Max Length** | 256 | Review'larin maksimum token uzunlugu. IMDB review'lari uzun oldugu icin 256 secilmistir. |
| **Warmup Ratio** | 0.1 | Ilk %10 adimda learning rate sifirdan yavasce artar, sonra lineer olarak azalir. |

#### Cihaz Secimi

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

GPU varsa GPU kullanilir (cok daha hizli), yoksa CPU. Bu projede CPU kullanilmistir.

#### Veri Yukleme

```python
train_data, val_data, test_data = load_sst2_dataset()
model, tokenizer = get_bert_model_and_tokenizer(num_labels=2)

train_dataset = tokenize_for_bert(train_data, tokenizer, MAX_LENGTH)
val_dataset = tokenize_for_bert(val_data, tokenizer, MAX_LENGTH)
test_dataset = tokenize_for_bert(test_data, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
```

1. Veri seti yuklenir ve tokenize edilir
2. `DataLoader` ile batch'ler halinde sunulur
3. Egitim verisi `shuffle=True` ile karistirilir (her epoch'ta farkli sirada gorulur)

#### Optimizer ve Scheduler

```python
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
```

- **AdamW**: Adam optimizer'in weight decay duzeltilmis hali. `weight_decay=0.01` overfitting'i onlemeye yardimci olur.
- **Linear Warmup + Decay Scheduler**: Learning rate ilk `warmup_steps` adimda 0'dan `LEARNING_RATE`'e lineer olarak artar, sonra 0'a dogru lineer azalir. Bu, egitimin baslangicinda buyuk gradyanlarla modelin bozulmasini onler.

#### Egitim Dongusu

```python
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
```

Her adimda olan islemler:

1. **Forward Pass**: Batch modele verilir, model logitler ve loss uretir
2. **`loss.backward()`**: Gradyanlar hesaplanir (backpropagation)
3. **`clip_grad_norm_(1.0)`**: Gradyanlar 1.0 normuna kırpılır — bu, gradyan patlamasini (gradient explosion) onler
4. **`optimizer.step()`**: Parametreler guncellenir
5. **`scheduler.step()`**: Learning rate guncellenir
6. **`optimizer.zero_grad()`**: Gradyanlar sifirlanir (sonraki adim icin)

#### En Iyi Modelin Saklanmasi

```python
best_val_f1 = 0.0
best_model_state = None

# ... her epoch sonunda:
val_metrics = _evaluate(model, val_loader, device)
if val_metrics["f1"] > best_val_f1:
    best_val_f1 = val_metrics["f1"]
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Her epoch sonunda validation seti uzerinde degerlendirme yapilir. En yuksek F1 skoruna sahip modelin parametreleri saklanir. Egitim bittikten sonra bu en iyi model yuklenerek test seti uzerinde nihai degerlendirme yapilir. Bu yaklasima **early stopping** / **model selection** denir.

#### Degerlendirme Fonksiyonu

```python
def _evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return compute_metrics(all_preds, all_labels)
```

- **`model.eval()`**: Modeli degerlendirme moduna alir (dropout kapatilir, batch norm donusturulur)
- **`torch.no_grad()`**: Gradyan hesaplamasini kapatir (bellek tasarrufu ve hiz)
- **`torch.argmax(logits, dim=1)`**: Her ornek icin en yuksek logit'e sahip sinifi secer

---

## 6. Part 3 — GPT-1 Modelinin Fine-Tuning Sureci

### GPT-1 Nasil Siniflandirmaya Uyarlanir?

GPT-1, **decoder-only** bir transformer modelidir. Soldan saga (autoregressive) calisir ve orijinal olarak metin uretimi icin tasarlanmistir.

Siniflandirma icin uyarlama sureci:

```
Girdi: this movie is great [CLS]
                    ↓
      GPT-1 Decoder (12 katman, masked self-attention)
                    ↓
      [CLS] pozisyonundaki hidden state (768 boyutlu vektor)
                    ↓
      Linear katman (768 → 2)
                    ↓
      Logitler → Softmax → Tahmin (pozitif/negatif)
```

BERT'ten temel farklar:

1. **`[CLS]` token'i sonda**: BERT'te `[CLS]` basinda, GPT-1'de sondadir. Cunku GPT-1 causal attention kullanir — her token yalnizca kendinden onceki tokenlara bakabilir. `[CLS]` sonda oldugunda tum cumleyi "gorebilir".
2. **Causal (masked) attention**: Her token yalnizca solundaki tokenlara attend edebilir. BERT'te ise her token tum tokenlara bakabilir (bidirectional).
3. **Auxiliary LM loss**: Orijinal GPT-1 makalesinde fine-tuning sirasinda siniflandirma loss'una ek olarak dil modelleme loss'u da kullanilir: `L = L_cls + λ × L_lm` (λ = 0.5).

### Kod Aciklamasi: `models/gpt_model.py`

```python
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification

MODEL_NAME = "openai-gpt"

def get_gpt_model_and_tokenizer(num_labels=2):
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
```

Burada onemli detaylar:

1. **`openai-gpt`**: Hugging Face'teki GPT-1 modeli. 12 katman, 768 gizli boyut, ~117M parametre.
2. **Ozel token ekleme**: GPT-1'in orijinal tokenizer'inda `[PAD]` ve `[CLS]` tokenlari yoktur. Bunlari `add_special_tokens()` ile ekliyoruz.
3. **`resize_token_embeddings()`**: Yeni tokenlar eklendigi icin embedding matrisini yeniden boyutlandirmak gerekir. Yeni tokenlarin embedding'leri rastgele baslatilir.
4. **`pad_token_id`**: Modele hangi token'in padding oldugunu bildiririz, boylece model pad tokenlarini siniflandirma icin ignore edebilir.

### Kod Aciklamasi: `training/train_gpt.py`

GPT-1'in egitim kodu BERT ile buyuk olcude aynidir. Temel farklar:

#### Farkli Learning Rate

```python
LEARNING_RATE = 6.25e-5  # GPT-1 orijinal makalesindeki deger
```

GPT-1 orijinal makalesinde fine-tuning icin 6.25×10⁻⁵ learning rate onerilmistir. Bu, BERT'in 2×10⁻⁵ degerinden yaklasik 3 kat daha yuksektir.

#### Batch Size ve Max Length

BERT ile ayni: `BATCH_SIZE=16`, `MAX_LENGTH=256`. IMDB review'lari uzun oldugu icin max_length 256'ya cikarilmis, GPU bellegini tasirmamak icin batch_size 16'ya dusurulmustur.

#### Geri Kalan Yaklasim Ayni

Egitim dongusu, optimizer (AdamW), scheduler (linear warmup), gradient clipping, validation ile model secimi ve test seti uzerinde nihai degerlendirme BERT ile birebir aynidir. Bu, adil bir karsilastirma yapabilmek icin onemlidir — iki model arasindaki tek fark **mimari** ve **learning rate** olmalidir.

---

## 7. Part 4 — Model Karsilastirmasi

### Kod Aciklamasi: `main.py`

`main.py` dosyasi projenin orkestratoru gorevini gorur:

```python
def main():
    bert_metrics = train_bert()    # BERT'i egit ve test metriklerini al
    gpt_metrics = train_gpt()      # GPT-1'i egit ve test metriklerini al
    compare_models(bert_metrics, gpt_metrics)  # Karsilastir
```

`compare_models()` fonksiyonu iki modelin metriklerini yan yana bir tablo halinde yazdirir:

```python
def compare_models(bert_metrics, gpt_metrics):
    for metric in ["accuracy", "precision", "recall", "f1"]:
        b = bert_metrics[metric]
        g = gpt_metrics[metric]
        diff = b - g
        print(f"  {metric.capitalize():<12} {b:>10.4f} {g:>10.4f} {sign}{diff:>13.4f}")
```

### Mimari Farklar

| Ozellik | BERT | GPT-1 |
|---|---|---|
| Tip | Encoder-only | Decoder-only |
| Attention | Bidirectional (tam) | Causal (maskeli) — yalnizca sola bakar |
| Katman Sayisi | 12 | 12 |
| Gizli Boyut | 768 | 768 |
| Parametre Sayisi | ~110M | ~117M |
| Pre-training | MLM + NSP | Autoregressive LM |
| Bağlam | Sol + sag (bidirectional) | Yalnizca sol (unidirectional) |

### Neden BERT Siniflandirmada Daha Iyi?

1. **Cift yonlu baglam**: BERT tum cumleyi ayni anda gorebilir. "Bu film **harika**" cumlesinde "harika" kelimesinin soldaki ve sagdaki tum kelimelere bakarak anlamini kavrar.
2. **[CLS] token tasarimi**: BERT'in `[CLS]` token'i NSP gorevi sirasinda cumle-seviyesi anlam yakalamak icin egitilmistir.
3. **Gorev uyumu**: Siniflandirma, cumlenin tamamini **butunsel olarak** anlamayi gerektirir. Bidirectional attention bunun icin dogal bir uyumdur.

### GPT-1'in Guclu Yanlari

1. **Metin uretimi**: Autoregressive yapisi sayesinde dogal metin uretebilir
2. **Birlesik cerceve**: Farkli gorevler icin prompt'lar araciligiyla tek bir model kullanilabilir
3. **Regularizer etkisi**: Fine-tuning sirasindaki auxiliary LM loss, modelin genel dil bilgisini korumaya yardimci olur

---

## 8. Part 5 — Kavramsal Sorular

Kavramsal sorularin cevaplari `report.md` dosyasinda Ingilizce olarak verilmistir. Asagida Turkce ozetleri bulunmaktadir:

### Soru 1: Multi-Head Attention Mekanizmasi

Transformer mimarisinin temel yapi tasidir. Her token icin uc vektor hesaplanir:

- **Query (Q)**: "Ne ariyorum?"
- **Key (K)**: "Ne sunuyorum?"
- **Value (V)**: "Icerigim nedir?"

Dikkat formulu:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Multi-head attention'da bu islem `h` paralel "kafa" ile yapilir. Her kafa farkli iliski turlerini ogrenebilir (sozdizimi, anlam, konum vb.), sonuc olarak daha zengin temsiller uretir.

### Soru 2: Makine Cevirisi icin Kayip Fonksiyonu

Cross-entropy loss kullanilir:

```
L = -1/T Σ log P(y_t | y_{<t}, x)
```

Model, dogru hedef token'a dusuk olasilik atadigi zaman cezalandirilir. Egitim sirasinda **tum** model parametreleri (encoder, decoder, embedding, attention agirliklari) guncellenir.

### Soru 3: Decoder'da Masked Self-Attention

Decoder otoregressif olarak calisir — her token yalnizca onceki tokenlere baglidir. Egitim sirasinda tum hedef dizi mevcuttur (teacher forcing), bu nedenle modelin gelecek tokenlara "bakarak kopya cekmesi" engellenmelidir. Causal mask bunu saglar — gelecek pozisyonlara -∞ uygulanir, softmax'tan sonra bu pozisyonlarin agirligi 0 olur.

### Soru 4: BERT Pre-training Gorevleri

1. **MLM (Masked Language Modeling)**: Tokenlarin %15'i maskelenir, model bunlari tahmin eder. %80 `[MASK]`, %10 rastgele token, %10 degistirilmez.
2. **NSP (Next Sentence Prediction)**: Iki cumlenin ardisik olup olmadigini tahmin eder.

Standart dil modelleme BERT icin kullanilmaz cunku BERT'in amaci **cift yonlu** temsil ogrenmektir — standart LM yalnizca tek yonlu calisir.

### Soru 5: GPT-1 Pre-training Gorevi

GPT-1, standart soldan saga dil modelleme gorevi ile egitilir:

```
L = Σ log P(u_i | u_1, ..., u_{i-1})
```

Fine-tuning sirasinda farkli gorevler icin girdi donusumleri kullanilir:
- **Siniflandirma**: `<start> metin <extract>`
- **Cikarsama (Entailment)**: `<start> onculler <ayirici> hipotez <extract>`
- **Benzerlik**: Her iki siralama islenir, sonuclar toplanir
- **Coktan Secmeli**: Her secenek ayri islenir

---

## 9. Kodun Calistirilmasi

### On Kosullar

1. Python 3.10+ yuklu olmalidir
2. Sanal ortam (virtual environment) olusturulmalidir

### Kurulum

```bash
# Sanal ortam olusturma (yapilmamissa)
python -m venv venv

# Sanal ortami aktive etme (Windows)
.\venv\Scripts\activate

# Bagimliliklar yukleme
pip install -r requirements.txt
```

### Calistirma

```bash
python main.py
```

Bu komut sirasiyla:

1. IMDB veri setini lokal CSV'den yukler ve on-isler
2. BERT'i 3 epoch boyunca GPU uzerinde fine-tune eder
3. BERT'in test sonuclarini yazdirir
4. GPT-1'i 3 epoch boyunca GPU uzerinde fine-tune eder
5. GPT-1'in test sonuclarini yazdirir
6. Iki modelin performansini yan yana karsilastirir

### Beklenen Cikti Formati

```
==============================================================
  PHASE 1: Fine-tuning BERT on IMDB
==============================================================
[BERT] Using device: cuda
============================================================
DATASET: IMDB Movie Reviews (50K)
============================================================
  Train       : 40000 samples
  Validation  : 5000 samples
  Test        : 5000 samples
------------------------------------------------------------
  ...egitim ciktilari...

==================================================
  BERT — Final Test — Evaluation Results
==================================================
  Accuracy  : 0.XXXX
  Precision : 0.XXXX
  Recall    : 0.XXXX
  F1 Score  : 0.XXXX
==================================================

  ...ayni GPT-1 icin...

==============================================================
  MODEL COMPARISON — BERT vs GPT-1 on IMDB
==============================================================
  Metric         BERT      GPT-1  Δ (BERT-GPT)
  ------------------------------------------------
  Accuracy     0.XXXX    0.XXXX       +0.XXXX
  Precision    0.XXXX    0.XXXX       +0.XXXX
  Recall       0.XXXX    0.XXXX       +0.XXXX
  F1           0.XXXX    0.XXXX       +0.XXXX
==============================================================
```

### Notlar

- **GPU ile**: NVIDIA GPU otomatik olarak kullanilir (GTX 1060 6GB ile egitim ~20-40 dakika surer)
- **CPU'da calisma suresi**: GPU olmadan egitim saatler surebilir
- **Bellek**: BERT ve GPT-1 modelleri ~2-4GB VRAM kullanir (batch_size=16, max_length=256 ile)

---

## Degerlendirme Metrikleri Aciklamasi

`utils/metrics.py` dosyasinda hesaplanan metrikler:

| Metrik | Formul | Aciklama |
|---|---|---|
| **Accuracy** | `dogru tahmin / toplam` | Tum tahminlerin ne kadari dogru? |
| **Precision** | `TP / (TP + FP)` | Pozitif dedigimizin ne kadari gercekten pozitif? |
| **Recall** | `TP / (TP + FN)` | Gercek pozitiflerin ne kadarini yakaladik? |
| **F1 Score** | `2 × (P × R) / (P + R)` | Precision ve Recall'un harmonik ortalamasi |

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(predictions, labels):
    preds = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
```

- `average="binary"`: Binary siniflandirma icin (pozitif sinif uzerinden hesaplanir)
- `zero_division=0`: Bolme hatasi durumunda 0 dondurur (ornegin hic pozitif tahmin yoksa)
