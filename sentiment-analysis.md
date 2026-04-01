# DishDash — ML Low-Level: Sentiment Analysis Pipeline

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`  
**Pipeline function:** `analyze_sentiment_hf()` — Cell 24, `DishDash_Milestone3-5.ipynb`

---

## Overview

The sentiment analysis pipeline classifies Google review text for each DishDash venue as either POSITIVE or NEGATIVE, with a confidence score. It uses HuggingFace's hosted Inference API rather than running the model locally, so no GPU or model download is required.

---

## Data Preprocessing

### Input
- **Source column:** `review_text` in `df_final` — a string containing up to 3 Google reviews concatenated with `" | "` as a separator
- **Filter:** Rows where `review_text` is `None`, empty string, or whitespace-only are skipped; their sentiment columns are set to `None`

### Text Truncation
- Input text is sliced to `[:512]` characters before being sent to the API
- **Rationale:** The DistilBERT tokenizer has a 512-token context limit. Character-level truncation is an approximation (tokens ≠ characters), but for English review text the 512-character slice reliably stays under the token limit while preserving the most semantically relevant content (the beginning of the review)
- **Limitation:** For venues with very long concatenated reviews, the third review may be entirely cut off

---

## Model Architecture

**Base model:** DistilBERT — a distilled (compressed) version of BERT that retains ~97% of BERT's performance at ~60% of the size and ~2× the inference speed.

**Fine-tuning dataset:** Stanford Sentiment Treebank v2 (SST-2) — a binary sentiment classification dataset of movie reviews. The model was fine-tuned to classify text as POSITIVE or NEGATIVE.

**Architecture details:**
- 6 transformer layers (vs. 12 in BERT-base)
- 768 hidden dimensions
- 12 attention heads
- ~66M parameters

**Note:** SST-2 is a movie review dataset, not a restaurant review dataset. This domain mismatch may reduce accuracy on food-specific language (e.g., the model may not correctly weight terms like "greasy" or "overpriced" compared to a domain-specific model). This is a known limitation.

---

## Hyperparameters

No training is performed by DishDash — inference only. The model weights are fixed as published on HuggingFace Hub. No hyperparameters are tunable at inference time beyond text truncation length.

---

## Inference Procedure

```python
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {hf_token}"}
payload = {"inputs": text[:512]}
response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
```

**Response format:**
```json
[[{"label": "POSITIVE", "score": 0.997}, {"label": "NEGATIVE", "score": 0.003}]]
```

The pipeline reads `result[0][0]` — the top-scoring label — and extracts `label` and `score`.

**Rate limiting:** A `time.sleep(0.3)` delay is inserted between requests. HuggingFace's free Inference API is rate-limited; without this delay, rapid consecutive calls may return 429 errors.

---

## Training Procedure

Not applicable — this is an inference-only integration using a pretrained HuggingFace model. No fine-tuning is performed on DishDash data.

---

## Evaluation Metrics

No formal evaluation has been run against a DishDash-specific labeled dataset. Informal assessment: the model correctly classifies obviously positive reviews ("best pizza in NYC, service was fantastic") and obviously negative ones ("waited 45 minutes, food was cold"). Edge cases include sarcasm and mixed reviews (positive food, negative service).

**Planned evaluation (Milestone 4):** Manually label a sample of 50 venues' reviews and compute precision, recall, and F1 against model predictions.

---

## Deployment Details

- **Deployment type:** External SaaS API (HuggingFace Inference API)
- **No model hosting required** by DishDash infrastructure
- **Dependency:** Valid HuggingFace access token stored in `HUGGINGFACE_TOKEN`
- **Fallback:** If token is absent or any request fails, sentiment columns default to `None`; the rest of the pipeline continues unaffected

---

## Versioning

The model is pinned implicitly to whatever version HuggingFace currently serves for `distilbert-base-uncased-finetuned-sst-2-english`. There is no explicit version pinning in the notebook. If the model is updated on HuggingFace Hub, results may change between runs.

---

## Monitoring Strategies

Currently none automated. Manual checks:
- Print `df_final["sentiment_label"].value_counts()` after the pipeline runs to verify both labels appear
- Inspect `df_final[df_final["sentiment_score"] < 0.6]` for low-confidence predictions that may warrant review

---

## Known Limitations and Biases

1. **Domain mismatch:** Trained on movie reviews (SST-2), applied to restaurant reviews. Restaurant-specific sentiment cues may be underweighted.
2. **Language:** English only. Non-English reviews (possible given NYU's international community) will be classified unreliably.
3. **Review aggregation:** Up to 3 reviews are concatenated. If a venue has 2 positive and 1 very negative review, the concatenated text may yield a POSITIVE label that masks genuine negative feedback.
4. **Character truncation:** The 512-character truncation may cut off the most critical part of a longer review.
5. **Model coldstart:** The HuggingFace Inference API may take 20–60 seconds to warm up the model on first call. This can cause the first request to time out (15s timeout in the code), returning `None` for that venue's sentiment.
