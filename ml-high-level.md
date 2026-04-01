# DishDash — ML High-Level Overview

---

## Current ML Systems

### 1. Sentiment Analysis (Production — Milestone 3)

**Purpose:** Classify the sentiment of Google review text attached to each restaurant venue, adding a human-interpretable signal beyond the raw star rating.

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`  
**Source:** HuggingFace Hub (Stanford Sentiment Treebank fine-tune of DistilBERT)  
**Inference:** HuggingFace Inference API (hosted; no local GPU required)  
**Integration point:** Runs after the OSM + Google + Open Food Facts merge; results are stored in the `venues` table as `sentiment_label` and `sentiment_score`

**Data source:** `review_text` column — concatenated text of up to 3 Google reviews per venue  
**Output:** Binary label (`POSITIVE` / `NEGATIVE`) + confidence score (0.0–1.0)

---

## Planned ML Systems (Milestones 4–5)

### 2. Rating Prediction / Regression (Planned)

**Purpose:** Predict a venue's Google rating from features available without needing crowd-sourced reviews — enabling scoring of new or under-reviewed restaurants.

**Candidate features:**
- Cuisine type (encoded categorically)
- Price level (ordinal: 1–4)
- Nutritional profile (calories, Nutri-Score)
- Geographic proximity to NYU campus center
- Amenity type (restaurant vs. cafe vs. fast food)

**Candidate models:** Linear regression, Ridge regression, Random Forest Regressor  
**Evaluation metric:** RMSE and R² on a train/test split  
**Data source:** `venues` table rows where `google_rating IS NOT NULL`

---

### 3. Venue Clustering (Planned)

**Purpose:** Discover natural groupings of restaurants to surface non-obvious similarity (e.g., "affordable healthy cafes" vs. "late-night indulgent spots").

**Approach:** K-Means or DBSCAN on normalized features (rating, price, calories, cuisine embedding)  
**Integration:** Cluster labels stored back into the `venues` table and surfaced in the frontend as a "Similar to..." feature

---

### 4. Personalized Recommendation (Stretch Goal)

**Purpose:** Given a user's stated preferences (cuisine, budget, health goals), rank venues using a lightweight scoring function or collaborative filtering.

**Approach:** Content-based filtering initially; collaborative filtering if user interaction data is collected via the Milestone 5 frontend

---

## Overall Training / Inference Pipeline

```
Data in SQLite (venues table)
        │
        ├──[Feature engineering]──> Pandas DataFrame
        │                           - encode categoricals
        │                           - normalize numerics
        │                           - handle NaN (imputation)
        │
        ├──[Train/test split]──> scikit-learn models (regression, clustering)
        │
        ├──[Evaluation]──> metrics logged in notebook
        │
        └──[Inference]──> predictions written back to SQLite
                          or served via Flask API endpoint
```

The sentiment model runs at **data collection time** (inference only; no training). All planned supervised models will train on the collected dataset and be evaluated in the analytics notebook.

---

## Integration with Backend

- Trained model predictions (cluster labels, predicted ratings) will be stored as new columns in the `venues` table via `UPDATE` SQL statements.
- The HuggingFace Inference API is called during the data pipeline run (Cell 24) and requires no local model weights.
- All feature engineering will be implemented in Python/Pandas within the notebook before being refactored into reusable modules for the Milestone 5 backend.
