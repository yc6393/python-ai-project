# DishDash — Backend High-Level Overview

---

## Architecture Summary

DishDash's backend is a **data pipeline** implemented in a single Jupyter notebook (`DishDash_Milestone3-5.ipynb`). It runs as a one-time (or periodic) batch job that fetches data from three external APIs, merges it, enriches it with NLP, and stores the result in a SQLite database. There is no persistent web server at this stage; the database file is the backend's output artifact.

---

## Components

### 1. Data Ingestion Pipeline

**Purpose:** Fetch raw venue, rating, and nutrition data from external APIs.

| Component | API | Auth | Rate Limits |
|-----------|-----|------|-------------|
| OSM Fetcher | Overpass API (`overpass-api.de`) | None | 1 request/60s recommended; implemented with retry logic |
| Google Places Fetcher | Places API (New) (`places.googleapis.com`) | API key (GCP free tier: 10K Essentials, 5K Pro requests/month) | Max 20 results per Nearby Search call |
| Nutrition Fetcher | Open Food Facts (`world.openfoodfacts.org`) | None | Polite delay of 1s between requests; implemented with retry logic |

**Execution context:** Runs sequentially in a Google Colab session. All functions are defined as Python callables and invoked in notebook cells.

---

### 2. Merge Engine

**Purpose:** Join the three data sources into a single unified DataFrame.

**Step A — OSM + Google (fuzzy geo-match):**
- Computes Haversine distance between every OSM–Google venue pair
- Filters to pairs within 75 meters of each other
- Applies `thefuzz.fuzz.token_sort_ratio` for name similarity (threshold: 55/100)
- Combines scores as `name_score × 0.7 + proximity_score × 0.3`
- Result: LEFT JOIN keeping all OSM venues, augmenting matched ones with Google data

**Step B — Venues + Nutrition (cuisine key join):**
- `normalize_cuisine()` maps raw OSM tags to a standardized key
- pandas `DataFrame.merge(how='left', on='cuisine_key')` joins aggregated nutrition profiles
- Result: every venue row includes cuisine-level nutritional averages

---

### 3. NLP Enrichment (Sentiment Analysis)

**Purpose:** Add a sentiment signal to venue records that have Google review text.

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` via HuggingFace Inference API
- **Input:** Concatenated text from up to 3 Google reviews, truncated to 512 tokens
- **Output:** `sentiment_label` (POSITIVE/NEGATIVE) and `sentiment_score` (0–1 confidence)
- **Dependency:** Requires a HuggingFace access token; gracefully skipped if token is absent

---

### 4. Storage Layer

**Purpose:** Persist the merged dataset for downstream analytics and (planned) frontend queries.

- **Database:** SQLite (`dishdash.db`) — a single-file, zero-configuration relational database
- **Tables written:** `venues`, `nutrition_profiles`, `data_sources`
- **Write method:** `DataFrame.to_sql(if_exists='replace')` — full table replacement on each run
- **Verification:** Post-write row counts and a sample SQL query are executed to confirm integrity

---

## Scheduled Jobs / Triggers

Currently, the pipeline is **manually triggered** by running the notebook. There is no scheduler. Planned extension: a cron job or GitHub Actions workflow to refresh data weekly.

---

## Error Handling

| Component | Strategy |
|-----------|----------|
| Overpass API | Up to 3 retries with exponential backoff (10s, 20s, 30s) on timeout |
| Open Food Facts | Up to 2 retries per search term with a 5s delay; failed terms are skipped gracefully |
| Google Places | Single attempt; non-200 responses print the error message and return an empty DataFrame |
| HuggingFace | Try/except per review; failures return `None` for both label and score |
| OSM fallback | If OSM returns zero results, the pipeline substitutes Google Places data as the base DataFrame |

---

## Service Integrations

| Service | Integration Type | Purpose |
|---------|-----------------|---------|
| Overpass API | HTTP GET | Geographic POI data |
| Google Places API (New) | HTTP POST with JSON body + FieldMask header | Ratings, reviews, price, categories |
| Open Food Facts API | HTTP GET with query params | Nutrition data by food keyword |
| HuggingFace Inference API | HTTP POST with Bearer token | Sentiment classification |

---

## Planned Backend Extensions (Milestones 4–5)

- **Flask API layer:** REST endpoints over the SQLite database (e.g., `GET /venues?cuisine=pizza&max_price=2`)
- **Expanded data collection:** Additional NYC neighborhoods; Yelp Fusion API for more review data
- **Scheduled refresh:** Weekly data update via cron or GitHub Actions
- **Persistent cloud storage:** Migrate from SQLite to PostgreSQL (e.g., Supabase) for multi-user access
