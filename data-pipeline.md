# DishDash — Backend Low-Level: Data Pipeline

**Component:** End-to-end data ingestion, merge, enrichment, and storage pipeline  
**Notebook:** `DishDash_Milestone3-5.ipynb`  
**Language:** Python 3 (Google Colab environment)

---

## Authentication

### Google Places API
- **Key variable:** `GOOGLE_API_KEY` (Cell 4)
- **Passed as:** HTTP header `X-Goog-Api-Key` on every POST request
- **FieldMask header** (`X-Goog-FieldMask`): Controls which fields are returned and billed. The notebook requests: `places.id`, `places.displayName`, `places.location`, `places.formattedAddress`, `places.types`, `places.rating`, `places.userRatingCount`, `places.priceLevel`, `places.reviews`, `places.businessStatus`
- **Billing tiers:** `displayName`, `location`, `formattedAddress` = Essentials (free 10K/month); `rating`, `reviews`, `priceLevel` = Pro (free 5K/month). A single student-project run uses ~1–5 requests total.

### HuggingFace Inference API
- **Token variable:** `HUGGINGFACE_TOKEN` (Cell 4)
- **Passed as:** HTTP header `Authorization: Bearer <token>`
- **Guard:** The sentiment block checks `if HUGGINGFACE_TOKEN != "YOUR_HF_TOKEN_HERE"` before running. If absent, all sentiment columns are set to `None`.

### Overpass API / Open Food Facts
- No authentication required. Rate limiting is handled with `time.sleep()` delays.

---

## Execution Context

- **Environment:** Google Colab (Python 3, Ubuntu-based)
- **Dependencies:** `requests`, `pandas`, `sqlite3` (stdlib), `math` (stdlib), `time` (stdlib), `thefuzz` + `python-Levenshtein` (installed via `!pip install` in Cell 2)
- **Execution order:** Cells must be run top-to-bottom. Global variables (`GOOGLE_API_KEY`, `NYU_LAT`, `NYU_LNG`, etc.) defined in Cell 4 are required by all downstream cells.
- **Session persistence:** The SQLite file `dishdash.db` is saved to the Colab session's local filesystem (`/content/dishdash.db`). It must be downloaded or copied to Google Drive before the session ends.

---

## Logic Flow (Step by Step)

### Phase 1: OSM Data Collection (`fetch_osm_restaurants`, Cell 7–8)

1. Constructs an Overpass QL query targeting `amenity=restaurant`, `amenity=cafe`, and `amenity=fast_food` within the bounding box `(40.724, -74.002, 40.738, -73.990)`.
2. Sends HTTP GET to `https://overpass-api.de/api/interpreter` with `timeout=90`.
3. On success, iterates over `elements[]` in the JSON response, extracting OSM tags into a list of dicts.
4. Converts to DataFrame; drops rows where `name == "Unknown"`.
5. On failure (status ≠ 200 or timeout), waits and retries up to 3 times with increasing delays (10s, 20s, 30s).
6. If all retries fail, returns an empty DataFrame (handled by the fallback in Cell 12).

**Output columns:** `osm_id`, `name`, `lat`, `lng`, `amenity_type`, `cuisine`, `addr_street`, `addr_number`, `phone`, `website`, `opening_hours`, `wheelchair`

---

### Phase 2: Google Places Data Collection (`fetch_google_places`, Cell 10–11)

1. Validates that `api_key` is not the placeholder string.
2. Sends HTTP POST to `https://places.googleapis.com/v1/places:searchNearby` with:
   - JSON body: `locationRestriction` (circle centered on NYU_LAT/NYU_LNG, radius 800m), `includedTypes` list, `maxResultCount: 20`
   - Headers: `X-Goog-Api-Key`, `X-Goog-FieldMask`
3. Parses `places[]` array; for each place:
   - Extracts `displayName.text` for the venue name
   - Extracts `location.latitude/longitude`
   - Takes up to 3 reviews; joins their text with `" | "` separator
   - Maps `priceLevel` enum to integer (1–4) or `None`
4. Returns DataFrame. On non-200 response, prints error message and returns empty DataFrame.

**Output columns:** `google_place_id`, `name`, `lat`, `lng`, `address`, `google_type`, `all_types`, `price_level`, `google_rating`, `rating_count`, `review_text`, `business_status`

---

### Phase 3: Nutrition Data Collection (`fetch_nutrition_for_cuisine`, Cells 14–16)

1. Iterates over 18 entries in `CUISINE_TO_SEARCH` dict, each mapping a cuisine key to 1–3 food search terms.
2. For each term, sends HTTP GET to `https://world.openfoodfacts.org/cgi/search.pl` with `page_size=5` and fields filter.
3. Extracts `nutriments` sub-dict from each product; skips products without `energy-kcal_100g`.
4. After collecting all products for a cuisine key, computes column-wise means and the modal Nutri-Score.
5. Sleeps 1 second between terms; retries once after 5 seconds on timeout.
6. If no valid products found, returns a row with all `None` values and `off_sample_size=0`.

**Output columns:** `cuisine_key`, `avg_calories_100g`, `avg_fat_100g`, `avg_protein_100g`, `avg_carbs_100g`, `avg_sugar_100g`, `avg_sodium_100g`, `common_nutriscore`, `off_sample_size`

---

### Phase 4: Fuzzy Geo-Merge (`fuzzy_geo_merge`, Cells 18–20)

1. For every OSM row, iterates over all Google rows.
2. Computes `haversine_meters(osm_lat, osm_lng, google_lat, google_lng)`. Skips pairs > 75m apart.
3. For candidates within 75m, computes `fuzz.token_sort_ratio(osm_name.lower(), google_name.lower())`. Skips if < 55.
4. Combined score: `name_score × 0.7 + (1 - dist/75) × 30`. Keeps the Google row with the highest combined score as the match.
5. Constructs the output DataFrame by appending matched Google columns to each OSM row (or `None` if unmatched).

**Pitfall:** This is O(n×m) — fine for ~200 OSM × 20 Google venues, but would need spatial indexing (e.g., a KD-tree) at larger scale.

---

### Phase 5: Cuisine Normalization (`normalize_cuisine`, Cell 21)

1. Splits raw OSM cuisine tags on `;` (OSM allows multi-value tags like `pizza;italian`) and takes the first value.
2. Strips whitespace, lowercases, replaces `-` and spaces with `_`.
3. Checks direct match against `CUISINE_TO_SEARCH` keys.
4. Falls back to an alias dict for common variations (e.g., `"ramen"` → `"japanese"`).
5. Returns `"unknown"` if no match found.

---

### Phase 6: Nutrition Join (Cell 22)

Simple pandas LEFT JOIN: `df_venues.merge(df_nutrition, on='cuisine_key', how='left')`. All OSM venues are retained; nutrition columns are `NaN` for venues with `cuisine_key = 'unknown'` or cuisines with no Open Food Facts data.

---

### Phase 7: Sentiment Analysis (`analyze_sentiment_hf`, Cell 24)

1. Skips rows where `review_text` is empty or `None`.
2. Truncates text to 512 characters before sending (model context limit).
3. Sends HTTP POST to `https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english`.
4. Parses the nested list response: `result[0][0]` for the top label and score.
5. Sleeps 0.3 seconds between requests to avoid rate limiting.
6. On any exception, returns `{sentiment_label: None, sentiment_score: None}`.

---

### Phase 8: SQLite Storage (Cells 28–31)

1. Opens (or creates) `dishdash.db` with `sqlite3.connect()`.
2. Writes three tables using `DataFrame.to_sql(if_exists='replace')` — this drops and recreates the table on each run, ensuring idempotency.
3. Commits and verifies with `SELECT COUNT(*)` per table and a sample analytical query.
4. Closes the connection.

---

## Helper Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `haversine_meters(lat1, lng1, lat2, lng2)` | Cell 18 | Returns distance in meters using the Haversine formula. Uses `math.radians` and `math.atan2`. No external dependencies. |
| `normalize_cuisine(cuisine_raw)` | Cell 21 | Normalizes OSM cuisine tags to standardized keys. Pure string manipulation; no I/O. |

---

## Resource Management

- All `requests` calls use explicit `timeout` parameters to prevent hanging indefinitely.
- The SQLite connection is explicitly closed in Cell 32 (`conn.close()`).
- No file handles are left open; all DataFrames are in-memory.
- The Colab session will lose the `dishdash.db` file if not downloaded or synced to Google Drive (see Cell 34 for the optional Drive sync code).

---

## Known Pitfalls

1. **Overpass API overload:** The public endpoint is shared and can time out during peak hours. The retry logic handles this, but runs may occasionally produce an empty OSM DataFrame. The Cell 12 fallback substitutes Google data in this case.
2. **Google API key exposure:** The API key is hardcoded in Cell 4. In a production setting, this should be loaded from environment variables or a secrets manager (e.g., `os.environ['GOOGLE_API_KEY']` or Colab Secrets).
3. **HuggingFace cold start:** Inference API models can take 20–60 seconds to warm up if not recently used. The first sentiment request may time out; subsequent ones succeed.
4. **Open Food Facts coverage gaps:** Some cuisine keys (e.g., niche or regional cuisines) may return zero products with valid calorie data, resulting in `None` nutrition rows.
5. **Colab session ephemerality:** The generated `dishdash.db` file is lost when the Colab session times out. The optional Google Drive sync in Cell 34 must be run explicitly to persist the file.
