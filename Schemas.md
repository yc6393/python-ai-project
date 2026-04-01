# DishDash — Database Schema Reference

**Database:** SQLite (`dishdash.db`)  
**Created by:** `DishDash_Milestone3-5.ipynb`, Cells 28–31

---

## Tables

### 1. `venues`

The primary merged table. Each row is one named restaurant/cafe/food venue near NYU. Data is sourced from OpenStreetMap (OSM), Google Places API, Open Food Facts, and HuggingFace NLP.

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `osm_id` | INTEGER | OSM | OpenStreetMap node ID (unique per OSM element) |
| `name` | TEXT | OSM | Venue name |
| `lat` | REAL | OSM | Latitude |
| `lng` | REAL | OSM | Longitude |
| `amenity_type` | TEXT | OSM | OSM amenity tag: `restaurant`, `cafe`, or `fast_food` |
| `cuisine` | TEXT | OSM | Raw OSM cuisine tag (e.g., `pizza`, `italian;american`) |
| `addr_street` | TEXT | OSM | Street name |
| `addr_number` | TEXT | OSM | Building/house number |
| `phone` | TEXT | OSM | Phone number (if tagged) |
| `website` | TEXT | OSM | Website URL (if tagged) |
| `opening_hours` | TEXT | OSM | OSM opening hours string (e.g., `Mo-Fr 09:00-21:00`) |
| `wheelchair` | TEXT | OSM | Wheelchair accessibility (`yes`, `no`, `limited`) |
| `google_place_id` | TEXT | Google | Google Places unique place ID |
| `google_type` | TEXT | Google | Primary Google place type (e.g., `restaurant`) |
| `all_types` | TEXT | Google | All Google place types (pipe-separated string) |
| `price_level` | TEXT | Google | Price level string (e.g., `PRICE_LEVEL_MODERATE`) |
| `google_rating` | REAL | Google | Google rating (1.0–5.0) |
| `rating_count` | INTEGER | Google | Total number of Google ratings |
| `review_text` | TEXT | Google | Concatenated text of top 3 Google reviews |
| `match_score` | REAL | Computed | Fuzzy geo-match combined score (name similarity + proximity) |
| `match_distance_m` | REAL | Computed | Haversine distance between OSM and Google coordinates (meters) |
| `cuisine_key` | TEXT | Computed | Normalized cuisine key used to join nutrition data |
| `avg_calories_100g` | REAL | Open Food Facts | Average calories per 100g for this cuisine type |
| `avg_fat_100g` | REAL | Open Food Facts | Average fat per 100g |
| `avg_protein_100g` | REAL | Open Food Facts | Average protein per 100g |
| `avg_carbs_100g` | REAL | Open Food Facts | Average carbohydrates per 100g |
| `avg_sugar_100g` | REAL | Open Food Facts | Average sugars per 100g |
| `avg_sodium_100g` | REAL | Open Food Facts | Average sodium per 100g |
| `common_nutriscore` | TEXT | Open Food Facts | Most common Nutri-Score grade for this cuisine (A–E) |
| `off_sample_size` | INTEGER | Open Food Facts | Number of products sampled for the nutrition profile |
| `sentiment_label` | TEXT | HuggingFace NLP | Sentiment classification: `POSITIVE` or `NEGATIVE` |
| `sentiment_score` | REAL | HuggingFace NLP | Confidence score for the sentiment label (0.0–1.0) |

**Key constraints:**
- `osm_id` is the natural primary key (not explicitly declared in SQLite, but unique per OSM node)
- `google_place_id` is nullable (NULL when OSM venue had no Google match)
- All Open Food Facts columns are nullable (NULL when `cuisine_key` = `unknown` or no products found)
- `sentiment_label` and `sentiment_score` are nullable when `review_text` is absent or the HuggingFace token was not provided

---

### 2. `nutrition_profiles`

Cuisine-level nutritional lookup table. One row per cuisine key. Used as the right side of the LEFT JOIN that populates nutrition columns in `venues`.

| Column | Type | Description |
|--------|------|-------------|
| `cuisine_key` | TEXT | Primary key; normalized cuisine identifier (e.g., `pizza`, `italian`) |
| `avg_calories_100g` | REAL | Mean calories per 100g across sampled products |
| `avg_fat_100g` | REAL | Mean fat per 100g |
| `avg_protein_100g` | REAL | Mean protein per 100g |
| `avg_carbs_100g` | REAL | Mean carbohydrates per 100g |
| `avg_sugar_100g` | REAL | Mean sugars per 100g |
| `avg_sodium_100g` | REAL | Mean sodium per 100g |
| `common_nutriscore` | TEXT | Modal Nutri-Score grade across sampled products |
| `off_sample_size` | INTEGER | Number of Open Food Facts products included in averages |

**Coverage:** 18 cuisine keys defined; actual coverage depends on API availability at collection time.

---

### 3. `data_sources`

Metadata table documenting the three external data sources used in the pipeline.

| Column | Type | Description |
|--------|------|-------------|
| `source_name` | TEXT | Human-readable name of the data source |
| `url` | TEXT | API endpoint URL |
| `data_type` | TEXT | Description of the data provided |
| `auth_required` | TEXT | Whether an API key or token is required |
| `records_fetched` | INTEGER | Count of records collected during the run |

---

## Relationships

```
nutrition_profiles.cuisine_key  ←──(LEFT JOIN)──  venues.cuisine_key
```

The `venues` and `nutrition_profiles` tables are the analytical core. The `data_sources` table is informational only and has no foreign key relationships.

---

## Denormalization Notes

The schema is intentionally denormalized: nutrition averages are duplicated across all `venues` rows sharing the same `cuisine_key`. This is appropriate for the read-heavy, single-user analytics use case and avoids the overhead of additional JOIN queries in the frontend. If the project scales to a production multi-user system, normalizing nutrition data into a separate lookup join would be preferable.
