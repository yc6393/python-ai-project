"""
DishDash — Content-based restaurant recommender.

Uses nearest-neighbour search in a 6-dimensional normalised feature space:
  geo distance (miles) | rating | price level | nutri-score | sentiment | cuisine match
"""

import math
import random
import sqlite3

import numpy as np
import pandas as pd

DB_PATH   = "dishdash.db"
NYU_LAT, NYU_LNG = 40.7295, -73.9965

PRICE_MAP = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
NS_MAP    = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_venues() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM venues", conn)
        conn.close()
        if not df.empty:
            return df.reset_index(drop=True)
    except Exception:
        pass
    return _make_demo()


def _make_demo() -> pd.DataFrame:
    """Mirror of app.py demo dataset so the recommender works without a live DB."""
    random.seed(42)
    venues = [
        ("Bleecker Street Pizza", "pizza",         "$",    4.6,  0.003, -0.002, "c"),
        ("Mamoun's Falafel",      "mediterranean", "$",    4.5,  0.001,  0.004, "b"),
        ("Joe Shanghai",          "chinese",       "$$",   4.4,  0.002, -0.003, "c"),
        ("Bar Pitti",             "italian",       "$$$",  4.5, -0.001,  0.002, "c"),
        ("Sushi Noz",             "sushi",         "$$$$", 4.8,  0.004,  0.001, "a"),
        ("Superiority Burger",    "burger",        "$",    4.5, -0.002,  0.005, "c"),
        ("The Grey Dog",          "cafe",          "$$",   4.4,  0.005, -0.001, "b"),
        ("Raoul's",               "french",        "$$$",  4.7, -0.003, -0.003, "c"),
        ("Kunjip",                "korean",        "$$",   4.2,  0.006,  0.003, "b"),
        ("Pho Grand",             "vietnamese",    "$$",   4.3, -0.005,  0.002, "b"),
        ("Dos Toros Taqueria",    "mexican",       "$",    4.1,  0.002,  0.006, "c"),
        ("Cafe Reggio",           "cafe",          "$",    4.5, -0.001, -0.004, "b"),
        ("Lucali",                "pizza",         "$$",   4.9,  0.007,  0.003, "d"),
        ("Spice Symphony",        "indian",        "$$",   4.2, -0.004,  0.004, "c"),
        ("Pure Thai Cookhouse",   "thai",          "$$",   4.4,  0.003, -0.005, "b"),
        ("Takahachi",             "japanese",      "$$",   4.5, -0.006,  0.001, "b"),
        ("Corner Bistro",         "burger",        "$",    4.3,  0.001,  0.007, "d"),
        ("Blue Hill",             "american",      "$$$$", 4.7, -0.004, -0.002, "b"),
        ("Buvette",               "french",        "$$$",  4.6,  0.005,  0.005, "c"),
        ("Ippudo NY",             "japanese",      "$$",   4.5, -0.007,  0.002, "b"),
        ("Xi'an Famous Foods",    "chinese",       "$",    4.3,  0.004, -0.006, "c"),
        ("Westville",             "american",      "$$",   4.3,  0.008,  0.001, "b"),
        ("Num Pang Kitchen",      "sandwich",      "$$",   4.3, -0.002,  0.006, "c"),
        ("Taim Falafel",          "mediterranean", "$$",   4.4,  0.006, -0.004, "b"),
        ("Lupa Osteria Romana",   "italian",       "$$$",  4.6,  0.001,  0.008, "c"),
        ("Okonomi Ramen",         "japanese",      "$$",   4.4, -0.003, -0.005, "b"),
        ("Artichoke Basilles",    "pizza",         "$",    4.2,  0.009,  0.002, "d"),
        ("Murray Cheese Bar",     "american",      "$$$",  4.6, -0.008, -0.001, "d"),
        ("Thai Villa",            "thai",          "$$",   4.1,  0.007, -0.007, "b"),
        ("Integral Yoga Natural", "cafe",          "$",    4.4, -0.009,  0.003, "a"),
    ]
    rows = []
    streets = ["Bleecker St", "MacDougal St", "Thompson St", "Sullivan St", "W 4th St", "6th Ave"]
    for name, cuisine, price, rating, dlat, dlng, ns in venues:
        rows.append({
            "name":            name,
            "cuisine_key":     cuisine,
            "price_level":     price,
            "google_rating":   rating,
            "lat":             round(NYU_LAT + dlat + random.uniform(-0.0007, 0.0007), 6),
            "lng":             round(NYU_LNG + dlng + random.uniform(-0.0007, 0.0007), 6),
            "address":         f"{random.randint(1, 300)} {random.choice(streets)}, New York NY",
            "nutriscore":      ns,
            "sentiment_label": random.choice(["POSITIVE", "POSITIVE", "POSITIVE", "NEGATIVE"]),
            "sentiment_score": round(random.uniform(0.72, 0.99), 3),
        })
    return pd.DataFrame(rows)


# ── Feature engineering ────────────────────────────────────────────────────────

def _haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _sentiment_to_float(label, score) -> float:
    """Maps (label, confidence) → [0, 1] where 1 = strongly positive."""
    lbl = str(label or "").upper()
    sc = float(score) if score is not None and str(score) not in ("nan", "None") else 0.5
    if lbl == "POSITIVE":
        return sc
    if lbl == "NEGATIVE":
        return 1.0 - sc
    return 0.5


def _minmax(arr: np.ndarray, lo: float = None, hi: float = None) -> np.ndarray:
    """Min-max normalise to [0, 1]; NaN → 0.5 (neutral)."""
    valid = arr[~np.isnan(arr)]
    lo = lo if lo is not None else (float(valid.min()) if len(valid) else 0.0)
    hi = hi if hi is not None else (float(valid.max()) if len(valid) else 1.0)
    span = hi - lo
    if span == 0:
        return np.where(np.isnan(arr), 0.5, 0.5)
    normed = (arr - lo) / span
    return np.where(np.isnan(arr), 0.5, np.clip(normed, 0.0, 1.0))


def _build_feature_matrix(df: pd.DataFrame,
                           target_lat: float, target_lng: float,
                           target_cuisine: str) -> np.ndarray:
    """
    Returns a (n_venues, 6) array of normalised features.

    Columns
    -------
    0  geo_dist   : haversine distance from target restaurant, normalised to [0, 1]
    1  rating     : google_rating normalised over fixed [1, 5] scale
    2  price      : price_level ($→1 … $$$$→4) normalised over [1, 4]
    3  nutriscore : nutri-score (a→1 … e→5) normalised over [1, 5]
    4  sentiment  : POSITIVE confidence in [0, 1]
    5  cuisine    : 0 = same cuisine as target, 1 = different
    """
    n = len(df)

    # 1. Geo distance (miles) from each venue to the target venue
    geo = np.array([
        _haversine_miles(row["lat"], row["lng"], target_lat, target_lng)
        if pd.notna(row.get("lat")) and pd.notna(row.get("lng")) else np.nan
        for _, row in df.iterrows()
    ])

    # 2. Rating — prefer google_rating, fall back to a generic "rating" column
    rating_col = "google_rating" if "google_rating" in df.columns else "rating"
    rating = pd.to_numeric(df.get(rating_col), errors="coerce").values.astype(float)

    # 3. Price — map $…$$$$ → 1…4; unknown → midpoint 2.5
    price_col = "price_level" if "price_level" in df.columns else None
    if price_col:
        price = df[price_col].map(PRICE_MAP).fillna(2.5).values.astype(float)
    else:
        price = np.full(n, 2.5)

    # 4. Nutri-score — real DB uses "common_nutriscore"; demo uses "nutriscore"
    ns_col = next(
        (c for c in ("nutriscore", "common_nutriscore") if c in df.columns),
        None,
    )
    if ns_col:
        nutriscore = (
            df[ns_col].astype(str).str.lower().map(NS_MAP).fillna(3.0).values.astype(float)
        )
    else:
        nutriscore = np.full(n, 3.0)

    # 5. Sentiment → scalar confidence in [0, 1]
    if "sentiment_label" in df.columns:
        sentiment = np.array([
            _sentiment_to_float(row.get("sentiment_label"), row.get("sentiment_score"))
            for _, row in df.iterrows()
        ])
    else:
        sentiment = np.full(n, 0.5)

    # 6. Cuisine match — 0 if same, 1 if different
    cuisine_col = "cuisine_key" if "cuisine_key" in df.columns else None
    if cuisine_col:
        cuisine_diff = (df[cuisine_col] != target_cuisine).astype(float).values
    else:
        cuisine_diff = np.zeros(n)

    # Normalise continuous dimensions
    # geo: normalise against the dataset range (target's own distance is always 0)
    geo_n        = _minmax(geo)
    # rating / price / nutriscore: normalise against fixed semantic scales
    rating_n     = _minmax(rating,     lo=1.0, hi=5.0)
    price_n      = _minmax(price,      lo=1.0, hi=4.0)
    nutriscore_n = _minmax(nutriscore, lo=1.0, hi=5.0)
    # sentiment and cuisine_diff are already in [0, 1]

    return np.column_stack([geo_n, rating_n, price_n, nutriscore_n, sentiment, cuisine_diff])


# ── Weights ────────────────────────────────────────────────────────────────────
# Each entry corresponds to one feature dimension (order matches _build_feature_matrix).
# Increase a weight to make that dimension matter more when finding the closest match.

DEFAULT_WEIGHTS = {
    "geo":        1.0,   # physical proximity
    "rating":     1.5,   # google rating (more discriminating than sentiment)
    "price":      1.0,   # price tier
    "nutriscore": 0.75,  # nutritional quality
    "sentiment":  0.5,   # review sentiment (noisier signal, lower weight)
    "cuisine":    2.0,   # same cuisine is a strong similarity signal
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_alternative(restaurant: str, weights: dict = None) -> dict | None:
    """
    Return the most similar alternative to *restaurant* in the DishDash database.

    Methodology
    -----------
    Each venue is represented as a point in a 6-dimensional feature space.
    Continuous features (rating, price, nutri-score) are min-max normalised to
    [0, 1] using fixed semantic scales so dimensions are comparable regardless
    of their raw ranges.  Geo distance is the haversine miles from the query
    restaurant to each candidate, also normalised to [0, 1] relative to the
    dataset spread.  Cuisine is a binary 0/1 flag (same / different).
    Sentiment is mapped from (label, confidence) to a [0, 1] scalar.

    The best match is the venue minimising the weighted Euclidean distance::

        d = sqrt( sum_i( w_i * (f_i_candidate - f_i_target)^2 ) )

    where w_i are the per-dimension weights (see DEFAULT_WEIGHTS).  The query
    restaurant itself is excluded from the result.

    Parameters
    ----------
    restaurant : str
        Name of the restaurant to match.  Case-insensitive substring search;
        the first match is used as the query point.
    weights : dict, optional
        Per-dimension weight overrides.  Valid keys: ``"geo"``, ``"rating"``,
        ``"price"``, ``"nutriscore"``, ``"sentiment"``, ``"cuisine"``.
        Omitted keys fall back to ``DEFAULT_WEIGHTS``.
        Example: ``{"cuisine": 4.0, "geo": 0.0}``

    Returns
    -------
    dict or None
        ``None`` if *restaurant* is not found or has no coordinates.
        Otherwise a dict with keys:

        name : str
            Name of the recommended alternative.
        cuisine : str
            Normalised cuisine key (e.g. ``"italian"``, ``"sushi"``).
        rating : float or None
            Google rating (1–5).
        price : str
            Price tier (``"$"`` … ``"$$$$"``).
        nutriscore : str
            Nutri-Score letter (``"a"`` … ``"e"``).
        sentiment : str
            Majority review sentiment (``"POSITIVE"`` or ``"NEGATIVE"``).
        address : str
            Street address, if available.
        distance_score : float
            Weighted L2 distance to the query restaurant (lower = more similar).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    weight_vec = np.array([w["geo"], w["rating"], w["price"],
                            w["nutriscore"], w["sentiment"], w["cuisine"]])

    df = _load_venues()

    # Locate the target by case-insensitive substring match
    name_lower = restaurant.strip().lower()
    mask = df["name"].str.lower().str.contains(name_lower, na=False, regex=False)
    if not mask.any():
        return None

    target_idx = int(df[mask].index[0])
    target = df.loc[target_idx]

    # Guard: need valid coordinates to compute geo distances
    if pd.isna(target.get("lat")) or pd.isna(target.get("lng")):
        return None

    features = _build_feature_matrix(
        df,
        target_lat=float(target["lat"]),
        target_lng=float(target["lng"]),
        target_cuisine=str(target.get("cuisine_key", "")),
    )

    target_vec = features[target_idx]

    # Weighted L2 distance from target to every other venue
    distances = np.sqrt(((features - target_vec) ** 2 * weight_vec).sum(axis=1))
    distances[target_idx] = np.inf   # exclude the target itself

    best_idx = int(np.argmin(distances))
    best = df.loc[best_idx]

    ns_col = next((c for c in ("nutriscore", "common_nutriscore") if c in df.columns), None)

    return {
        "name":           str(best["name"]),
        "cuisine":        str(best.get("cuisine_key", "") or ""),
        "rating":         best.get("google_rating"),
        "price":          str(best.get("price_level", "") or ""),
        "nutriscore":     str(best.get(ns_col, "") or "") if ns_col else "",
        "sentiment":      str(best.get("sentiment_label", "") or ""),
        "address":        str(best.get("address", "") or ""),
        "distance_score": round(float(distances[best_idx]), 4),
    }


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = ["Bleecker Street Pizza", "Sushi Noz", "The Grey Dog", "Raoul's", "Kunjip"]

    print("=== Default weights ===")
    for name in test_cases:
        alt = get_alternative(name)
        if alt:
            print(f"  {name:28s} -> {alt['name']:28s}  cuisine={alt['cuisine']:14s}  "
                  f"rating={alt['rating']}  price={alt['price']:4s}  score={alt['distance_score']}")

    print("\n=== Cuisine-heavy weights (cuisine=4.0, rating=0.5) ===")
    for name in test_cases:
        alt = get_alternative(name, weights={"cuisine": 4.0, "rating": 0.5})
        if alt:
            print(f"  {name:28s} -> {alt['name']:28s}  cuisine={alt['cuisine']:14s}  "
                  f"rating={alt['rating']}  price={alt['price']:4s}  score={alt['distance_score']}")
