"""
DishDash — Data Pipeline
========================
Run this once (or on a schedule) to populate dishdash.db.
The Streamlit app reads from that database; it never hits the APIs directly.

Usage:
    python pipeline.py

Requirements:
    pip install requests pandas thefuzz python-Levenshtein
"""

import requests
import pandas as pd
import sqlite3
import time
import math
import os
from thefuzz import fuzz

# ── API Keys ─────────────────────────────────────────────────
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "AIzaSyAsv4bYyskX4ney6fmrzjVDUo_CWGM-qv8")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")

# ── Search area: NYU / Greenwich Village ─────────────────────
NYU_LAT, NYU_LNG = 40.7295, -73.9965
BBOX = (40.7240, -74.0020, 40.7380, -73.9900)   # S, W, N, E
SEARCH_RADIUS_M  = 800
DB_PATH          = "dishdash.db"

# ── Cuisine → Open Food Facts keyword mapping ─────────────────
CUISINE_MAP = {
    "pizza":         ["pizza"],
    "italian":       ["pasta", "pizza", "risotto"],
    "chinese":       ["noodles", "fried rice", "dumplings"],
    "japanese":      ["sushi", "ramen", "miso"],
    "sushi":         ["sushi", "sashimi"],
    "mexican":       ["burrito", "taco", "nachos"],
    "burger":        ["burger", "hamburger"],
    "american":      ["burger", "sandwich", "fries"],
    "indian":        ["curry", "naan", "biryani"],
    "thai":          ["pad thai", "curry", "spring roll"],
    "mediterranean": ["hummus", "falafel", "pita"],
    "french":        ["croissant", "baguette", "quiche"],
    "korean":        ["bibimbap", "kimchi", "bulgogi"],
    "vietnamese":    ["pho", "banh mi", "spring roll"],
    "sandwich":      ["sandwich", "sub"],
    "coffee_shop":   ["coffee", "latte", "muffin"],
    "cafe":          ["coffee", "pastry", "sandwich"],
    "unknown":       ["meal", "food"],
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_cuisine(raw):
    if not raw or raw == "unknown":
        return "unknown"
    p = raw.split(";")[0].strip().lower().replace("-", "_").replace(" ", "_")
    if p in CUISINE_MAP:
        return p
    aliases = {
        "coffee": "coffee_shop", "espresso": "coffee_shop",
        "ramen": "japanese", "tacos": "mexican", "burritos": "mexican",
        "kebab": "mediterranean", "falafel": "mediterranean",
        "pho": "vietnamese", "noodle": "chinese",
    }
    for alias, key in aliases.items():
        if alias in p:
            return key
    return "unknown"


# ─────────────────────────────────────────────────────────────
# STEP 1 — Overpass / OpenStreetMap
# ─────────────────────────────────────────────────────────────

def fetch_osm(bbox, retries=2):
    s, w, n, e = bbox
    q = (
        "[out:json][timeout:50];"
        f"(node[\"amenity\"=\"restaurant\"]({s},{w},{n},{e});"
        f" node[\"amenity\"=\"cafe\"]({s},{w},{n},{e});"
        f" node[\"amenity\"=\"fast_food\"]({s},{w},{n},{e}););"
        "out body;"
    )
    for attempt in range(retries):
        try:
            r = requests.get(
                "https://overpass-api.de/api/interpreter",
                params={"data": q}, timeout=55
            )
            if r.status_code != 200:
                continue
            rows = []
            for el in r.json().get("elements", []):
                t = el.get("tags", {})
                name = t.get("name", "")
                if not name:
                    continue
                rows.append({
                    "name":          name,
                    "lat":           el["lat"],
                    "lng":           el["lon"],
                    "cuisine":       t.get("cuisine", "unknown"),
                    "amenity_type":  t.get("amenity", "restaurant"),
                    "addr":          (t.get("addr:housenumber", "") + " " + t.get("addr:street", "")).strip(),
                    "phone":         t.get("phone", ""),
                    "website":       t.get("website", ""),
                    "opening_hours": t.get("opening_hours", ""),
                    "wheelchair":    t.get("wheelchair", ""),
                })
            if rows:
                print(f"  OSM: {len(rows)} venues")
                return pd.DataFrame(rows)
        except Exception as exc:
            print(f"  OSM attempt {attempt + 1} failed: {exc}")
            time.sleep(5)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# STEP 2 — Google Places API (New)
# ─────────────────────────────────────────────────────────────

def fetch_google(lat, lng, radius, key, n=20):
    if not key:
        return pd.DataFrame()
    PRICE = {
        "PRICE_LEVEL_INEXPENSIVE": "$",
        "PRICE_LEVEL_MODERATE":    "$$",
        "PRICE_LEVEL_EXPENSIVE":   "$$$",
        "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
    }
    try:
        r = requests.post(
            "https://places.googleapis.com/v1/places:searchNearby",
            headers={
                "Content-Type":   "application/json",
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": (
                    "places.id,places.displayName,places.location,"
                    "places.formattedAddress,places.types,places.rating,"
                    "places.userRatingCount,places.priceLevel,places.reviews"
                ),
            },
            json={
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": radius,
                    }
                },
                "includedTypes":  ["restaurant", "cafe", "fast_food_restaurant", "coffee_shop"],
                "maxResultCount": n,
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  Google Places error {r.status_code}")
            return pd.DataFrame()
        rows = []
        for p in r.json().get("places", []):
            name = p.get("displayName", {}).get("text", "")
            if not name:
                continue
            loc   = p.get("location", {})
            types = p.get("types", [])
            useful = [t for t in types if t not in ("point_of_interest", "establishment", "food")]
            rev = " | ".join(
                x.get("text", {}).get("text", "")
                for x in p.get("reviews", [])[:3]
                if x.get("text", {}).get("text")
            )
            rows.append({
                "gname":        name,
                "lat":          loc.get("latitude"),
                "lng":          loc.get("longitude"),
                "address":      p.get("formattedAddress", ""),
                "google_type":  useful[0] if useful else (types[0] if types else ""),
                "price_level":  PRICE.get(p.get("priceLevel", ""), ""),
                "google_rating": p.get("rating"),
                "rating_count": p.get("userRatingCount", 0),
                "review_text":  rev,
            })
        print(f"  Google: {len(rows)} venues")
        return pd.DataFrame(rows)
    except Exception as exc:
        print(f"  Google Places failed: {exc}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# STEP 3 — Fuzzy geo-merge (same as Milestone 3)
# ─────────────────────────────────────────────────────────────

def fuzzy_merge(df_osm, df_g, max_dist=75, min_score=55):
    extras = ["address", "google_type", "price_level",
              "google_rating", "rating_count", "review_text"]
    if df_g.empty:
        for c in extras:
            df_osm[c] = None
        return df_osm
    g_recs = df_g.to_dict("records")
    matched = []
    for _, osm in df_osm.iterrows():
        best, best_s = None, 0
        for g in g_recs:
            if pd.isna(g.get("lat")) or pd.isna(g.get("lng")):
                continue
            d  = haversine_m(osm["lat"], osm["lng"], g["lat"], g["lng"])
            if d > max_dist:
                continue
            ns = fuzz.token_sort_ratio(osm["name"].lower(), g["gname"].lower())
            if ns < min_score:
                continue
            score = ns * 0.7 + (1 - d / max_dist) * 30
            if score > best_s:
                best_s, best = score, {c: g.get(c) for c in extras}
        matched.append(best or {c: None for c in extras})
    merged = pd.concat([df_osm.reset_index(drop=True), pd.DataFrame(matched)], axis=1)
    matched_count = sum(1 for m in matched if m.get("google_rating") is not None)
    print(f"  Merge: {matched_count}/{len(df_osm)} OSM venues matched to Google")
    return merged


# ─────────────────────────────────────────────────────────────
# STEP 4 — Open Food Facts nutrition
# ─────────────────────────────────────────────────────────────

def fetch_nutrition():
    BASE = "https://world.openfoodfacts.org/cgi/search.pl"
    results = []
    for cuisine, terms in CUISINE_MAP.items():
        rows = []
        for term in terms:
            try:
                r = requests.get(
                    BASE,
                    params={"search_terms": term, "json": 1, "page_size": 5,
                            "fields": "nutriments,nutriscore_grade"},
                    timeout=20,
                )
                if r.status_code != 200:
                    continue
                for p in r.json().get("products", []):
                    nm = p.get("nutriments", {})
                    if nm.get("energy-kcal_100g") is not None:
                        rows.append({
                            "cal":  nm.get("energy-kcal_100g", 0),
                            "fat":  nm.get("fat_100g", 0),
                            "pro":  nm.get("proteins_100g", 0),
                            "carb": nm.get("carbohydrates_100g", 0),
                            "ns":   p.get("nutriscore_grade", ""),
                        })
            except Exception:
                pass
            time.sleep(0.4)
        if rows:
            tmp  = pd.DataFrame(rows)
            nsvs = tmp["ns"].replace("", pd.NA).dropna()
            results.append({
                "cuisine_key": cuisine,
                "avg_cal":     round(tmp["cal"].mean(), 1),
                "avg_fat":     round(tmp["fat"].mean(), 1),
                "avg_protein": round(tmp["pro"].mean(), 1),
                "avg_carbs":   round(tmp["carb"].mean(), 1),
                "nutriscore":  nsvs.mode().iloc[0] if len(nsvs) else None,
            })
        else:
            results.append({
                "cuisine_key": cuisine, "avg_cal": None, "avg_fat": None,
                "avg_protein": None, "avg_carbs": None, "nutriscore": None,
            })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# STEP 5 — HuggingFace sentiment (optional)
# ─────────────────────────────────────────────────────────────

def analyze_sentiment(text, token):
    if not text or not token:
        return None, None
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    try:
        r = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text[:512]},
            timeout=15,
        )
        if r.status_code == 200:
            result = r.json()
            if result and isinstance(result, list):
                top = result[0][0] if isinstance(result[0], list) else result[0]
                return top.get("label"), round(top.get("score", 0), 3)
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n🍜 DishDash — Data Pipeline\n" + "=" * 40)

    print("\n[1/4] Fetching OpenStreetMap venues...")
    df_osm = fetch_osm(BBOX)
    if df_osm.empty:
        print("  ⚠️  OSM unavailable — aborting. Try again or use demo data.")
        return

    print("\n[2/4] Fetching Google Places ratings...")
    df_g = fetch_google(NYU_LAT, NYU_LNG, SEARCH_RADIUS_M, GOOGLE_API_KEY)

    print("\n[3/4] Merging sources...")
    df = fuzzy_merge(df_osm, df_g)
    df["cuisine_key"] = df["cuisine"].apply(normalize_cuisine)

    print("\n[4/4] Fetching Open Food Facts nutrition...")
    df_nutr = fetch_nutrition()
    df = df.merge(df_nutr, on="cuisine_key", how="left")

    # Sentiment (optional — slow, comment out if not needed)
    if HUGGINGFACE_TOKEN:
        print("\n[+] Running sentiment analysis...")
        labels, scores = [], []
        for _, row in df.iterrows():
            text = row.get("review_text", "")
            lbl, sc = analyze_sentiment(str(text) if text else "", HUGGINGFACE_TOKEN)
            labels.append(lbl)
            scores.append(sc)
            time.sleep(0.3)
        df["sentiment_label"] = labels
        df["sentiment_score"]  = scores
    else:
        df["sentiment_label"] = None
        df["sentiment_score"]  = None

    # Clean
    df = df.dropna(subset=["lat", "lng", "name"]).reset_index(drop=True)
    df["google_rating"] = pd.to_numeric(df.get("google_rating"), errors="coerce")
    df["avg_cal"]       = pd.to_numeric(df.get("avg_cal"), errors="coerce")

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("venues", conn, if_exists="replace", index=False)
    df_nutr.to_sql("nutrition_profiles", conn, if_exists="replace", index=False)
    pd.DataFrame([
        {"source": "OpenStreetMap", "records": len(df_osm)},
        {"source": "Google Places", "records": len(df_g)},
        {"source": "Open Food Facts", "records": len(df_nutr)},
    ]).to_sql("data_sources", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"\n✅ Pipeline complete!")
    print(f"   Venues saved   : {len(df)}")
    print(f"   With ratings   : {df['google_rating'].notna().sum()}")
    print(f"   With nutrition : {df['avg_cal'].notna().sum()}")
    print(f"   Database       : {DB_PATH}")


if __name__ == "__main__":
    run_pipeline()
