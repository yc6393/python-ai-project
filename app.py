"""
DishDash — Streamlit Web App
=============================
Run with:   streamlit run app.py
Make sure dishdash.db exists first (run pipeline.py once).
"""

import sqlite3
import math
import random

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from recommender import get_alternative

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DishDash",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
NYU_LAT, NYU_LNG = 40.7295, -73.9965
SEARCH_RADIUS_M  = 800
DB_PATH          = "dishdash.db"

NS_COLOR = {
    "a": "#038141", "b": "#85BB2F", "c": "#FECB02",
    "d": "#EE8100", "e": "#E63E11",
}
PRICE_ORDER = {"": 0, None: 0, "$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
PALETTE = [
    "#0f3460", "#16213e", "#1a1a2e", "#533483", "#e94560",
    "#0f8a5f", "#f5a623", "#2255cc", "#c0392b", "#27ae60",
    "#8e44ad", "#2980b9", "#d35400", "#1abc9c",
]

CUISINE_MAP = {
    "pizza": ["pizza"], "italian": ["pasta", "pizza", "risotto"],
    "chinese": ["noodles", "fried rice", "dumplings"],
    "japanese": ["sushi", "ramen", "miso"], "sushi": ["sushi", "sashimi"],
    "mexican": ["burrito", "taco", "nachos"], "burger": ["burger", "hamburger"],
    "american": ["burger", "sandwich", "fries"], "indian": ["curry", "naan", "biryani"],
    "thai": ["pad thai", "curry", "spring roll"],
    "mediterranean": ["hummus", "falafel", "pita"],
    "french": ["croissant", "baguette", "quiche"],
    "korean": ["bibimbap", "kimchi", "bulgogi"],
    "vietnamese": ["pho", "banh mi", "spring roll"],
    "sandwich": ["sandwich", "sub"], "coffee_shop": ["coffee", "latte", "muffin"],
    "cafe": ["coffee", "pastry", "sandwich"], "unknown": ["meal", "food"],
}

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #f7f6f3; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f3460;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #e8edf5 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label { color: #b8c7e0 !important; font-size: 13px !important; }
section[data-testid="stSidebar"] h2 { color: #ffffff !important; font-size: 18px !important; }

/* Hero banner */
.dd-hero {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dd-hero::before {
    content: '🍜';
    position: absolute;
    right: 32px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.12;
}
.dd-hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -1px;
}
.dd-hero p { color: #a8bcda; font-size: 15px; margin: 0 0 16px 0; }
.dd-pill {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    color: #e0e8f5;
    margin-right: 8px;
    margin-top: 4px;
}

/* Restaurant card */
.dd-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border: 1px solid #ececec;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    transition: box-shadow 0.2s;
}
.dd-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.10); }
.dd-card-left { flex: 1; }
.dd-card-right { text-align: right; min-width: 90px; }
.dd-card-name { font-size: 16px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
.dd-cuisine-tag {
    display: inline-block;
    background: #eef2ff;
    color: #3347cc;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: capitalize;
    margin-right: 5px;
}
.dd-detail { font-size: 12px; color: #777; margin-top: 5px; }
.dd-rating { font-size: 24px; font-weight: 800; color: #1a1a2e; line-height: 1; }
.dd-stars  { color: #f5a623; font-size: 12px; }
.dd-price  { font-size: 13px; color: #2d8a4e; font-weight: 700; margin-top: 3px; }
.dd-ns-badge {
    display: inline-block;
    color: #fff;
    border-radius: 5px;
    padding: 1px 7px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    margin-right: 5px;
}
.dd-sentiment-pos { color: #2d8a4e; font-weight: 600; }
.dd-sentiment-neg { color: #c0392b; font-weight: 600; }

/* Section header */
.dd-section {
    font-size: 20px;
    font-weight: 700;
    color: #1a1a2e;
    border-left: 4px solid #0f3460;
    padding-left: 12px;
    margin: 24px 0 14px;
}

/* Metric cards */
.dd-metric {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #ececec;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    text-align: center;
}
.dd-metric-val { font-size: 30px; font-weight: 800; color: #0f3460; }
.dd-metric-lbl { font-size: 12px; color: #888; margin-top: 2px; }

/* Empty state */
.dd-empty { text-align: center; color: #bbb; padding: 48px 0; font-size: 15px; }

/* Tabs — always show icon + label text, never truncate */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    white-space: nowrap;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 600;
    color: #000000 !important;
}
.stTabs [data-baseweb="tab"] p { display: inline !important; color: #000000 !important; }
.stTabs [data-baseweb="tab"] * { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    """Load from SQLite, fall back to demo data if DB not found."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM venues", conn)
        conn.close()
        df["google_rating"] = pd.to_numeric(df.get("google_rating"), errors="coerce")
        df["avg_cal"]       = pd.to_numeric(df.get("avg_cal"), errors="coerce")
        for c in ["price_level", "cuisine_key", "address", "nutriscore",
                  "sentiment_label", "avg_fat", "avg_protein", "avg_carbs"]:
            if c not in df.columns:
                df[c] = None
        return df, "live"
    except Exception:
        return make_demo(), "demo"


def make_demo():
    random.seed(42)
    venues = [
        ("Bleecker Street Pizza", "pizza",         "$",    4.6,  0.003, -0.002, "c", 285),
        ("Mamoun's Falafel",      "mediterranean", "$",    4.5,  0.001,  0.004, "b", 195),
        ("Joe Shanghai",          "chinese",        "$$",   4.4,  0.002, -0.003, "c", 340),
        ("Bar Pitti",             "italian",        "$$$",  4.5, -0.001,  0.002, "c", 520),
        ("Sushi Noz",             "sushi",          "$$$$", 4.8,  0.004,  0.001, "a", 180),
        ("Superiority Burger",    "burger",         "$",    4.5, -0.002,  0.005, "c", 410),
        ("The Grey Dog",          "cafe",           "$$",   4.4,  0.005, -0.001, "b", 250),
        ("Raoul's",               "french",         "$$$",  4.7, -0.003, -0.003, "c", 580),
        ("Kunjip",                "korean",         "$$",   4.2,  0.006,  0.003, "b", 340),
        ("Pho Grand",             "vietnamese",     "$$",   4.3, -0.005,  0.002, "b", 290),
        ("Dos Toros Taqueria",    "mexican",        "$",    4.1,  0.002,  0.006, "c", 430),
        ("Cafe Reggio",           "cafe",           "$",    4.5, -0.001, -0.004, "b", 220),
        ("Lucali",                "pizza",          "$$",   4.9,  0.007,  0.003, "d", 295),
        ("Spice Symphony",        "indian",         "$$",   4.2, -0.004,  0.004, "c", 410),
        ("Pure Thai Cookhouse",   "thai",           "$$",   4.4,  0.003, -0.005, "b", 360),
        ("Takahachi",             "japanese",       "$$",   4.5, -0.006,  0.001, "b", 390),
        ("Corner Bistro",         "burger",         "$",    4.3,  0.001,  0.007, "d", 560),
        ("Blue Hill",             "american",       "$$$$", 4.7, -0.004, -0.002, "b", 340),
        ("Buvette",               "french",         "$$$",  4.6,  0.005,  0.005, "c", 510),
        ("Ippudo NY",             "japanese",       "$$",   4.5, -0.007,  0.002, "b", 430),
        ("Xi'an Famous Foods",    "chinese",        "$",    4.3,  0.004, -0.006, "c", 370),
        ("Westville",             "american",       "$$",   4.3,  0.008,  0.001, "b", 380),
        ("Num Pang Kitchen",      "sandwich",       "$$",   4.3, -0.002,  0.006, "c", 380),
        ("Taim Falafel",          "mediterranean",  "$$",   4.4,  0.006, -0.004, "b", 200),
        ("Lupa Osteria Romana",   "italian",        "$$$",  4.6,  0.001,  0.008, "c", 530),
        ("Okonomi Ramen",         "japanese",       "$$",   4.4, -0.003, -0.005, "b", 390),
        ("Artichoke Basilles",    "pizza",          "$",    4.2,  0.009,  0.002, "d", 310),
        ("Murray Cheese Bar",     "american",       "$$$",  4.6, -0.008, -0.001, "d", 490),
        ("Thai Villa",            "thai",           "$$",   4.1,  0.007, -0.007, "b", 380),
        ("Integral Yoga Natural", "cafe",           "$",    4.4, -0.009,  0.003, "a", 180),
    ]
    streets = ["Bleecker St", "MacDougal St", "Thompson St", "Sullivan St", "W 4th St", "6th Ave"]
    rows = []
    for name, cuisine, price, rating, dlat, dlng, ns, cal in venues:
        rows.append({
            "name":            name,
            "cuisine_key":     cuisine,
            "price_level":     price,
            "google_rating":   rating,
            "rating_count":    random.randint(60, 3000),
            "lat":             round(NYU_LAT + dlat + random.uniform(-0.0007, 0.0007), 6),
            "lng":             round(NYU_LNG + dlng + random.uniform(-0.0007, 0.0007), 6),
            "amenity_type":    "cafe" if cuisine in ("cafe", "coffee_shop") else "restaurant",
            "address":         f"{random.randint(1,300)} {random.choice(streets)}, New York NY",
            "review_text":     f"Great {cuisine.replace('_',' ')} near NYU!",
            "avg_cal":         float(cal),
            "avg_fat":         round(cal * 0.11, 1),
            "avg_protein":     round(cal * 0.08, 1),
            "avg_carbs":       round(cal * 0.15, 1),
            "nutriscore":      ns,
            "sentiment_label": random.choice(["POSITIVE", "POSITIVE", "POSITIVE", "NEGATIVE"]),
            "sentiment_score": round(random.uniform(0.72, 0.99), 3),
        })
    return pd.DataFrame(rows)


# ── Card renderer ─────────────────────────────────────────────

def render_card(row):
    r     = row.get("google_rating")
    r_str = f"{r:.1f}" if pd.notna(r) else "—"
    stars = ("★" * int(round(r)) + "☆" * (5 - int(round(r)))) if pd.notna(r) else ""
    price = row.get("price_level") or ""
    ns    = str(row.get("nutriscore") or "").lower()
    ns_bg = NS_COLOR.get(ns, "#aaa")
    ns_html = f'<span class="dd-ns-badge" style="background:{ns_bg}">{ns.upper()}</span>' if ns else ""
    cuisine = str(row.get("cuisine_key") or "").replace("_", " ").title()
    cal   = row.get("avg_cal")
    cal_s = f"{int(cal)} kcal/100g" if pd.notna(cal) else ""
    addr  = row.get("address") or row.get("addr") or ""
    rc    = row.get("rating_count")
    rc_s  = f"({int(rc):,} reviews)" if rc else ""
    sent  = row.get("sentiment_label")
    sent_html = ""
    if sent:
        cls = "pos" if sent == "POSITIVE" else "neg"
        icon = "😊" if sent == "POSITIVE" else "😞"
        sent_html = f'<span class="dd-sentiment-{cls}">{icon} {sent.title()}</span>'

    return f"""
    <div class="dd-card">
      <div class="dd-card-left">
        <div class="dd-card-name">{row['name']}</div>
        <span class="dd-cuisine-tag">{cuisine}</span>
        <div class="dd-detail" style="margin-top:7px">{ns_html}{cal_s}</div>
        <div class="dd-detail">{sent_html}</div>
        <div class="dd-detail">{addr}</div>
      </div>
      <div class="dd-card-right">
        <div class="dd-rating">{r_str}</div>
        <div class="dd-stars">{stars}</div>
        <div class="dd-detail" style="font-size:11px">{rc_s}</div>
        <div class="dd-price">{price}</div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

df, data_src = load_data()

# ── Sidebar filters ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍜 DishDash")
    st.markdown("---")

    search_q = st.text_input("Search by name", placeholder="e.g. pizza, sushi…")

    cuisines = sorted(df["cuisine_key"].dropna().unique().tolist())
    cuisine_sel = st.selectbox("Cuisine", ["All cuisines"] + cuisines)

    prices = ["Any price", "$", "$$", "$$$", "$$$$"]
    price_sel = st.selectbox("Price level", prices)

    min_rating = st.slider("Minimum rating ⭐", 0.0, 5.0, 0.0, 0.5)

    ns_opts = {"Any": "any", "A only (best)": "a", "A or B": "b", "A, B, or C": "c"}
    ns_sel = st.selectbox("Nutri-Score ≤", list(ns_opts.keys()))

    sort_opts = ["Rating ↓", "Name A–Z", "Price ↑", "Calories ↑"]
    sort_sel = st.selectbox("Sort by", sort_opts)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:12px;color:#8aa0c0'>"
        f"Data source: {'🟢 Live (SQLite)' if data_src == 'live' else '🔵 Demo'}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if data_src == "demo":
        st.info("Run `python pipeline.py` to load live data.", icon="ℹ️")

# ── Apply filters ─────────────────────────────────────────────
filtered = df.copy()
if search_q:
    filtered = filtered[filtered["name"].str.lower().str.contains(search_q.lower(), na=False)]
if cuisine_sel != "All cuisines":
    filtered = filtered[filtered["cuisine_key"] == cuisine_sel]
if price_sel != "Any price":
    filtered = filtered[filtered["price_level"] == price_sel]
if min_rating > 0:
    filtered = filtered[filtered["google_rating"] >= min_rating]

ns_val = ns_opts[ns_sel]
NS_ORD = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
if ns_val != "any":
    thr = NS_ORD[ns_val]
    filtered = filtered[
        filtered["nutriscore"].apply(
            lambda x: NS_ORD.get(str(x).lower(), 99) <= thr if x else False
        )
    ]

if sort_sel == "Rating ↓":
    filtered = filtered.sort_values("google_rating", ascending=False, na_position="last")
elif sort_sel == "Name A–Z":
    filtered = filtered.sort_values("name")
elif sort_sel == "Price ↑":
    filtered = filtered.sort_values("price_level", key=lambda s: s.map(lambda x: PRICE_ORDER.get(x, 0)))
elif sort_sel == "Calories ↑":
    filtered = filtered.sort_values("avg_cal", na_position="last")


# ── Hero banner ───────────────────────────────────────────────
avg_r  = df["google_rating"].mean()
n_cuis = df["cuisine_key"].nunique()
st.markdown(f"""
<div class="dd-hero">
  <h1>DishDash</h1>
  <p>Hyperlocal Restaurant Discovery · NYU / Greenwich Village</p>
  <span class="dd-pill">📍 {len(df)} venues</span>
  <span class="dd-pill">⭐ {avg_r:.2f} avg rating</span>
  <span class="dd-pill">🍴 {n_cuis} cuisines</span>
  <span class="dd-pill">{'🟢 Live data' if data_src == 'live' else '🔵 Demo data'}</span>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Search",
    "🗺️ Map",
    "📊 Analytics",
    "🔮 Recom - Nat Lang",
    "🎇 Recom - N Neighbor",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Search & Filter
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="dd-section">Results ({len(filtered)} restaurants)</div>',
                unsafe_allow_html=True)

    if filtered.empty:
        st.markdown('<div class="dd-empty">No restaurants match your filters — try widening them.</div>',
                    unsafe_allow_html=True)
    else:
        for _, row in filtered.head(25).iterrows():
            st.markdown(render_card(row), unsafe_allow_html=True)
        if len(filtered) > 25:
            st.caption(f"Showing 25 of {len(filtered)} results")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Folium Map
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="dd-section">Restaurant Map</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        map_cuisine = st.selectbox("Filter cuisine", ["All cuisines"] + cuisines, key="map_c")
    with col_b:
        healthy_only = st.checkbox("Healthy only (Nutri-Score A or B)", key="map_h")

    map_df = filtered.copy()
    if map_cuisine != "All cuisines":
        map_df = map_df[map_df["cuisine_key"] == map_cuisine]
    if healthy_only:
        map_df = map_df[map_df["nutriscore"].isin(["a", "b"])]

    m = folium.Map(location=[NYU_LAT, NYU_LNG], zoom_start=15, tiles="CartoDB positron")
    folium.Marker(
        [NYU_LAT, NYU_LNG],
        tooltip="NYU / Washington Square Park",
        icon=folium.Icon(color="darkblue", icon="university", prefix="fa"),
    ).add_to(m)
    folium.Circle(
        [NYU_LAT, NYU_LNG], radius=SEARCH_RADIUS_M,
        color="#0f3460", weight=1.5, fill=True,
        fill_color="#0f3460", fill_opacity=0.04,
    ).add_to(m)

    n_added = 0
    for _, row in map_df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lng"]):
            continue
        r = row.get("google_rating")
        color = (
            "#27ae60" if pd.notna(r) and r >= 4.5 else
            "#90EE90" if pd.notna(r) and r >= 4.0 else
            "#f39c12" if pd.notna(r) and r >= 3.5 else
            "#e74c3c" if pd.notna(r) else "#888"
        )
        ns  = str(row.get("nutriscore") or "?").upper()
        cal = row.get("avg_cal")
        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"🍴 {str(row.get('cuisine_key', '')).replace('_', ' ').title()}<br>"
            f"⭐ {f'{r:.1f}' if pd.notna(r) else 'Unrated'} &nbsp; {row.get('price_level') or '—'}<br>"
            f"🥗 Nutri-Score <b>{ns}</b>"
            + (f" · {int(cal)} kcal/100g" if pd.notna(cal) else "")
        )
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=8, color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.88,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=row["name"],
        ).add_to(m)
        n_added += 1

    st.caption(
        f"Showing **{n_added}** venues · "
        "🟢 ≥4.5 · 🟡 ≥4.0 · 🟠 ≥3.5 · 🔴 <3.5 · ⚫ Unrated"
    )
    st_folium(m, width=None, height=520, returned_objects=[])


# ══════════════════════════════════════════════════════════════
# TAB 3 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="dd-section">Analytics Dashboard</div>', unsafe_allow_html=True)

    # ── Top metrics ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="dd-metric"><div class="dd-metric-val">{len(df)}</div><div class="dd-metric-lbl">Total Venues</div></div>', unsafe_allow_html=True)
    with m2:
        avg = df["google_rating"].mean()
        st.markdown(f'<div class="dd-metric"><div class="dd-metric-val">{avg:.2f}</div><div class="dd-metric-lbl">Avg Rating</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="dd-metric"><div class="dd-metric-val">{df["cuisine_key"].nunique()}</div><div class="dd-metric-lbl">Cuisines</div></div>', unsafe_allow_html=True)
    with m4:
        pct = int(df["google_rating"].notna().mean() * 100)
        st.markdown(f'<div class="dd-metric"><div class="dd-metric-val">{pct}%</div><div class="dd-metric-lbl">Have Ratings</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("DishDash Analytics — NYU Area", fontsize=15,
                 fontweight="bold", color="#1a1a2e", y=1.01)

    # 1. Rating histogram
    ax = axes[0, 0]
    rated = df["google_rating"].dropna()
    if len(rated):
        bins = [1, 2, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75, 5.01]
        cnts, edges = np.histogram(rated, bins=bins)
        clrs = ["#e74c3c" if e < 3.5 else ("#f39c12" if e < 4.0 else "#27ae60") for e in edges[:-1]]
        bars = ax.bar(range(len(cnts)), cnts, color=clrs, edgecolor="white")
        ax.set_xticks(range(len(cnts)))
        ax.set_xticklabels([f"{edges[i]:.2g}–{edges[i+1]:.2g}" for i in range(len(cnts))],
                           rotation=35, ha="right", fontsize=8)
        ax.axvline(np.searchsorted(edges[:-1], rated.mean()) - 0.5,
                   color="#0f3460", ls="--", lw=1.5, label=f"Mean {rated.mean():.2f}")
        ax.legend(fontsize=8)
        for b, c in zip(bars, cnts):
            if c:
                ax.text(b.get_x() + b.get_width() / 2, c + 0.1, str(c),
                        ha="center", va="bottom", fontsize=8)
    ax.set_title("⭐ Rating Distribution", fontweight="bold", color="#1a1a2e")
    ax.set_ylabel("Restaurants")
    ax.set_facecolor("#f8f9fa")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 2. Cuisine pie
    ax = axes[0, 1]
    cc = df["cuisine_key"].value_counts().head(12)
    if len(cc):
        lbls = [c.replace("_", " ").title() for c in cc.index]
        expl = [0.05 if i == 0 else 0 for i in range(len(cc))]
        ws, ts, ats = ax.pie(
            cc.values, labels=lbls,
            autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
            colors=PALETTE[:len(cc)], explode=expl, startangle=140,
            pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in ts:  t.set_fontsize(8)
        for at in ats: at.set_fontsize(7); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("🍴 Cuisine Breakdown", fontweight="bold", color="#1a1a2e")

    # 3. Calories by cuisine
    ax = axes[1, 0]
    cdf = (df[df["avg_cal"].notna()].groupby("cuisine_key")["avg_cal"].mean()
           .sort_values().head(12))
    if len(cdf):
        ns_map = df.groupby("cuisine_key")["nutriscore"].first()
        clrs   = [NS_COLOR.get(str(ns_map.get(c, "")).lower(), "#aaa") for c in cdf.index]
        hbs = ax.barh([c.replace("_", " ").title() for c in cdf.index],
                      cdf.values, color=clrs, edgecolor="white")
        ax.axvline(cdf.values.mean(), color="#0f3460", ls="--", lw=1.5)
        for b in hbs:
            ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                    f"{int(b.get_width())}", va="center", fontsize=8)
        patches = [mpatches.Patch(color=v, label=f"Nutri-Score {k.upper()}")
                   for k, v in NS_COLOR.items()]
        ax.legend(handles=patches, fontsize=7, loc="lower right")
    ax.set_title("🔥 Avg Calories / 100g by Cuisine", fontweight="bold", color="#1a1a2e")
    ax.set_xlabel("kcal / 100g")
    ax.set_facecolor("#f8f9fa")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 4. Sentiment or Rating vs Price
    ax = axes[1, 1]
    has_sent = df["sentiment_label"].notna().sum() > 2
    if has_sent:
        sc = df["sentiment_label"].value_counts()
        sc_c = ["#27ae60" if l == "POSITIVE" else "#e74c3c" for l in sc.index]
        bs = ax.bar(sc.index, sc.values, color=sc_c, edgecolor="white", width=0.5)
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                    str(int(b.get_height())), ha="center", fontweight="bold")
        ax.set_title(f"😊 Review Sentiment (n={sc.sum()})", fontweight="bold", color="#1a1a2e")
    else:
        sdf = df[df["google_rating"].notna() & df["price_level"].notna()].copy()
        pm  = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
        sdf = sdf[sdf["price_level"].isin(pm)]
        if len(sdf):
            sdf["pn"] = sdf["price_level"].map(pm)
            jit = np.random.uniform(-0.15, 0.15, len(sdf))
            sc2 = ["#0f3460" if r >= 4.5 else ("#f5a623" if r >= 4.0 else "#e94560")
                   for r in sdf["google_rating"]]
            ax.scatter(sdf["pn"] + jit, sdf["google_rating"],
                       c=sc2, alpha=0.72, s=55, edgecolors="white", lw=0.8)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(["$", "$$", "$$$", "$$$$"], fontsize=12)
            ax.set_ylim(2.5, 5.2)
        ax.set_title("💰 Rating vs Price Level", fontweight="bold", color="#1a1a2e")
        ax.set_xlabel("Price Level"); ax.set_ylabel("Google Rating")
    ax.set_facecolor("#f8f9fa")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Natural-Language Recommender
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="dd-section">Recommender - Natural Language</div>',
                unsafe_allow_html=True)
    st.caption("Describe what you want in plain English and get personalised picks from the DishDash database.")

    examples = [
        "— Pick an example —",
        "I want something cheap and healthy near NYU",
        "Best rated sushi or Japanese food",
        "Late-night comfort food, not worried about calories",
        "Impressive upscale dinner, budget not a concern",
        "Vegetarian-friendly with a good Nutri-Score",
        "Quick cheap lunch under $10",
    ]
    example_sel = st.selectbox("Try an example", examples)
    default_q   = "" if example_sel == examples[0] else example_sel
    user_query  = st.text_area("Your query", value=default_q,
                               placeholder='e.g. "cheap healthy food near campus"',
                               height=80)

    if st.button("Get Recommendations", type="primary"):
        q = user_query.strip().lower()
        if not q:
            st.warning("Please enter a query above.")
        else:
            sc = df.copy()
            sc["_s"] = sc["google_rating"].fillna(3.5) * 2.0

            # Price
            if any(w in q for w in ["cheap", "budget", "affordable", "inexpensive", "$10"]):
                sc["_s"] += sc["price_level"].map({"$": 6, "$$": 3, "$$$": 1, "$$$$": 0}).fillna(0)
            if any(w in q for w in ["upscale", "fancy", "fine dining", "impress", "splurge", "special"]):
                sc["_s"] += sc["price_level"].map({"$$$$": 6, "$$$": 4, "$$": 1, "$": 0}).fillna(0)

            # Health
            if any(w in q for w in ["healthy", "health", "nutri", "light", "vegetarian", "vegan", "diet"]):
                sc["_s"] += sc["nutriscore"].map({"a": 6, "b": 4, "c": 2, "d": 0, "e": 0}).fillna(0)

            # Cuisine
            for ck in CUISINE_MAP:
                if ck.replace("_", " ") in q or ck in q:
                    sc["_s"] += (sc["cuisine_key"] == ck).astype(float) * 10
            for kw, ck in [("sushi", "sushi"), ("ramen", "japanese"), ("taco", "mexican"),
                            ("pasta", "italian"), ("falafel", "mediterranean"),
                            ("pho", "vietnamese"), ("noodle", "chinese"), ("burger", "burger")]:
                if kw in q:
                    sc["_s"] += (sc["cuisine_key"] == ck).astype(float) * 8

            top3 = sc.nlargest(3, "_s")
            st.markdown(f"*Query: \"{user_query.strip()}\"*")
            st.markdown("")

            for _, row in top3.iterrows():
                r   = row.get("google_rating")
                ns  = str(row.get("nutriscore") or "?").upper()
                bg  = NS_COLOR.get(ns.lower(), "#aaa")
                cal = row.get("avg_cal")
                cuis = str(row.get("cuisine_key", "")).replace("_", " ").title()
                price = row.get("price_level") or ""
                r_str = f"{r:.1f} ⭐" if pd.notna(r) else "Unrated"
                cal_s = f"{int(cal)} kcal/100g" if pd.notna(cal) else ""

                why = []
                if pd.notna(r) and r >= 4.5:   why.append("highly rated")
                if price == "$":                why.append("budget-friendly")
                if price == "$$$$":             why.append("upscale experience")
                if ns.lower() in ("a", "b"):    why.append(f"healthy (Nutri-Score {ns})")
                if ns.lower() in ("d", "e"):    why.append("indulgent comfort food")
                why_s = ", ".join(why) if why else "good all-round pick"

                st.markdown(
                    f'<div class="dd-card" style="border-left:5px solid {bg}">'
                    f'<div class="dd-card-left">'
                    f'<div class="dd-card-name">{row["name"]}</div>'
                    f'<span class="dd-cuisine-tag">{cuis}</span>'
                    f'<div class="dd-detail" style="margin-top:7px">'
                    f'{r_str} &nbsp; {price} &nbsp; '
                    f'<span class="dd-ns-badge" style="background:{bg}">{ns}</span>{cal_s}'
                    f'</div>'
                    f'<div class="dd-detail" style="font-style:italic;color:#555">Why: {why_s}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════
# TAB 5 — Nearest-Neighbour Recommender
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="dd-section">Recommender - Nearest Neighbor</div>',
                unsafe_allow_html=True)

    st.markdown("""<div style="font-family:monospace;color:#000000;white-space:pre;background:#f0f2f6;padding:16px;border-radius:8px;font-size:13px;line-height:1.6;">Methodology
-----------
Each venue is represented as a point in a 6-dimensional feature space. 
Continuous features (rating, price, nutri-score) are min-max normalised to
[0, 1] using fixed semantic scales so dimensions are comparable regardless
of their raw ranges.  Geo distance is the haversine miles from the query
restaurant to each candidate, also normalised to [0, 1] relative to the
dataset spread.  Cuisine is a binary 0/1 flag (same / different).
Sentiment is mapped from (label, confidence) to a [0, 1] scalar.
The best match is the venue minimising the weighted Euclidean distance
The query restaurant itself is excluded from the result.</div>""", unsafe_allow_html=True)

    st.markdown("---")

    restaurant_names = sorted(df["name"].dropna().unique().tolist())
    selected = st.selectbox("Choose a restaurant", ["— select —"] + restaurant_names)

    if selected != "— select —":
        alt = get_alternative(selected)

        if alt is None:
            st.warning(f"Could not find a match for '{selected}'.")
        else:
            st.markdown(f"**Nearest neighbour for:** {selected}")
            st.markdown("")

            ns      = str(alt.get("nutriscore") or "?").upper()
            ns_bg   = NS_COLOR.get(ns.lower(), "#aaa")
            r       = alt.get("rating")
            r_str   = f"{r:.1f}" if r is not None and str(r) != "nan" else "—"
            stars   = ("★" * int(round(r)) + "☆" * (5 - int(round(r)))) if r and str(r) != "nan" else ""
            price   = alt.get("price", "")
            cuisine = str(alt.get("cuisine", "")).replace("_", " ").title()
            sent    = alt.get("sentiment", "")
            addr    = alt.get("address", "")
            score   = alt.get("distance_score")

            sent_html = ""
            if sent:
                cls  = "pos" if sent == "POSITIVE" else "neg"
                icon = "😊" if sent == "POSITIVE" else "😞"
                sent_html = f'<span class="dd-sentiment-{cls}">{icon} {sent.title()}</span>'

            st.markdown(
                f'<div class="dd-card" style="border-left:5px solid {ns_bg}">'
                f'<div class="dd-card-left">'
                f'<div class="dd-card-name">{alt["name"]}</div>'
                f'<span class="dd-cuisine-tag">{cuisine}</span>'
                f'<div class="dd-detail" style="margin-top:7px">'
                f'<span class="dd-ns-badge" style="background:{ns_bg}">{ns}</span>'
                f'</div>'
                f'<div class="dd-detail">{sent_html}</div>'
                f'<div class="dd-detail">{addr}</div>'
                f'<div class="dd-detail" style="margin-top:8px;font-style:italic;color:#555">'
                f'Similarity distance: {score} (lower = more similar)'
                f'</div>'
                f'</div>'
                f'<div class="dd-card-right">'
                f'<div class="dd-rating">{r_str}</div>'
                f'<div class="dd-stars">{stars}</div>'
                f'<div class="dd-price">{price}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
