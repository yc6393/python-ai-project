import streamlit as st
import pandas as pd
import sqlite3
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="DishDash", layout="wide")
PRICE_RANK = {"Free": 0, "$": 1, "$$": 2, "$$$": 3, "$$$$": 4}

@st.cache_data
def load_data(db_path="dishdash.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM venues", conn)
    conn.close()
    return df

df = load_data()
st.title("DishDash")
st.caption(f"{len(df)} venues across {df['neighborhood'].nunique()} neighborhoods")

with st.sidebar:
    st.header("Filters")
    name_q     = st.text_input("Search by name")
    cuisines   = st.multiselect("Cuisine", sorted(df["cuisine_key"].dropna().unique()))
    nbhds      = st.multiselect("Neighborhood", sorted(df["neighborhood"].dropna().unique()))
    max_price  = st.selectbox("Max price tier", ["Any", "$", "$$", "$$$", "$$$$"])
    min_rating = st.slider("Min rating", 0.0, 5.0, 0.0, 0.1)
    max_cal    = st.slider("Max kcal/100g", 100, 600, 600, 10)
    sentiment  = st.selectbox("Sentiment", ["Any", "POSITIVE", "NEGATIVE"])
    has_rev    = st.checkbox("Has reviews only")

res = df.copy()
if name_q:    res = res[res["name"].str.contains(name_q, case=False, na=False)]
if cuisines:  res = res[res["cuisine_key"].isin(cuisines)]
if nbhds:     res = res[res["neighborhood"].isin(nbhds)]
if max_price != "Any":
    cap = PRICE_RANK[max_price]
    res = res[res["price_level"].map(lambda p: PRICE_RANK.get(p, 99) <= cap)]
if min_rating > 0: res = res[res["google_rating"].fillna(-1) >= min_rating]
if max_cal < 600:  res = res[res["avg_calories_100g"].fillna(1e9) <= max_cal]
if sentiment != "Any": res = res[res["sentiment_label"] == sentiment]
if has_rev:    res = res[res["review_text"].fillna("").str.strip().astype(bool)]

left, right = st.columns([2, 1])

with left:
    st.subheader(f"Map - {len(res)} matches")
    if len(res) > 0:
        m = folium.Map(location=[res["lat"].mean(), res["lng"].mean()],
                       zoom_start=13, tiles="OpenStreetMap")
        for _, row in res.iterrows():
            popup = f"<b>{row['name']}</b><br>{row.get('cuisine_key', '')}"
            folium.CircleMarker([row["lat"], row["lng"]], radius=6,
                color="#c0392b", fill=True, fill_opacity=0.85,
                popup=popup).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.info("No matches. Try loosening the filters.")

with right:
    st.subheader("Detail")
    if len(res) > 0:
        names = res["name"].tolist()
        pick = st.selectbox("Pick a venue", names)
        row = res[res["name"] == pick].iloc[0]
        st.markdown(f"### {row['name']}")
        st.caption(f"{row.get('cuisine_key', '-')} - {row.get('neighborhood', '-')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rating",
                  f"{row['google_rating']:.1f}" if pd.notna(row.get('google_rating')) else "-")
        c2.metric("Price",     row.get("price_level") or "-")
        c3.metric("Sentiment", row.get("sentiment_label") or "-")
        st.markdown("**Nutrition (cuisine avg, per 100 g)**")
        nut = pd.DataFrame({
            "metric": ["Calories", "Fat", "Protein", "Carbs", "Sugar", "Sodium"],
            "value": [
                f"{row['avg_calories_100g']:.0f} kcal" if pd.notna(row.get('avg_calories_100g')) else "-",
                f"{row['avg_fat_100g']:.1f} g"        if pd.notna(row.get('avg_fat_100g'))      else "-",
                f"{row['avg_protein_100g']:.1f} g"    if pd.notna(row.get('avg_protein_100g'))  else "-",
                f"{row['avg_carbs_100g']:.1f} g"      if pd.notna(row.get('avg_carbs_100g'))    else "-",
                f"{row['avg_sugar_100g']:.1f} g"      if pd.notna(row.get('avg_sugar_100g'))    else "-",
                f"{row['avg_sodium_100g']:.3f} g"     if pd.notna(row.get('avg_sodium_100g'))   else "-",
            ]})
        st.dataframe(nut, hide_index=True)
        if row.get("review_text"):
            st.markdown("**Sample reviews**")
            st.write(str(row["review_text"])[:500])
