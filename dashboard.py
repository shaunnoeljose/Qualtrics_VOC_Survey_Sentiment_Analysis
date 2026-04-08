"""
dashboard.py
------------
Streamlit VOC Health Monitoring Dashboard.

Simulates an enterprise-grade Voice of Customer (VOC) analytics platform
with real-time OKR health monitoring, sentiment breakdowns, theme analysis,
and executive-level reporting — aligned to the JD tools: Qualtrics, VOC,
OKR dashboards, health monitoring, Customer Journey Analytics.

Run:
  streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sentiment_analysis import run_sentiment_pipeline


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VOC Health Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Colour palette ────────────────────────────────────────────────────────────

COLORS = {
    "Positive": "#1D9E75",
    "Neutral":  "#888780",
    "Negative": "#D85A30",
    "primary":  "#185FA5",
    "light_bg": "#F1EFE8",
}

# ── Data loader ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Run pipeline once and cache results for the session."""
    df     = run_sentiment_pipeline()
    report = json.loads(Path("data/voc_insight_report.json").read_text())
    return df, report


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("VOC Analytics")
    st.caption("Qualtrics-powered sentiment & OKR monitoring")
    st.divider()

    st.subheader("Filters")
    channel_filter = st.multiselect(
        "Channel",
        options=["web", "mobile", "airport"],
        default=["web", "mobile", "airport"],
    )
    surface_filter = st.multiselect(
        "Product Surface",
        options=["booking", "check_in", "flight", "baggage", "loyalty", "support"],
        default=["booking", "check_in", "flight", "baggage", "loyalty", "support"],
    )
    respondent_filter = st.multiselect(
        "Respondent Type",
        options=["frequent_flyer", "occasional", "new"],
        default=["frequent_flyer", "occasional", "new"],
    )
    st.divider()
    st.caption("Data source: Qualtrics API (OAuth 2.0)")
    st.caption("NLP Engine: VADER + TF-IDF")


# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading VOC data from Qualtrics pipeline…"):
    df_full, report = load_data()

df = df_full[
    df_full["channel"].isin(channel_filter) &
    df_full["product_surface"].isin(surface_filter) &
    df_full["respondent_type"].isin(respondent_filter)
].copy()

stats = report["overall_stats"]


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 VOC Health Monitoring Dashboard")
st.caption("Voice of Customer · Customer Journey Analytics · OKR Monitoring")
st.divider()


# ── OKR Alert banner ──────────────────────────────────────────────────────────

alerts = report.get("okr_alerts", [])
if alerts:
    for alert in alerts:
        st.error(f"⚠ OKR BREACH — {alert['message']}", icon="🚨")
else:
    st.success("✅ All OKR thresholds are healthy.", icon="✅")

st.divider()


# ── KPI metric cards ──────────────────────────────────────────────────────────

st.subheader("Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

total        = len(df)
nps_scores   = df["nps_score"].astype(int)
promoters    = (nps_scores >= 9).sum()
detractors   = (nps_scores <= 6).sum()
nps          = round(((promoters - detractors) / total) * 100, 1) if total else 0
avg_csat     = round(df["csat_score"].astype(float).mean(), 2) if total else 0
pos_pct      = round((df["sentiment"] == "Positive").sum() / total * 100, 1) if total else 0
neg_pct      = round((df["sentiment"] == "Negative").sum() / total * 100, 1) if total else 0

col1.metric("Total Responses",    total)
col2.metric("NPS Score",          nps,      delta=f"Target: 30", delta_color="normal")
col3.metric("Avg CSAT",           avg_csat, delta=f"Target: 3.8", delta_color="normal")
col4.metric("Positive Sentiment", f"{pos_pct}%")
col5.metric("Negative Sentiment", f"{neg_pct}%", delta_color="inverse")

st.divider()


# ── Row 1: Sentiment distribution + NPS breakdown ─────────────────────────────

st.subheader("Sentiment Overview")
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown("**Sentiment distribution**")
    sent_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(
        sent_counts.index,
        sent_counts.values,
        color=[COLORS.get(s, "#888780") for s in sent_counts.index],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_ylabel("Responses", fontsize=10)
    ax.set_title("Responses by Sentiment", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with r1c2:
    st.markdown("**NPS promoter / passive / detractor split**")
    nps_labels  = ["Promoters\n(9-10)", "Passives\n(7-8)", "Detractors\n(0-6)"]
    nps_vals    = [
        (nps_scores >= 9).sum(),
        ((nps_scores >= 7) & (nps_scores <= 8)).sum(),
        (nps_scores <= 6).sum(),
    ]
    nps_colors  = [COLORS["Positive"], COLORS["Neutral"], COLORS["Negative"]]
    fig2, ax2   = plt.subplots(figsize=(5, 3.5))
    wedges, texts, autotexts = ax2.pie(
        nps_vals,
        labels=nps_labels,
        colors=nps_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title(f"NPS = {nps}", fontsize=11, pad=8)
    fig2.patch.set_alpha(0)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

st.divider()


# ── Row 2: Sentiment by channel + surface heatmap ─────────────────────────────

st.subheader("Customer Journey Breakdown")
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown("**Sentiment by channel**")
    channel_df = (
        df.groupby(["channel", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    channel_df.plot(
        kind="bar",
        ax=ax3,
        color=[COLORS["Positive"], COLORS["Neutral"], COLORS["Negative"]],
        edgecolor="white",
        linewidth=0.5,
        rot=0,
    )
    ax3.set_ylabel("Responses", fontsize=10)
    ax3.set_title("Sentiment by Channel", fontsize=11, pad=8)
    ax3.legend(title="Sentiment", fontsize=9, title_fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor("none")
    fig3.patch.set_alpha(0)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

with r2c2:
    st.markdown("**Avg CSAT by product surface (heatmap)**")
    surface_csat = (
        df.groupby("product_surface")["csat_score"]
        .mean()
        .round(2)
        .sort_values()
        .reset_index()
    )
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    pivot = surface_csat.set_index("product_surface")[["csat_score"]].T
    sns.heatmap(
        pivot,
        ax=ax4,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax4.set_title("Avg CSAT by Product Surface", fontsize=11, pad=8)
    ax4.set_ylabel("")
    ax4.set_xlabel("")
    fig4.patch.set_alpha(0)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

st.divider()


# ── Row 3: VADER score distribution + respondent type breakdown ────────────────

st.subheader("Sentiment Depth Analysis")
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.markdown("**VADER compound score distribution**")
    fig5, ax5 = plt.subplots(figsize=(5, 3.5))
    for sentiment, grp in df.groupby("sentiment"):
        ax5.hist(
            grp["vader_compound"],
            bins=15,
            alpha=0.65,
            color=COLORS.get(sentiment, "#888780"),
            label=sentiment,
            edgecolor="white",
            linewidth=0.3,
        )
    ax5.axvline(0.05,  color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax5.axvline(-0.05, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
    ax5.set_xlabel("VADER Compound Score", fontsize=10)
    ax5.set_ylabel("Count", fontsize=10)
    ax5.set_title("Score Distribution by Sentiment", fontsize=11, pad=8)
    ax5.legend(fontsize=9)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.set_facecolor("none")
    fig5.patch.set_alpha(0)
    st.pyplot(fig5, use_container_width=True)
    plt.close()

with r3c2:
    st.markdown("**Sentiment by respondent type**")
    resp_df = (
        df.groupby(["respondent_type", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    fig6, ax6 = plt.subplots(figsize=(5, 3.5))
    resp_df.plot(
        kind="barh",
        ax=ax6,
        stacked=True,
        color=[COLORS["Positive"], COLORS["Neutral"], COLORS["Negative"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax6.set_xlabel("Responses", fontsize=10)
    ax6.set_title("Sentiment by Respondent Type", fontsize=11, pad=8)
    ax6.legend(title="Sentiment", fontsize=9, title_fontsize=9, loc="lower right")
    ax6.spines[["top", "right"]].set_visible(False)
    ax6.set_facecolor("none")
    fig6.patch.set_alpha(0)
    st.pyplot(fig6, use_container_width=True)
    plt.close()

st.divider()


# ── Row 4: Top themes ─────────────────────────────────────────────────────────

st.subheader("Top VOC Themes (TF-IDF)")
themes = report.get("top_themes", {})
tc1, tc2, tc3 = st.columns(3)

for col, label, color in [
    (tc1, "Positive", COLORS["Positive"]),
    (tc2, "Neutral",  COLORS["Neutral"]),
    (tc3, "Negative", COLORS["Negative"]),
]:
    with col:
        st.markdown(f"**{label} themes**")
        kws = themes.get(label, [])
        for kw in kws:
            st.markdown(
                f'<span style="background:{color}22;color:{color};'
                f'padding:2px 10px;border-radius:12px;font-size:13px;">'
                f'{kw}</span>',
                unsafe_allow_html=True,
            )

st.divider()


# ── Row 5: Raw response explorer ──────────────────────────────────────────────

st.subheader("Response Explorer")
sentiment_filter_table = st.selectbox(
    "Filter by sentiment",
    options=["All", "Positive", "Neutral", "Negative"],
)

display_df = df if sentiment_filter_table == "All" else df[df["sentiment"] == sentiment_filter_table]

st.dataframe(
    display_df[["response_id", "timestamp", "channel", "product_surface",
                "nps_score", "csat_score", "sentiment", "vader_compound", "open_text"]]
    .sort_values("vader_compound")
    .reset_index(drop=True),
    use_container_width=True,
    height=300,
)

st.divider()
st.caption("Built with Python · NLTK VADER · TF-IDF · Streamlit  |  Data source: Qualtrics API (OAuth 2.0 simulated)")
