"""
sentiment_analysis.py
---------------------
NLP sentiment analysis engine for Qualtrics VOC survey responses.

Pipeline:
  1. Load exported survey responses
  2. Preprocess open-text feedback (tokenise, clean, remove stopwords)
  3. Score sentiment using VADER (rule-based, domain-agnostic)
  4. Classify into Positive / Neutral / Negative
  5. Extract top themes per sentiment bucket using TF-IDF
  6. Generate a structured VOC insight report saved as JSON

Dependencies:
  pip install vaderSentiment pandas scikit-learn
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Simple English stopwords (no NLTK download needed)
_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very","s",
    "t","can","will","just","don","should","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Text preprocessing ────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Lowercase → strip punctuation/numbers → tokenise → remove stopwords → rejoin.
    Returns cleaned string suitable for TF-IDF and VADER.
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def classify_sentiment(compound: float) -> str:
    """
    VADER compound score thresholds (standard industry convention):
      >= 0.05  → Positive
      <= -0.05 → Negative
      else     → Neutral
    """
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER compound score and sentiment label columns to the DataFrame."""
    sia = SentimentIntensityAnalyzer()

    df = df.copy()
    df["cleaned_text"] = df["open_text"].fillna("").apply(preprocess_text)

    scores = df["open_text"].fillna("").apply(lambda t: sia.polarity_scores(t))
    df["vader_compound"] = scores.apply(lambda s: round(s["compound"], 4))
    df["vader_pos"]      = scores.apply(lambda s: round(s["pos"], 4))
    df["vader_neu"]      = scores.apply(lambda s: round(s["neu"], 4))
    df["vader_neg"]      = scores.apply(lambda s: round(s["neg"], 4))
    df["sentiment"]      = df["vader_compound"].apply(classify_sentiment)

    logger.info(
        "Sentiment distribution — %s",
        df["sentiment"].value_counts().to_dict(),
    )
    return df


# ── Theme extraction ──────────────────────────────────────────────────────────

def extract_themes(texts: list, top_n: int = 5) -> list:
    """
    Use TF-IDF to surface the most discriminative terms in a corpus.
    Returns the top_n keyword strings.
    """
    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    mean_scores = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_scores.argsort()[::-1][:top_n]
    terms       = vectorizer.get_feature_names_out()
    return [terms[i] for i in top_indices]


# ── VOC insight report ────────────────────────────────────────────────────────

def generate_voc_report(df: pd.DataFrame) -> dict:
    """
    Aggregate sentiment results into an executive VOC insight report.
    """
    total = len(df)

    nps_scores = df["nps_score"].astype(int)
    promoters  = (nps_scores >= 9).sum()
    passives   = ((nps_scores >= 7) & (nps_scores <= 8)).sum()
    detractors = (nps_scores <= 6).sum()
    nps        = round(((promoters - detractors) / total) * 100, 1)
    avg_csat   = round(df["csat_score"].astype(float).mean(), 2)

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    sentiment_pct = {
        k: round((v / total) * 100, 1)
        for k, v in sentiment_counts.items()
    }

    channel_sentiment = (
        df.groupby(["channel", "sentiment"])
        .size().unstack(fill_value=0)
        .to_dict(orient="index")
    )
    surface_sentiment = (
        df.groupby(["product_surface", "sentiment"])
        .size().unstack(fill_value=0)
        .to_dict(orient="index")
    )

    themes = {}
    for label in ["Positive", "Neutral", "Negative"]:
        bucket_texts = df.loc[df["sentiment"] == label, "cleaned_text"].tolist()
        themes[label] = extract_themes(bucket_texts, top_n=5)

    okr_thresholds = {"nps_target": 30, "csat_target": 3.8, "negative_pct_max": 25.0}
    alerts = []
    negative_pct = sentiment_pct.get("Negative", 0)

    if nps < okr_thresholds["nps_target"]:
        alerts.append({
            "metric": "NPS", "value": nps,
            "threshold": okr_thresholds["nps_target"],
            "message": f"NPS {nps} is below target of {okr_thresholds['nps_target']}",
        })
    if avg_csat < okr_thresholds["csat_target"]:
        alerts.append({
            "metric": "CSAT", "value": avg_csat,
            "threshold": okr_thresholds["csat_target"],
            "message": f"Avg CSAT {avg_csat} is below target of {okr_thresholds['csat_target']}",
        })
    if negative_pct > okr_thresholds["negative_pct_max"]:
        alerts.append({
            "metric": "Negative Sentiment %", "value": negative_pct,
            "threshold": okr_thresholds["negative_pct_max"],
            "message": f"Negative sentiment {negative_pct}% exceeds max of {okr_thresholds['negative_pct_max']}%",
        })

    low_csat = (
        df[df["csat_score"].astype(float) <= 2][
            ["response_id", "open_text", "sentiment", "vader_compound"]
        ].head(5).to_dict(orient="records")
    )

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "overall_stats": {
            "total_responses": total,
            "nps_score": nps,
            "avg_csat": avg_csat,
            "promoters": int(promoters),
            "passives": int(passives),
            "detractors": int(detractors),
            "sentiment_counts": sentiment_counts,
            "sentiment_pct": sentiment_pct,
        },
        "sentiment_by_channel": channel_sentiment,
        "sentiment_by_surface": surface_sentiment,
        "top_themes": themes,
        "okr_alerts": alerts,
        "okr_status": "BREACH" if alerts else "HEALTHY",
        "low_csat_samples": low_csat,
    }

    report_path = OUTPUT_DIR / "voc_insight_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("VOC insight report saved to: %s", report_path)

    return report


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_sentiment_pipeline(csv_path=None) -> pd.DataFrame:
    """End-to-end pipeline: load → score → report → return enriched DataFrame."""
    if csv_path is None:
        files = sorted(DATA_DIR.glob("responses_*.csv"))
        csv_path = str(files[-1]) if files else str(DATA_DIR / "sample_survey_responses.csv")

    logger.info("Loading survey responses from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d responses.", len(df))

    df = score_responses(df)

    enriched_path = OUTPUT_DIR / "enriched_responses.csv"
    df.to_csv(enriched_path, index=False)
    logger.info("Enriched responses saved to: %s", enriched_path)

    report = generate_voc_report(df)

    print("\n" + "=" * 55)
    print("  VOC SENTIMENT ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Total Responses    : {report['overall_stats']['total_responses']}")
    print(f"  NPS Score          : {report['overall_stats']['nps_score']}")
    print(f"  Avg CSAT           : {report['overall_stats']['avg_csat']}")
    print(f"  Positive Sentiment : {report['overall_stats']['sentiment_pct'].get('Positive', 0)}%")
    print(f"  Neutral  Sentiment : {report['overall_stats']['sentiment_pct'].get('Neutral', 0)}%")
    print(f"  Negative Sentiment : {report['overall_stats']['sentiment_pct'].get('Negative', 0)}%")
    print(f"  OKR Status         : {report['okr_status']}")
    if report["okr_alerts"]:
        print("\n  ALERTS:")
        for alert in report["okr_alerts"]:
            print(f"  ⚠  {alert['message']}")
    print("\n  TOP NEGATIVE THEMES:", ", ".join(report["top_themes"].get("Negative", [])))
    print("  TOP POSITIVE THEMES:", ", ".join(report["top_themes"].get("Positive", [])))
    print("=" * 55)

    return df


if __name__ == "__main__":
    run_sentiment_pipeline()
