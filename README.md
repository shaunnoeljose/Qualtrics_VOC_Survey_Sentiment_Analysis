# Qualtrics VOC Sentiment Pipeline

An end-to-end **Voice of Customer (VOC)** analytics pipeline that simulates a production-grade Qualtrics API integration, enriched with NLP sentiment analysis, TF-IDF theme extraction, OKR health monitoring, and an interactive Streamlit dashboard.

Built to demonstrate enterprise digital analytics workflows aligned with tools like **Qualtrics**, **Adobe Analytics**, **Azure Databricks**, and **VOC / Customer Journey Analytics** programs.

---

## Features

- **Qualtrics API simulation** — OAuth 2.0 token flow, async export job, polling mechanism, retry logic with exponential backoff, and CloudWatch-style logging
- **VADER sentiment scoring** — classifies open-text survey responses into Positive / Neutral / Negative with compound scores
- **TF-IDF theme extraction** — surfaces top keywords per sentiment bucket using unigrams and bigrams
- **OKR health monitoring** — threshold alerts for NPS, CSAT, and negative sentiment % breach
- **Customer Journey breakdown** — sentiment and CSAT drill-downs by channel (web / mobile / airport) and product surface (booking / check-in / flight / loyalty)
- **Streamlit dashboard** — interactive VOC monitoring UI with filters, KPI cards, charts, and response explorer
- **Jupyter notebook** — step-by-step walkthrough of the full pipeline

---

## Project Structure

```
qualtrics-voc-sentiment-pipeline/
├── qualtrics_export.py          # Qualtrics API OAuth 2.0 export simulation
├── sentiment_analysis.py        # VADER + TF-IDF NLP pipeline
├── dashboard.py                 # Streamlit VOC health monitoring dashboard
├── requirements.txt
├── data/
│   └── sample_survey_responses.csv   # 50 simulated airline passenger VOC responses
└── notebooks/
    └── voc_analysis.ipynb       # End-to-end analysis walkthrough
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/shaunnoeljose/qualtrics-voc-sentiment-pipeline.git
cd qualtrics-voc-sentiment-pipeline
pip install -r requirements.txt

# 2. Run the export pipeline
python qualtrics_export.py

# 3. Run sentiment analysis
python sentiment_analysis.py

# 4. Launch the Streamlit dashboard
streamlit run dashboard.py
```

---

## Dashboard Preview

The Streamlit dashboard includes:

| Section | Description |
|---|---|
| OKR Alert Banner | Real-time breach flags for NPS, CSAT, and negative sentiment % |
| KPI Cards | Total responses, NPS, Avg CSAT, Positive %, Negative % |
| Sentiment Distribution | Bar chart of Positive / Neutral / Negative counts |
| NPS Pie Chart | Promoter / Passive / Detractor breakdown |
| Sentiment by Channel | Grouped bar chart across web, mobile, airport |
| CSAT Heatmap | Avg CSAT by product surface |
| VADER Distribution | Compound score histogram per sentiment class |
| VOC Themes | TF-IDF top keywords per sentiment bucket |
| Response Explorer | Filterable raw response table with sentiment labels |

---

## OKR Thresholds

| Metric | Target | Alert Trigger |
|---|---|---|
| NPS | ≥ 30 | Below 30 |
| Avg CSAT | ≥ 3.8 | Below 3.8 |
| Negative Sentiment % | ≤ 25% | Above 25% |

---

## Tech Stack

| Layer | Tools |
|---|---|
| API Integration | Python `requests`, OAuth 2.0, exponential backoff |
| NLP | NLTK VADER, TF-IDF (scikit-learn), custom preprocessing |
| Data Processing | Pandas, NumPy |
| Visualisation | Streamlit, Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## In Production

To connect to a live Qualtrics account, replace the `MockQualtricsAPI` class in `qualtrics_export.py` with real HTTP calls:

```python
# Token
requests.post(f"https://{data_center}.qualtrics.com/oauth2/token", ...)

# Start export
requests.post(f"https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses", ...)

# Poll status
requests.get(f"https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses/{progress_id}", ...)

# Download file
requests.get(f"https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses/{file_id}/file", ...)
```

Set environment variables:
```bash
export QUALTRICS_CLIENT_ID="your_client_id"
export QUALTRICS_CLIENT_SECRET="your_client_secret"
export QUALTRICS_DATA_CENTER="sjc1"
export QUALTRICS_SURVEY_ID="SV_xxxxxxxxxxxxxxx"
```
