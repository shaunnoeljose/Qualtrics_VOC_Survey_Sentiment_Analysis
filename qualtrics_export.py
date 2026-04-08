"""
qualtrics_export.py
-------------------
Simulates a production-grade Qualtrics API integration using OAuth 2.0.
Automates survey response exports with robust error handling, retry logic,
polling mechanism, and CloudWatch-style logging.

In a live environment, replace MockQualtricsAPI with real Qualtrics endpoints:
  - Token:  POST https://{data_center}.qualtrics.com/oauth2/token
  - Export: POST https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses
  - Status: GET  https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses/{progress_id}
  - File:   GET  https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses/{file_id}/file
"""

import os
import csv
import time
import logging
import json
import random
import string
from datetime import datetime
from pathlib import Path


# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qualtrics_workflow.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────

QUALTRICS_CLIENT_ID     = os.getenv("QUALTRICS_CLIENT_ID",     "mock_client_id")
QUALTRICS_CLIENT_SECRET = os.getenv("QUALTRICS_CLIENT_SECRET", "mock_client_secret")
QUALTRICS_DATA_CENTER   = os.getenv("QUALTRICS_DATA_CENTER",   "sjc1")
SURVEY_ID               = os.getenv("QUALTRICS_SURVEY_ID",     "SV_mock_survey_001")

POLL_INTERVAL_SECONDS = 2
MAX_RETRIES           = 3
OUTPUT_DIR            = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Mock Qualtrics API (replace with real HTTP calls in production) ───────────

class MockQualtricsAPI:
    """
    Mirrors the real Qualtrics REST API flow:
      1. POST /oauth2/token          → access_token
      2. POST /export-responses      → progress_id
      3. GET  /export-responses/{id} → status + file_id when complete
      4. GET  /file                  → CSV bytes
    """

    def __init__(self, client_id: str, client_secret: str, data_center: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.data_center   = data_center
        self._token        = None
        self._token_expiry = 0

    # ── OAuth 2.0 ─────────────────────────────────────────────────────────

    def get_access_token(self) -> str:
        """Fetch (or return cached) OAuth 2.0 bearer token."""
        now = time.time()
        if self._token and now < self._token_expiry:
            logger.info("Reusing cached OAuth token.")
            return self._token

        logger.info("Requesting new OAuth 2.0 token from Qualtrics...")
        time.sleep(0.3)  # simulate network latency

        # In production:
        # response = requests.post(
        #     f"https://{self.data_center}.qualtrics.com/oauth2/token",
        #     data={"grant_type": "client_credentials"},
        #     auth=(self.client_id, self.client_secret),
        # )
        # token_data = response.json()

        token_data = {
            "access_token": "mock_" + "".join(random.choices(string.ascii_lowercase, k=32)),
            "expires_in": 3600,
        }

        self._token        = token_data["access_token"]
        self._token_expiry = now + token_data["expires_in"] - 60  # 60 s safety buffer
        logger.info("OAuth token acquired. Expires in %d seconds.", token_data["expires_in"])
        return self._token

    # ── Export initiation ────────────────────────────────────────────────

    def start_export(self, survey_id: str, token: str) -> str:
        """Kick off an async export job; returns a progress_id."""
        logger.info("Initiating export for survey: %s", survey_id)
        time.sleep(0.4)

        # In production:
        # response = requests.post(
        #     f"https://{self.data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses",
        #     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        #     json={"format": "csv"},
        # )
        # return response.json()["result"]["progressId"]

        progress_id = "ES_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
        logger.info("Export job started. Progress ID: %s", progress_id)
        return progress_id

    # ── Polling ───────────────────────────────────────────────────────────

    def poll_export_status(self, survey_id: str, progress_id: str, token: str) -> str:
        """
        Poll until status is 'complete'; returns file_id.
        Raises RuntimeError if export fails.
        """
        logger.info("Polling export status for progress ID: %s", progress_id)
        statuses = ["inProgress", "inProgress", "complete"]

        for attempt, status in enumerate(statuses, 1):
            time.sleep(POLL_INTERVAL_SECONDS)
            logger.info("Poll attempt %d — status: %s", attempt, status)

            if status == "complete":
                file_id = "EF_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
                logger.info("Export complete. File ID: %s", file_id)
                return file_id

            if status == "failed":
                raise RuntimeError("Qualtrics export job failed. Check survey permissions.")

        raise RuntimeError("Export polling timed out after maximum attempts.")

    # ── Download ──────────────────────────────────────────────────────────

    def download_file(self, survey_id: str, file_id: str, token: str) -> list[dict]:
        """Download completed export; returns list of response dicts."""
        logger.info("Downloading export file: %s", file_id)
        time.sleep(0.3)

        # In production:
        # response = requests.get(
        #     f"https://{self.data_center}.qualtrics.com/API/v3/surveys/{survey_id}"
        #     f"/export-responses/{file_id}/file",
        #     headers={"Authorization": f"Bearer {token}"},
        # )
        # with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        #     with z.open(z.namelist()[0]) as f:
        #         reader = csv.DictReader(io.TextIOWrapper(f))
        #         return list(reader)

        sample_path = OUTPUT_DIR / "sample_survey_responses.csv"
        with open(sample_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        logger.info("Downloaded %d survey responses.", len(rows))
        return rows


# ── Core export workflow ──────────────────────────────────────────────────────

def export_survey_responses(survey_id: str = SURVEY_ID) -> list[dict]:
    """
    Full Qualtrics export workflow:
      authenticate → start export → poll → download → save locally.

    Returns list of response dicts for downstream processing.
    """
    api = MockQualtricsAPI(QUALTRICS_CLIENT_ID, QUALTRICS_CLIENT_SECRET, QUALTRICS_DATA_CENTER)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("=== Export attempt %d of %d ===", attempt, MAX_RETRIES)

            token       = api.get_access_token()
            progress_id = api.start_export(survey_id, token)
            file_id     = api.poll_export_status(survey_id, progress_id, token)
            responses   = api.download_file(survey_id, file_id, token)

            # Persist locally
            out_path = OUTPUT_DIR / f"responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=responses[0].keys())
                writer.writeheader()
                writer.writerows(responses)

            logger.info("Responses saved to: %s", out_path)
            return responses

        except RuntimeError as exc:
            logger.error("Attempt %d failed: %s", attempt, exc)
            if attempt == MAX_RETRIES:
                logger.critical("All %d attempts exhausted. Aborting export.", MAX_RETRIES)
                raise
            wait = 2 ** attempt
            logger.info("Retrying in %d seconds (exponential backoff)…", wait)
            time.sleep(wait)

    return []


# ── OKR threshold alert ───────────────────────────────────────────────────────

def check_okr_thresholds(responses: list[dict]) -> dict:
    """
    Evaluate key OKR thresholds and flag breaches.
    Thresholds mirror real-world digital analytics OKR targets.
    """
    nps_scores  = [int(r["nps_score"])  for r in responses]
    csat_scores = [float(r["csat_score"]) for r in responses]

    promoters  = sum(1 for s in nps_scores if s >= 9)
    detractors = sum(1 for s in nps_scores if s <= 6)
    nps        = round(((promoters - detractors) / len(nps_scores)) * 100, 1)
    avg_csat   = round(sum(csat_scores) / len(csat_scores), 2)

    thresholds = {"nps_target": 30, "csat_target": 3.8}
    alerts     = []

    if nps < thresholds["nps_target"]:
        alerts.append(f"NPS ALERT: {nps} is below target of {thresholds['nps_target']}")
    if avg_csat < thresholds["csat_target"]:
        alerts.append(f"CSAT ALERT: {avg_csat} is below target of {thresholds['csat_target']}")

    result = {
        "nps": nps,
        "avg_csat": avg_csat,
        "total_responses": len(responses),
        "promoters": promoters,
        "detractors": detractors,
        "alerts": alerts,
        "status": "BREACH" if alerts else "HEALTHY",
    }

    if alerts:
        for alert in alerts:
            logger.warning(alert)
    else:
        logger.info("All OKR thresholds met. Status: HEALTHY")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Qualtrics VOC export pipeline...")

    responses = export_survey_responses()
    okr_status = check_okr_thresholds(responses)

    print("\n" + "=" * 50)
    print("  OKR HEALTH SUMMARY")
    print("=" * 50)
    print(f"  Total Responses : {okr_status['total_responses']}")
    print(f"  NPS Score       : {okr_status['nps']}")
    print(f"  Avg CSAT        : {okr_status['avg_csat']}")
    print(f"  Status          : {okr_status['status']}")
    if okr_status["alerts"]:
        print("\n  ALERTS:")
        for a in okr_status["alerts"]:
            print(f"  ⚠  {a}")
    print("=" * 50)
