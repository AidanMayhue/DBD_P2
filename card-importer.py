"""
load_standard_cards.py

Fetches all Standard-legal MTG cards from the Scryfall bulk data API
and upserts them into a MongoDB Atlas collection.

Requirements:
    pip install pymongo requests python-dotenv

Environment variables (create a .env file or set directly):
    MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
    MONGO_DB=mtg
    MONGO_COLLECTION=cards
    LOG_LEVEL=INFO        (optional: DEBUG | INFO | WARNING | ERROR)
    LOG_FILE=pipeline.log (optional: path to log file, logs to stdout if unset)
"""

import os
import sys
import time
import json
import logging
import logging.handlers
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure, OperationFailure

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
# Configures a root logger with a consistent format, console output, and optional
def setup_logging() -> logging.Logger:
    """Configure root logger with console + optional rotating file handler."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file  = os.getenv("LOG_FILE")

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("mtg_pipeline")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    # Console handler (always on)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional rotating file handler (10 MB x 5 backups)
    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("File logging enabled -> %s", log_file)

    return logger


log = setup_logging()

# ── Config ────────────────────────────────────────────────────────────────────

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB         = os.getenv("MONGO_DB", "mtg")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "cards")

SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
BATCH_SIZE        = 500    # cards per MongoDB bulk write
REQUEST_DELAY     = 0.1    # seconds between batches (Scryfall rate-limit courtesy)
HTTP_TIMEOUT      = 60     # seconds for HTTP requests
MAX_RETRIES       = 3      # retry attempts for transient HTTP errors
RETRY_BACKOFF     = 1.0    # exponential backoff factor

# ── HTTP session with retry logic ─────────────────────────────────────────────

def build_http_session() -> requests.Session:
    """Return a requests.Session with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "mtg-pipeline/1.0"})
    return session


# ── Scryfall helpers ──────────────────────────────────────────────────────────
#These access the scryfall API, a website that has info on magic card data
def get_bulk_download_url(session: requests.Session) -> str:
    """Fetch the download URL for the latest 'default_cards' bulk export."""
    log.info("Fetching Scryfall bulk data index from %s", SCRYFALL_BULK_URL)
    try:
        resp = session.get(SCRYFALL_BULK_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        log.error("Timed out fetching Scryfall bulk index (timeout=%ds)", HTTP_TIMEOUT)
        raise
    except requests.exceptions.ConnectionError as exc:
        log.error("Network error fetching Scryfall bulk index: %s", exc)
        raise
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP %s from Scryfall bulk index: %s", resp.status_code, exc)
        raise

    try:
        payload = resp.json()
    except ValueError as exc:
        log.error("Failed to parse Scryfall bulk index JSON: %s", exc)
        raise

    for entry in payload.get("data", []):
        if entry.get("type") == "default_cards":
            size_mb = entry.get("size", 0) // 1_000_000
            log.info(
                "Found bulk export: '%s'  size=%dMB  updated=%s",
                entry.get("name"),
                size_mb,
                entry.get("updated_at", "unknown"),
            )
            return entry["download_uri"]

    raise ValueError("'default_cards' bulk export not found in Scryfall response.")

#downloads card data, filters by those that are standard legal (cards from 2024 to the current day, will reset next year)
def download_cards(url: str, session: requests.Session) -> list:
    """Stream-download the bulk JSON and return all Standard-legal cards."""
    log.info("Downloading bulk card data from %s", url)
    try:
        resp = session.get(url, timeout=HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        log.error("Timed out downloading bulk card data (timeout=%ds)", HTTP_TIMEOUT)
        raise
    except requests.exceptions.ConnectionError as exc:
        log.error("Network error downloading bulk card data: %s", exc)
        raise
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP %s downloading bulk data: %s", resp.status_code, exc)
        raise

    log.debug("Streaming response into memory...")
    chunks = []
    total_bytes = 0
    try:
        for chunk in resp.iter_content(chunk_size=65_536):
            chunks.append(chunk)
            total_bytes += len(chunk)
        raw = b"".join(chunks)
    except requests.exceptions.ChunkedEncodingError as exc:
        log.error("Download interrupted mid-stream after %d bytes: %s", total_bytes, exc)
        raise

    log.info("Download complete -- %.1f MB received", total_bytes / 1_000_000)

    try:
        all_cards = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Failed to parse bulk card JSON (offset %d): %s", exc.pos, exc.msg)
        raise

    if not isinstance(all_cards, list):
        raise TypeError(f"Expected a JSON array of cards, got {type(all_cards).__name__}")

    total = len(all_cards)
    log.info("Bulk export contains %d cards total", total)

    standard_cards = [
        c for c in all_cards
        if c.get("legalities", {}).get("standard") == "legal"
    ]
    log.info(
        "Standard-legal cards: %d (%.1f%%)",
        len(standard_cards),
        len(standard_cards) / total * 100,
    )
    return standard_cards


# ── Document builder ──────────────────────────────────────────────────────────
#builds a document for each card, extracting features useful for this project and inserting into mongo db.
def build_document(card: dict):
    """
    Extract and structure the fields most useful for win-probability modelling.
    Returns None (and logs a warning) if the card is missing required fields.
    """
    card_id   = card.get("id")
    card_name = card.get("name")

    if not card_id:
        log.warning("Skipping card with missing 'id' field: %s", card)
        return None
    if not card_name:
        log.warning("Skipping card id=%s with missing 'name' field", card_id)
        return None

    try:
        doc = {
            # --- Identity ---
            "scryfall_id":      card_id,
            "oracle_id":        card.get("oracle_id"),
            "name":             card_name,
            "set_code":         card.get("set"),
            "set_name":         card.get("set_name"),
            "collector_number": card.get("collector_number"),

            # --- Oracle text ---
            "mana_cost":        card.get("mana_cost"),
            "cmc":              float(card.get("cmc", 0.0)),
            "type_line":        card.get("type_line", ""),
            "oracle_text":      card.get("oracle_text", ""),
            "power":            card.get("power"),
            "toughness":        card.get("toughness"),
            "loyalty":          card.get("loyalty"),
            "keywords":         card.get("keywords", []),
            "colors":           card.get("colors", []),
            "color_identity":   card.get("color_identity", []),

            # --- Legalities ---
            "legalities":       card.get("legalities", {}),

            # --- Prices ---
            "prices": {
                "usd":      card.get("prices", {}).get("usd"),
                "usd_foil": card.get("prices", {}).get("usd_foil"),
            },

            # --- URLs ---
            "scryfall_uri": card.get("scryfall_uri"),
            "image_uris":   card.get("image_uris", {}),

            # --- Double-faced cards ---
            "card_faces": card.get("card_faces"),

            # --- Timestamps ---
            "released_at":  card.get("released_at"),
            "last_synced":  datetime.now(timezone.utc).isoformat(),

            # --- Full raw payload ---
            "raw": card,
        }
    except Exception as exc:
        log.warning(
            "Error building document for card id=%s name=%s: %s",
            card_id, card_name, exc,
        )
        return None

    return doc


# ── MongoDB helpers ───────────────────────────────────────────────────────────
#Hanldes connection to mongo db and inserting documents in batches.
def connect_to_mongo():
    """Connect to MongoDB Atlas and verify the connection with a ping."""
    log.info("Connecting to MongoDB at ...%s", MONGO_URI.split("@")[-1])
    try:
        client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10_000,
    tlsAllowInvalidCertificates=True
)
        client.admin.command("ping")
    except ConnectionFailure as exc:
        log.critical("Cannot connect to MongoDB: %s", exc)
        raise
    except Exception as exc:
        log.critical("Unexpected error connecting to MongoDB: %s", exc)
        raise

    collection = client[MONGO_DB][MONGO_COLLECTION]
    log.info("Connected -> %s.%s", MONGO_DB, MONGO_COLLECTION)
    return client, collection

#uploads to mongo in batches.
def upsert_to_mongo(documents: list, collection) -> dict:
    """
    Upsert documents in batches. Returns a summary dict with counts.
    Partial failures (single-batch errors) are logged but do not abort the run.
    """
    total   = len(documents)
    inserted = 0
    modified = 0
    failed   = 0
    batches  = (total + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(
        "Starting upsert: %d documents in %d batches (batch_size=%d)",
        total, batches, BATCH_SIZE,
    )

    for batch_num, start in enumerate(range(0, total, BATCH_SIZE), start=1):
        batch = documents[start : start + BATCH_SIZE]
        ops = [
            UpdateOne(
                {"scryfall_id": doc["scryfall_id"]},
                {"$set": doc},
                upsert=True,
            )
            for doc in batch
        ]

        try:
            result = collection.bulk_write(ops, ordered=False)
            inserted += result.upserted_count
            modified += result.modified_count
            log.debug(
                "Batch %d/%d -- upserted=%d modified=%d",
                batch_num, batches,
                result.upserted_count,
                result.modified_count,
            )
        except BulkWriteError as exc:
            details = exc.details
            inserted += details.get("nUpserted", 0)
            modified += details.get("nModified", 0)
            batch_failed = len(details.get("writeErrors", []))
            failed += batch_failed
            log.error(
                "Batch %d/%d partial failure -- %d write errors. First: %s",
                batch_num, batches,
                batch_failed,
                details["writeErrors"][0] if details.get("writeErrors") else "unknown",
            )
        except OperationFailure as exc:
            failed += len(batch)
            log.error(
                "Batch %d/%d failed entirely (OperationFailure): %s",
                batch_num, batches, exc,
            )
        except Exception as exc:
            failed += len(batch)
            log.error("Batch %d/%d unexpected error: %s", batch_num, batches, exc)

        pct = min(start + BATCH_SIZE, total) / total * 100
        log.info("Progress: %d/%d (%.0f%%)", min(start + BATCH_SIZE, total), total, pct)
        time.sleep(REQUEST_DELAY)

    summary = {"inserted": inserted, "modified": modified, "failed": failed}
    log.info(
        "Upsert complete -- inserted=%d  modified=%d  failed=%d",
        summary["inserted"], summary["modified"], summary["failed"],
    )
    return summary

#creates indexes for common querry patterns
def create_indexes(collection) -> None:
    """Create indexes for common query patterns. Skips indexes that already exist."""
    indexes = [
        {"keys": "scryfall_id", "kwargs": {"unique": True}},
        {"keys": "oracle_id",   "kwargs": {}},
        {"keys": "name",        "kwargs": {}},
        {"keys": "set_code",    "kwargs": {}},
        {"keys": "cmc",         "kwargs": {}},
        {"keys": "colors",      "kwargs": {}},
        {"keys": "keywords",    "kwargs": {}},
        {
            "keys":   [("name", "text"), ("oracle_text", "text")],
            "kwargs": {"name": "text_search"},
        },
    ]

    for idx in indexes:
        try:
            collection.create_index(idx["keys"], **idx["kwargs"])
            log.debug("Index ensured: %s", idx["keys"])
        except OperationFailure as exc:
            log.warning("Could not create index %s: %s", idx["keys"], exc)
        except Exception as exc:
            log.error("Unexpected error creating index %s: %s", idx["keys"], exc)

    log.info("Index creation complete.")


# ── Main ──────────────────────────────────────────────────────────────────────
#main function to run the full pipeline.
def main() -> int:
    """
    Run the full pipeline. Returns 0 on success, 1 on fatal error.
    Partial write failures still return 0 but are noted in the log summary.
    """
    start_time = time.monotonic()
    log.info("=" * 60)
    log.info("MTG Standard pipeline starting")
    log.info("=" * 60)

    session = build_http_session()
    client  = None

    try:
        # 1. Connect to MongoDB
        client, collection = connect_to_mongo()

        # 2. Fetch card data from Scryfall
        download_url   = get_bulk_download_url(session)
        standard_cards = download_cards(download_url, session)

        if not standard_cards:
            log.warning("No Standard-legal cards found -- nothing to upsert.")
            return 0

        # 3. Build structured documents (skip malformed cards)
        log.info("Building documents from %d cards...", len(standard_cards))
        documents  = [build_document(c) for c in standard_cards]
        valid_docs = [d for d in documents if d is not None]
        skipped    = len(documents) - len(valid_docs)

        if skipped:
            log.warning("Skipped %d malformed cards during document build.", skipped)
        log.info("Valid documents ready: %d", len(valid_docs))

        if not valid_docs:
            log.error("All documents were invalid -- aborting upsert.")
            return 1

        # 4. Upsert into MongoDB
        summary = upsert_to_mongo(valid_docs, collection)

        # 5. Ensure indexes exist
        create_indexes(collection)

        # 6. Final summary
        elapsed     = time.monotonic() - start_time
        total_in_db = collection.count_documents({"legalities.standard": "legal"})

        log.info("=" * 60)
        log.info("Pipeline complete in %.1fs", elapsed)
        log.info("  Cards fetched     : %d", len(standard_cards))
        log.info("  Documents built   : %d  (skipped=%d)", len(valid_docs), skipped)
        log.info("  Upserted (new)    : %d", summary["inserted"])
        log.info("  Modified (updated): %d", summary["modified"])
        log.info("  Write failures    : %d", summary["failed"])
        log.info("  Total in DB now   : %d", total_in_db)
        log.info("=" * 60)

        if summary["failed"] > 0:
            log.warning(
                "%d documents failed to write -- check logs above for details.",
                summary["failed"],
            )

        return 0

    except (requests.exceptions.RequestException, ValueError, TypeError) as exc:
        log.critical("Fatal error during Scryfall fetch: %s", exc, exc_info=True)
        return 1
    except (ConnectionFailure, OperationFailure) as exc:
        log.critical("Fatal MongoDB error: %s", exc, exc_info=True)
        return 1
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user.")
        return 1
    except Exception as exc:
        log.critical("Unexpected fatal error: %s", exc, exc_info=True)
        return 1
    finally:
        if client:
            client.close()
            log.info("MongoDB connection closed.")


if __name__ == "__main__":
    sys.exit(main())