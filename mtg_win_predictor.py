"""
MTG Deck Win Probability Predictor
------------------------------------
Reads card data directly from the MongoDB collection populated by
load_standard_cards.py. Each document follows the schema produced by
build_document() in that script.

Pipeline:
    1. CardScorer      — converts card documents into a single numeric score
    2. DeckAggregator  — aggregates card scores into deck-level features
    3. WinPredictor    — XGBoost model: P(deck_a wins) given two decks

Environment variables (same .env as load_standard_cards.py):
    MONGO_URI        mongodb+srv://...
    MONGO_DB         mtg
    MONGO_COLLECTION cards
    LOG_LEVEL        INFO  (optional)
    LOG_FILE         predictor.log  (optional)

Usage:
    # Train (requires a matchup DataFrame with deck_a, deck_b, winner columns)
    trainer = ModelTrainer.from_mongo()
    metrics = trainer.train(matchup_df)
    trainer.save("mtg_model.pkl")

    # Predict
    predictor = WinPredictor.load("mtg_model.pkl")
    result = predictor.predict(deck_a_names, deck_b_names)
"""

import logging
import logging.handlers
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file  = os.getenv("LOG_FILE")

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("mtg_predictor")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("File logging -> %s", log_file)

    return logger


logger = setup_logging()

# ---------------------------------------------------------------------------
# MongoDB config  (mirrors load_standard_cards.py)
# ---------------------------------------------------------------------------

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB         = os.getenv("MONGO_DB", "mtg")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "cards")

RARITY_MAP = {"common": 1, "uncommon": 2, "rare": 3, "mythic": 4, "special": 3}

# ---------------------------------------------------------------------------
# Feature config
# Each entry: feature_name -> (extractor_fn, default, weight)
# Extractor receives a single Mongo document (dict) and returns a float.
# Prices live at doc["prices"]["usd"], rarity at doc["raw"]["rarity"] —
# matching exactly what build_document() in load_standard_cards.py produces.
# Adjust weights to encode domain knowledge; the model refines them further.
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


FEATURE_CONFIG: dict[str, tuple] = {
    "cmc": (
        lambda doc: _safe_float(doc.get("cmc"), 0.0),
        0.0,
        0.15,
    ),
    "price_usd": (
        # Nested under prices.usd per build_document()
        lambda doc: _safe_float(doc.get("prices", {}).get("usd"), 0.0),
        0.0,
        0.25,
    ),
    "price_usd_foil": (
        lambda doc: _safe_float(doc.get("prices", {}).get("usd_foil"), 0.0),
        0.0,
        0.10,
    ),
    "power": (
        # Top-level string field — coerce to float, non-numeric ("*") -> 0
        lambda doc: _safe_float(doc.get("power"), 0.0),
        0.0,
        0.20,
    ),
    "toughness": (
        lambda doc: _safe_float(doc.get("toughness"), 0.0),
        0.0,
        0.15,
    ),
    "rarity_score": (
        # Rarity sits inside doc["raw"]["rarity"] per build_document()
        lambda doc: float(
            RARITY_MAP.get(
                str((doc.get("raw") or {}).get("rarity", "")).lower(), 0
            )
        ),
        0.0,
        0.15,
    ),
}

# ---------------------------------------------------------------------------
# MongoDB connection helpers
# ---------------------------------------------------------------------------

def get_collection():
    """Connect to MongoDB and return (client, collection)."""
    logger.info("Connecting to MongoDB...")
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=10_000,
            tlsAllowInvalidCertificates=True,
        )
        client.admin.command("ping")
        collection = client[MONGO_DB][MONGO_COLLECTION]
        logger.info("Connected -> %s.%s", MONGO_DB, MONGO_COLLECTION)
        return client, collection
    except ConnectionFailure as exc:
        logger.critical("Cannot connect to MongoDB: %s", exc)
        raise
    except Exception as exc:
        logger.critical("Unexpected MongoDB connection error: %s", exc)
        raise


def load_cards_from_mongo(collection) -> list[dict]:
    """
    Fetch only the fields required for scoring from all Standard-legal cards.
    Using a projection keeps memory usage low on large collections.
    """
    projection = {
        "name":       1,
        "cmc":        1,
        "power":      1,
        "toughness":  1,
        "prices":     1,
        "raw.rarity": 1,
        "_id":        0,
    }
    logger.info("Fetching Standard-legal cards (projected fields only)...")
    try:
        cards = list(
            collection.find({"legalities.standard": "legal"}, projection)
        )
        logger.info("Loaded %d cards from MongoDB", len(cards))
        return cards
    except OperationFailure as exc:
        logger.error("MongoDB query failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error loading cards: %s", exc)
        raise


# ---------------------------------------------------------------------------
# CardScorer
# ---------------------------------------------------------------------------

class CardScorer:
    """
    Converts a list of MongoDB card documents into numeric card scores.
    Score = weighted sum of normalised features.
    """

    def __init__(self, feature_config: dict = FEATURE_CONFIG):
        self.feature_config = feature_config
        self.scaler = StandardScaler()
        self._fitted = False

    def _extract_matrix(self, cards: list[dict]) -> np.ndarray:
        """Return an (n_cards x n_features) float array."""
        n, k = len(cards), len(self.feature_config)
        matrix = np.zeros((n, k), dtype=float)

        for j, (feat_name, (extractor, default, _)) in enumerate(self.feature_config.items()):
            for i, doc in enumerate(cards):
                try:
                    matrix[i, j] = extractor(doc)
                except Exception as exc:
                    logger.debug(
                        "Feature '%s' failed for card '%s': %s — using default %.2f",
                        feat_name, doc.get("name", "?"), exc, default,
                    )
                    matrix[i, j] = default

        return matrix

    def fit(self, cards: list[dict]) -> "CardScorer":
        if not cards:
            raise ValueError("Cannot fit CardScorer on an empty card list.")
        try:
            matrix = self._extract_matrix(cards)
            self.scaler.fit(matrix)
            self._fitted = True
            logger.info(
                "CardScorer fitted — %d cards, %d features", len(cards), matrix.shape[1]
            )
        except Exception as exc:
            logger.error("CardScorer.fit failed: %s", exc, exc_info=True)
            raise
        return self

    def transform(self, cards: list[dict]) -> np.ndarray:
        """Return a 1-D array of card scores (one per document)."""
        if not self._fitted:
            raise RuntimeError("CardScorer must be fitted before transform.")
        try:
            matrix = self._extract_matrix(cards)
            normed  = self.scaler.transform(matrix)
            weights = np.array([w for _, _, w in self.feature_config.values()])
            weights = weights / weights.sum()
            scores  = normed @ weights
            logger.debug(
                "Scored %d cards — mean=%.4f  std=%.4f", len(scores), scores.mean(), scores.std()
            )
            return scores
        except Exception as exc:
            logger.error("CardScorer.transform failed: %s", exc, exc_info=True)
            raise

    def fit_transform(self, cards: list[dict]) -> np.ndarray:
        return self.fit(cards).transform(cards)


# ---------------------------------------------------------------------------
# DeckAggregator
# ---------------------------------------------------------------------------

class DeckAggregator:
    """
    Converts a decklist (list of card name strings) into a fixed-length
    feature vector by looking up each card's score and computing aggregates.
    """

    def __init__(self, cards: list[dict], card_scorer: CardScorer):
        scores = card_scorer.transform(cards)

        # name (lowercase) -> mean score across all printings
        lookup: dict[str, list[float]] = {}
        for doc, score in zip(cards, scores):
            name = doc.get("name", "").strip().lower()
            if name:
                lookup.setdefault(name, []).append(float(score))

        self.score_lookup: dict[str, float] = {
            name: float(np.mean(vals)) for name, vals in lookup.items()
        }
        logger.info(
            "DeckAggregator ready — %d unique card names in lookup",
            len(self.score_lookup),
        )

    def _lookup(self, card_name: str) -> Optional[float]:
        score = self.score_lookup.get(card_name.strip().lower())
        if score is None:
            logger.warning("Card not in lookup: '%s' — skipping", card_name)
        return score

    def aggregate(self, decklist: list[str]) -> dict[str, float]:
        """
        Given a list of card name strings (duplicates = playsets),
        return a dict of deck-level features.
        """
        if not decklist:
            raise ValueError("Decklist is empty.")

        scores  = [s for name in decklist if (s := self._lookup(name)) is not None]
        missing = len(decklist) - len(scores)

        if not scores:
            raise ValueError("No cards in decklist matched the card pool.")
        if missing:
            logger.warning(
                "%d/%d cards not recognised (not Standard-legal or misspelled)",
                missing, len(decklist),
            )

        arr = np.array(scores)
        return {
            "deck_mean_score":  float(arr.mean()),
            "deck_sum_score":   float(arr.sum()),
            "deck_max_score":   float(arr.max()),
            "deck_min_score":   float(arr.min()),
            "deck_std_score":   float(arr.std()),
            "deck_p25_score":   float(np.percentile(arr, 25)),
            "deck_p75_score":   float(np.percentile(arr, 75)),
            "deck_p90_score":   float(np.percentile(arr, 90)),
            "deck_size":        float(len(scores)),
            "deck_missing_pct": float(missing / len(decklist)),
        }


# ---------------------------------------------------------------------------
# Matchup feature builder
# ---------------------------------------------------------------------------

def build_matchup_features(deck_a: dict, deck_b: dict) -> dict:
    """
    Combine two deck feature dicts into a single matchup vector.
    Delta and ratio features allow XGBoost to compare decks directly.
    """
    combined = {}
    for k in deck_a:
        combined[f"a_{k}"] = deck_a[k]
        combined[f"b_{k}"] = deck_b[k]
        combined[f"delta_{k}"] = deck_a[k] - deck_b[k]

    denom = deck_b["deck_mean_score"] if deck_b["deck_mean_score"] != 0 else 1e-9
    combined["score_ratio"] = deck_a["deck_mean_score"] / denom
    return combined


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Trains the full pipeline end-to-end.

    matchup_df expected columns:
        deck_a  — list[str] of card names
        deck_b  — list[str] of card names
        winner  — int: 1 if deck_a won, 0 if deck_b won
    """

    def __init__(self, cards: list[dict]):
        if not cards:
            raise ValueError("Card list is empty — cannot initialise trainer.")
        logger.info("Initialising ModelTrainer with %d cards", len(cards))
        self.card_scorer = CardScorer()
        self.card_scorer.fit(cards)
        self.aggregator = DeckAggregator(cards, self.card_scorer)
        self.model: Optional[XGBClassifier] = None
        self.feature_names: list[str] = []

    @classmethod
    def from_mongo(cls) -> "ModelTrainer":
        """Convenience constructor — loads cards directly from MongoDB."""
        client, collection = get_collection()
        try:
            cards = load_cards_from_mongo(collection)
        finally:
            client.close()
            logger.debug("MongoDB connection closed after card load.")
        return cls(cards)

    def _build_training_data(self, matchup_df: pd.DataFrame):
        rows, labels, skipped = [], [], 0

        for idx, row in matchup_df.iterrows():
            try:
                feats_a = self.aggregator.aggregate(row["deck_a"])
                feats_b = self.aggregator.aggregate(row["deck_b"])
                rows.append(build_matchup_features(feats_a, feats_b))
                labels.append(int(row["winner"]))
            except Exception as exc:
                logger.warning("Skipping matchup row %s: %s", idx, exc)
                skipped += 1

        if skipped:
            logger.warning(
                "Skipped %d/%d matchup rows during feature build", skipped, len(matchup_df)
            )
        if not rows:
            raise ValueError("No valid matchup rows — check your matchup_df format.")

        X = pd.DataFrame(rows)
        y = np.array(labels)
        logger.info("Training matrix: %d rows x %d features", len(X), X.shape[1])
        return X, y

    def train(self, matchup_df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train XGBoost. Returns eval metrics dict."""
        logger.info("Starting training on %d matchups", len(matchup_df))
        try:
            X, y = self._build_training_data(matchup_df)
            self.feature_names = list(X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            probs = self.model.predict_proba(X_test)[:, 1]
            metrics = {
                "roc_auc":  round(roc_auc_score(y_test, probs), 4),
                "log_loss": round(log_loss(y_test, probs), 4),
                "n_train":  len(X_train),
                "n_test":   len(X_test),
            }
            logger.info(
                "Training complete — ROC-AUC: %s | Log-loss: %s",
                metrics["roc_auc"], metrics["log_loss"],
            )
            return metrics
        except Exception as exc:
            logger.error("Training failed: %s", exc, exc_info=True)
            raise

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return (
            pd.DataFrame({
                "feature":    self.feature_names,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str) -> None:
        payload = {
            "card_scorer":   self.card_scorer,
            "aggregator":    self.aggregator,
            "model":         self.model,
            "feature_names": self.feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Model saved -> %s", path)


# ---------------------------------------------------------------------------
# WinPredictor  (inference only)
# ---------------------------------------------------------------------------

class WinPredictor:
    """
    Loads a saved pipeline and predicts P(deck_a wins).

    deck_a / deck_b: list of card name strings, duplicates allowed for playsets.
    e.g. ["Lightning Bolt", "Lightning Bolt", "Mountain", ...]
    """

    def __init__(self, card_scorer, aggregator, model, feature_names):
        self.card_scorer   = card_scorer
        self.aggregator    = aggregator
        self.model         = model
        self.feature_names = feature_names

    @classmethod
    def load(cls, path: str) -> "WinPredictor":
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        logger.info("Model loaded <- %s", path)
        return cls(**payload)

    def predict(self, deck_a: list[str], deck_b: list[str]) -> dict:
        """
        Returns:
            prob_a_wins   float  P(deck_a wins)
            prob_b_wins   float  1 - prob_a_wins
            deck_a_score  float  mean card score for deck A
            deck_b_score  float  mean card score for deck B
        """
        try:
            feats_a = self.aggregator.aggregate(deck_a)
            feats_b = self.aggregator.aggregate(deck_b)
            matchup = build_matchup_features(feats_a, feats_b)

            X      = pd.DataFrame([matchup])[self.feature_names]
            prob_a = float(self.model.predict_proba(X)[0, 1])

            result = {
                "prob_a_wins":  round(prob_a, 4),
                "prob_b_wins":  round(1 - prob_a, 4),
                "deck_a_score": round(feats_a["deck_mean_score"], 4),
                "deck_b_score": round(feats_b["deck_mean_score"], 4),
            }
            logger.info(
                "Prediction — P(A wins): %.1f%%  A_score=%.3f  B_score=%.3f",
                prob_a * 100, feats_a["deck_mean_score"], feats_b["deck_mean_score"],
            )
            return result
        except Exception as exc:
            logger.error("Prediction failed: %s", exc, exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MTG Win Predictor — example run")
    logger.info("=" * 60)

    # 1. Load cards from MongoDB and initialise trainer
    try:
        trainer = ModelTrainer.from_mongo()
    except Exception as exc:
        logger.critical("Could not initialise trainer: %s", exc)
        sys.exit(1)

    # 2. Build synthetic matchups for smoke-testing.
    #    In production, replace matchup_df with real match result data.
    all_names = list(trainer.aggregator.score_lookup.keys())
    if len(all_names) < 60:
        logger.error("Not enough cards in pool to build test decks.")
        sys.exit(1)

    rng = np.random.default_rng(42)

    def random_deck(size: int = 60) -> list[str]:
        return rng.choice(all_names, size=size, replace=True).tolist()

    n_matchups = 500
    matchup_df = pd.DataFrame({
        "deck_a":  [random_deck() for _ in range(n_matchups)],
        "deck_b":  [random_deck() for _ in range(n_matchups)],
        "winner":  rng.integers(0, 2, size=n_matchups).tolist(),
    })
    logger.info("Generated %d synthetic matchups for smoke test", n_matchups)

    # 3. Train
    try:
        metrics = trainer.train(matchup_df)
        logger.info("Eval metrics: %s", metrics)
    except Exception as exc:
        logger.critical("Training failed: %s", exc)
        sys.exit(1)

    print("\nTop 10 features by importance:")
    print(trainer.feature_importance().head(10).to_string(index=False))

    # 4. Save and reload
    trainer.save("mtg_model.pkl")
    predictor = WinPredictor.load("mtg_model.pkl")

    # 5. Example prediction
    result = predictor.predict(random_deck(), random_deck())
    print("\nExample prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
