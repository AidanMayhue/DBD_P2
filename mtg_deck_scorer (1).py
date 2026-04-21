"""
MTG Deck Scorer
----------------
Scores two decklists purely from card features in MongoDB.
No match data or training required.

How it works:
    1. Each card gets a raw score from a weighted combination of features.
    2. Each deck gets aggregated stats from its cards' scores.
    3. A final deck power score is computed and converted to a win probability
       via a sigmoid function so the output is always between 0 and 1.

Scoring intuitions baked into the weights:
    - Lower average CMC  → faster, more consistent deck  (negative weight on CMC)
    - Higher price       → more powerful/optimised cards
    - Higher power/toughness → better combat stats
    - Higher rarity      → generally more impactful cards

Environment variables (.env):
    MONGO_URI         mongodb+srv://...
    MONGO_DB          mtg
    MONGO_COLLECTION  cards
    LOG_LEVEL         INFO   (optional)
    LOG_FILE          scorer.log  (optional)

Usage:
    scorer = DeckScorer.from_mongo()

    deck_a = ["Lightning Bolt", "Lightning Bolt", "Mountain", ...]
    deck_b = ["Llanowar Elves", "Forest", ...]

    result = scorer.compare(deck_a, deck_b)
    print(result)
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from sklearn.preprocessing import StandardScaler


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
    logger = logging.getLogger("mtg_scorer")
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
# Config
# ---------------------------------------------------------------------------

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB         = os.getenv("MONGO_DB", "mtg")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "cards")

RARITY_MAP = {"common": 1, "uncommon": 2, "rare": 3, "mythic": 4, "special": 3}

# Feature weights — edit these to change scoring priorities.
# Negative CMC weight: lower mana curve = higher score.
# All other weights are positive: more = better.
FEATURE_WEIGHTS = {
    "cmc":            -0.20,   # lower curve is stronger
    "price_usd":       0.30,   # expensive cards are generally better
    "price_usd_foil":  0.10,   # secondary price signal
    "power":           0.20,   # combat stats matter
    "toughness":       0.10,   # survivability
    "rarity_score":    0.10,   # rarer = more impactful on average
}

# How aggressively the sigmoid converts score difference to probability.
# Higher = more decisive predictions. Lower = closer to 50/50.
SIGMOID_SCALE = 2.0

# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def get_collection():
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


def load_cards(collection) -> list[dict]:
    """Fetch only scoring-relevant fields for all Standard-legal cards."""
    projection = {
        "name":       1,
        "cmc":        1,
        "power":      1,
        "toughness":  1,
        "prices":     1,
        "raw.rarity": 1,
        "_id":        0,
    }
    logger.info("Loading Standard-legal cards from MongoDB...")
    try:
        cards = list(collection.find({"legalities.standard": "legal"}, projection))
        logger.info("Loaded %d cards", len(cards))
        return cards
    except OperationFailure as exc:
        logger.error("MongoDB query failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error loading cards: %s", exc)
        raise

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def extract_features(doc: dict) -> dict[str, float]:
    """Extract raw numeric features from a single MongoDB card document."""
    rarity_str = str((doc.get("raw") or {}).get("rarity", "")).lower()
    return {
        "cmc":            _safe_float(doc.get("cmc"), 0.0),
        "price_usd":      _safe_float(doc.get("prices", {}).get("usd"), 0.0),
        "price_usd_foil": _safe_float(doc.get("prices", {}).get("usd_foil"), 0.0),
        "power":          _safe_float(doc.get("power"), 0.0),
        "toughness":      _safe_float(doc.get("toughness"), 0.0),
        "rarity_score":   float(RARITY_MAP.get(rarity_str, 0)),
    }

# ---------------------------------------------------------------------------
# DeckScorer
# ---------------------------------------------------------------------------

class DeckScorer:
    """
    Scores decklists using a weighted combination of card features.
    No training data required.
    """

    def __init__(self, cards: list[dict]):
        if not cards:
            raise ValueError("Card list is empty.")

        self._build_lookup(cards)
        logger.info(
            "DeckScorer ready — %d unique card names", len(self.score_lookup)
        )

    def _build_lookup(self, cards: list[dict]) -> None:
        """
        Normalise features across the full card pool, apply weights,
        then store a mean score per card name.
        """
        feature_names = list(FEATURE_WEIGHTS.keys())
        weights       = np.array([FEATURE_WEIGHTS[f] for f in feature_names])

        # Build feature matrix
        matrix = np.array([
            [extract_features(doc)[f] for f in feature_names]
            for doc in cards
        ], dtype=float)

        # Normalise so each feature contributes on the same scale
        scaler = StandardScaler()
        normed = scaler.fit_transform(matrix)

        # Weighted sum -> raw card scores
        raw_scores = normed @ weights

        # Group by name (lowercase) and average across printings
        lookup: dict[str, list[float]] = {}
        for doc, score in zip(cards, raw_scores):
            name = doc.get("name", "").strip().lower()
            if name:
                lookup.setdefault(name, []).append(float(score))

        self.score_lookup: dict[str, float] = {
            name: float(np.mean(vals)) for name, vals in lookup.items()
        }

        # Also store per-feature normalised means for breakdown reporting
        self._scaler        = scaler
        self._feature_names = feature_names
        self._weights       = weights

    @classmethod
    def from_mongo(cls) -> "DeckScorer":
        client, collection = get_collection()
        try:
            cards = load_cards(collection)
        finally:
            client.close()
            logger.debug("MongoDB connection closed.")
        return cls(cards)

    def _lookup(self, card_name: str) -> Optional[float]:
        score = self.score_lookup.get(card_name.strip().lower())
        if score is None:
            logger.warning("Card not found: '%s' — skipping", card_name)
        return score

    def score_deck(self, decklist: list[str]) -> dict:
        """
        Compute aggregate stats for a decklist.
        Returns a dict with mean, top10, cmc_avg and other useful metrics.
        """
        if not decklist:
            raise ValueError("Decklist is empty.")

        scores  = [s for name in decklist if (s := self._lookup(name)) is not None]
        missing = len(decklist) - len(scores)

        if not scores:
            raise ValueError("No cards in decklist matched the card pool.")
        if missing:
            logger.warning(
                "%d/%d cards not recognised — excluded from scoring",
                missing, len(decklist),
            )

        arr = np.array(scores)
        return {
            "mean_score":   float(arr.mean()),
            "top10_score":  float(np.percentile(arr, 90)),  # ceiling of the deck
            "consistency":  float(-arr.std()),               # lower variance = more consistent
            "deck_size":    len(scores),
            "missing":      missing,
        }

    def _sigmoid(self, x: float) -> float:
        """Map score difference to (0, 1) probability."""
        return 1.0 / (1.0 + np.exp(-SIGMOID_SCALE * x))

    def compare(self, deck_a: list[str], deck_b: list[str]) -> dict:
        """
        Compare two decklists and return win probabilities + explanations.

        Returns
        -------
        dict with keys:
            prob_a_wins     float   P(deck A wins)
            prob_b_wins     float   P(deck B wins)
            deck_a          dict    per-deck score breakdown
            deck_b          dict    per-deck score breakdown
            reasoning       list    human-readable factors behind the prediction
        """
        logger.info("Scoring deck A (%d cards)...", len(deck_a))
        stats_a = self.score_deck(deck_a)

        logger.info("Scoring deck B (%d cards)...", len(deck_b))
        stats_b = self.score_deck(deck_b)

        # Combined power score: weight mean score heavily, top10 as a ceiling signal
        power_a = stats_a["mean_score"] * 0.6 + stats_a["top10_score"] * 0.3 + stats_a["consistency"] * 0.1
        power_b = stats_b["mean_score"] * 0.6 + stats_b["top10_score"] * 0.3 + stats_b["consistency"] * 0.1

        score_diff = power_a - power_b
        prob_a     = self._sigmoid(score_diff)
        prob_b     = 1.0 - prob_a

        # Build human-readable reasoning
        reasoning = _build_reasoning(deck_a, deck_b, stats_a, stats_b, self)

        result = {
            "prob_a_wins": round(prob_a, 4),
            "prob_b_wins": round(prob_b, 4),
            "deck_a": {
                "mean_score":  round(stats_a["mean_score"], 4),
                "top10_score": round(stats_a["top10_score"], 4),
                "deck_size":   stats_a["deck_size"],
                "missing":     stats_a["missing"],
            },
            "deck_b": {
                "mean_score":  round(stats_b["mean_score"], 4),
                "top10_score": round(stats_b["top10_score"], 4),
                "deck_size":   stats_b["deck_size"],
                "missing":     stats_b["missing"],
            },
            "reasoning": reasoning,
        }

        logger.info(
            "Result — P(A wins): %.1f%%  P(B wins): %.1f%%",
            prob_a * 100, prob_b * 100,
        )
        return result


# ---------------------------------------------------------------------------
# Reasoning builder
# ---------------------------------------------------------------------------

def _avg_feature(decklist: list[str], feature: str, scorer: DeckScorer) -> float:
    """Compute the average raw (un-normalised) feature value across a decklist."""
    vals = []
    for name in decklist:
        key = name.strip().lower()
        # We need the raw feature — re-extract from a dummy lookup isn't ideal,
        # so we store nothing here. Instead we just use the score_lookup delta
        # as a proxy. For richer breakdowns, extend DeckScorer to cache raw features.
    return float(np.mean(vals)) if vals else 0.0


def _build_reasoning(
    deck_a: list[str],
    deck_b: list[str],
    stats_a: dict,
    stats_b: dict,
    scorer: DeckScorer,
) -> list[str]:
    """Return a short list of plain-English factors explaining the prediction."""
    reasons = []

    # Mean score comparison
    delta = stats_a["mean_score"] - stats_b["mean_score"]
    if abs(delta) > 0.05:
        winner = "Deck A" if delta > 0 else "Deck B"
        reasons.append(
            f"{winner} has a higher average card score "
            f"(Δ={abs(delta):.3f}), suggesting stronger individual cards."
        )

    # Top-end comparison
    top_delta = stats_a["top10_score"] - stats_b["top10_score"]
    if abs(top_delta) > 0.05:
        winner = "Deck A" if top_delta > 0 else "Deck B"
        reasons.append(
            f"{winner} has a stronger top-10% of cards (ceiling score Δ={abs(top_delta):.3f})."
        )

    # Consistency
    cons_delta = stats_a["consistency"] - stats_b["consistency"]
    if abs(cons_delta) > 0.05:
        winner = "Deck A" if cons_delta > 0 else "Deck B"
        reasons.append(
            f"{winner} is more consistent (lower variance in card quality)."
        )

    # Missing cards warning
    if stats_a["missing"] > 5:
        reasons.append(
            f"Warning: {stats_a['missing']} cards in Deck A were not found "
            f"in the card pool and were excluded from scoring."
        )
    if stats_b["missing"] > 5:
        reasons.append(
            f"Warning: {stats_b['missing']} cards in Deck B were not found "
            f"in the card pool and were excluded from scoring."
        )

    if not reasons:
        reasons.append("Decks are very evenly matched — prediction is close to 50/50.")

    return reasons


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_deck_comparison(
    deck_a: list[str],
    deck_b: list[str],
    scorer: "DeckScorer",
    deck_a_label: str = "Deck A",
    deck_b_label: str = "Deck B",
    output_path: str = "deck_comparison.png",
) -> None:
    """
    Horizontal bar chart of every unique card across both decklists,
    ordered by score (highest at top). Bars are coloured by deck membership:
        - Deck A only  → blue
        - Deck B only  → coral
        - In both      → purple

    Parameters
    ----------
    deck_a, deck_b   : raw decklists (duplicates allowed — they are deduped here)
    scorer           : fitted DeckScorer instance
    deck_a_label     : display name for deck A
    deck_b_label     : display name for deck B
    output_path      : file to save the figure (PNG)
    """
    COLOR_A    = "#378ADD"   # blue  — Deck A
    COLOR_B    = "#D85A30"   # coral — Deck B
    COLOR_BOTH = "#7F77DD"   # purple — shared

    # Deduplicate while preserving which deck each card belongs to
    unique_a = set(n.strip() for n in deck_a)
    unique_b = set(n.strip() for n in deck_b)
    all_cards = unique_a | unique_b

    # Build (card_name, score, deck_membership) rows — skip unknowns
    rows = []
    for name in all_cards:
        score = scorer.score_lookup.get(name.strip().lower())
        if score is None:
            logger.warning("Plot: card '%s' not in lookup — skipping", name)
            continue
        if name in unique_a and name in unique_b:
            membership = "both"
        elif name in unique_a:
            membership = "a"
        else:
            membership = "b"
        rows.append((name, score, membership))

    if not rows:
        logger.error("No scoreable cards found — skipping plot.")
        return

    # Sort by score ascending so highest score is at the top of the chart
    rows.sort(key=lambda r: r[1])
    names      = [r[0] for r in rows]
    scores     = [r[1] for r in rows]
    colors     = [
        COLOR_BOTH if r[2] == "both" else (COLOR_A if r[2] == "a" else COLOR_B)
        for r in rows
    ]

    # Dynamic figure height — give each card ~0.32 inches of vertical space
    fig_height = max(6, len(rows) * 0.32)
    fig, ax    = plt.subplots(figsize=(10, fig_height))

    y_pos = range(len(names))
    bars  = ax.barh(list(y_pos), scores, color=colors, edgecolor="white", linewidth=0.4, height=0.7)

    # Score labels at the end of each bar
    for bar, score in zip(bars, scores):
        x_offset = 0.01 if score >= 0 else -0.01
        ha        = "left" if score >= 0 else "right"
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center", ha=ha,
            fontsize=7.5,
            color="#444441",
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Card score (normalised)", fontsize=10)
    ax.set_title("Card scores by deck", fontsize=12, fontweight="medium", pad=12)

    # Zero reference line
    ax.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR_A,    label=deck_a_label),
        mpatches.Patch(color=COLOR_B,    label=deck_b_label),
        mpatches.Patch(color=COLOR_BOTH, label="In both decks"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#D3D1C7",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D3D1C7")
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.tick_params(colors="#5F5E5A")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved -> %s", output_path)
    except Exception as exc:
        logger.error("Failed to save plot: %s", exc)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MTG Deck Scorer")
    logger.info("=" * 60)

    try:
        scorer = DeckScorer.from_mongo()
    except Exception as exc:
        logger.critical("Failed to initialise DeckScorer: %s", exc)
        sys.exit(1)

    # Example decklists — replace with real ones
    deck_a = [
        "Lightning Bolt", "Lightning Bolt", "Lightning Bolt", "Lightning Bolt",
        "Monastery Swiftspear", "Monastery Swiftspear", "Monastery Swiftspear",
        "Goblin Guide", "Goblin Guide", "Goblin Guide", "Goblin Guide",
        "Eidolon of the Great Revel", "Eidolon of the Great Revel",
        "Light Up the Stage", "Light Up the Stage", "Light Up the Stage",
        "Shard Volley", "Shard Volley", "Shard Volley", "Shard Volley",
        "Skullcrack", "Skullcrack", "Skullcrack", "Skullcrack",
        "Inspiring Vantage", "Sacred Foundry", "Sacred Foundry",
        "Mountain", "Mountain", "Mountain", "Mountain", "Mountain",
        "Mountain", "Mountain", "Mountain", "Mountain", "Mountain",
        "Mountain", "Mountain", "Mountain", "Mountain", "Mountain",
    ]

    deck_b = [
        "Tarmogoyf", "Tarmogoyf", "Tarmogoyf", "Tarmogoyf",
        "Snapcaster Mage", "Snapcaster Mage", "Snapcaster Mage",
        "Thoughtseize", "Thoughtseize", "Thoughtseize", "Thoughtseize",
        "Fatal Push", "Fatal Push", "Fatal Push", "Fatal Push",
        "Liliana of the Veil", "Liliana of the Veil", "Liliana of the Veil",
        "Inquisition of Kozilek", "Inquisition of Kozilek", "Inquisition of Kozilek",
        "Bloodstained Mire", "Polluted Delta", "Verdant Catacombs",
        "Swamp", "Swamp", "Swamp", "Swamp", "Forest", "Forest",
        "Island", "Island", "Watery Grave", "Watery Grave",
        "Overgrown Tomb", "Overgrown Tomb", "Breeding Pool",
        "Shelldock Isle", "Creeping Tar Pit",
        "Damnation", "Damnation",
    ]

    try:
        result = scorer.compare(deck_a, deck_b)
    except Exception as exc:
        logger.critical("Comparison failed: %s", exc)
        sys.exit(1)

    print("\n=== Result ===")
    print(f"  P(Deck A wins): {result['prob_a_wins'] * 100:.1f}%")
    print(f"  P(Deck B wins): {result['prob_b_wins'] * 100:.1f}%")
    print(f"\n  Deck A — mean score: {result['deck_a']['mean_score']}  "
          f"top10: {result['deck_a']['top10_score']}  "
          f"cards scored: {result['deck_a']['deck_size']}")
    print(f"  Deck B — mean score: {result['deck_b']['mean_score']}  "
          f"top10: {result['deck_b']['top10_score']}  "
          f"cards scored: {result['deck_b']['deck_size']}")
    print("\n  Reasoning:")
    for r in result["reasoning"]:
        print(f"    - {r}")

    plot_deck_comparison(
        deck_a, deck_b, scorer,
        deck_a_label="Burn (Deck A)",
        deck_b_label="Jund (Deck B)",
        output_path="deck_comparison.png",
    )
