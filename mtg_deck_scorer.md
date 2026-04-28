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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
#sets up loggin with file logging and console logging.
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
#connection strings for mongo database
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB         = os.getenv("MONGO_DB", "mtg")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "cards")

RARITY_MAP = {"common": 1, "uncommon": 2, "rare": 3, "mythic": 4, "special": 3}

# Ordered list of features fed into PCA.
# PCA will learn the weights from the card pool data itself.
FEATURE_NAMES = ["cmc", "price_usd", "price_usd_foil", "power", "toughness", "rarity_score"]

# How aggressively the sigmoid converts score difference to probability.
# Higher = more decisive predictions. Lower = closer to 50/50.
SIGMOID_SCALE = 2.0

# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------
#accesses mongo db for collections.
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

#loads cards and deduplicates them, keeping the cheapest variant.
def load_cards(collection) -> list[dict]:
    """
    Fetch Standard-legal cards and deduplicate to one document per unique
    card name — keeping the cheapest printing by USD price.

    Why: cards like Island have 900+ printings ranging from $0.01 to $5.
    Using the mean would inflate the price signal for heavily-reprinted cards.
    The cheapest printing best represents the floor cost of including a card.

    Cards with no USD price are treated as $0 so they are never preferred
    over a card that does have a price.
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
    logger.info("Loading Standard-legal cards from MongoDB...")
    try:
        all_cards = list(collection.find({"legalities.standard": "legal"}, projection))
        logger.info("Loaded %d printings from MongoDB", len(all_cards))
    except OperationFailure as exc:
        logger.error("MongoDB query failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error loading cards: %s", exc)
        raise

    # Group by lowercase name, keep the printing with the lowest USD price
    cheapest: dict[str, tuple] = {}
    for card in all_cards:
        name = card.get("name", "").strip()
        if not name:
            continue
        key = name.lower()
        try:
            price = float(card.get("prices", {}).get("usd") or 0.0)
        except (TypeError, ValueError):
            price = 0.0

        if key not in cheapest or price < cheapest[key][1]:
            cheapest[key] = (card, price)

    unique_cards = [doc for doc, _ in cheapest.values()]
    logger.info(
        "Deduplicated to %d unique card names (cheapest printing kept)",
        len(unique_cards),
    )
    return unique_cards

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
#Extracts numeric features from the raw MongoDB document for each card. Handles missing or malformed data gracefully by defaulting to 0.
def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

# Extracts numeric features from the raw MongoDB document for each card. Handles missing or malformed data gracefully by defaulting to 0.
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
# Initializes the DeckScorer by building a PCA-based scoring lookup from the card pool.
    def __init__(self, cards: list[dict]):
        if not cards:
            raise ValueError("Card list is empty.")

        self._build_lookup(cards)
        logger.info(
            "DeckScorer ready — %d unique card names", len(self.score_lookup)
        )
# Builds the card name -> score lookup using PCA on the card features.
    def _build_lookup(self, cards: list[dict]) -> None:
        """
        Derive feature weights from the card pool using PCA, then score
        every card as its projection onto PC1 (the direction of maximum
        variance across all cards).

        Why PC1?
        PC1 captures the single axis along which cards differ the most.
        In a card pool where price, rarity, power, and toughness all tend
        to move together for strong cards, PC1 naturally points in the
        direction of overall card quality — without any hand-tuned numbers.

        The loadings (eigenvector coefficients) become the weights.
        A negative loading on CMC means high-CMC cards score lower, which
        matches the intuition that cheaper-to-cast cards are more flexible.
        If PC1 ends up pointing the wrong way (i.e. expensive cards score
        low), we flip the sign so that higher score always means stronger.
        """
        # Build normalised feature matrix
        matrix = np.array([
            [extract_features(doc)[f] for f in FEATURE_NAMES]
            for doc in cards
        ], dtype=float)

        scaler = StandardScaler()
        normed = scaler.fit_transform(matrix)

        # Fit PCA and extract PC1 loadings as weights
        pca = PCA(n_components=len(FEATURE_NAMES))
        pca.fit(normed)
        weights = pca.components_[0]   # shape: (n_features,)

        # Convention: ensure price_usd loading is positive so that
        # more expensive cards score higher (flip entire vector if not)
        price_idx = FEATURE_NAMES.index("price_usd")
        if weights[price_idx] < 0:
            weights = -weights
            logger.debug("PC1 sign flipped so that price_usd loading is positive")

        logger.info(
            "PCA weights (PC1 explains %.1f%% of variance): %s",
            pca.explained_variance_ratio_[0] * 100,
            {f: round(float(w), 4) for f, w in zip(FEATURE_NAMES, weights)},
        )

        # Card score = projection onto PC1
        raw_scores = normed @ weights

        # Build name -> score lookup (one entry per unique card name)
        lookup: dict[str, float] = {}
        for doc, score in zip(cards, raw_scores):
            name = doc.get("name", "").strip().lower()
            if name:
                lookup[name] = float(score)

        self.score_lookup: dict[str, float] = lookup

        # Store for external use (plot, logging)
        self._scaler        = scaler
        self._pca           = pca
        self._weights       = weights
        self._feature_names = FEATURE_NAMES
        self._explained_variance = pca.explained_variance_ratio_
# Factory method to create a DeckScorer instance from MongoDB data.
    @classmethod
    def from_mongo(cls) -> "DeckScorer":
        client, collection = get_collection()
        try:
            cards = load_cards(collection)
        finally:
            client.close()
            logger.debug("MongoDB connection closed.")
        return cls(cards)
# Internal method to look up a card's score by name, with logging for missing cards.
    def _lookup(self, card_name: str) -> Optional[float]:
        score = self.score_lookup.get(card_name.strip().lower())
        if score is None:
            logger.warning("Card not found: '%s' — skipping", card_name)
        return score
# Compute aggregate stats for a decklist and return them in a dict.
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
# Map score difference to (0, 1) probability using a sigmoid function.
    def _sigmoid(self, x: float) -> float:
        """Map score difference to (0, 1) probability."""
        return 1.0 / (1.0 + np.exp(-SIGMOID_SCALE * x))
# Compare two decklists and return win probabilities + explanations.
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
# Helper function to compute average feature value across a decklist, with logging for missing cards.
def _avg_feature(decklist: list[str], feature: str, scorer: DeckScorer) -> float:
    """Compute the average raw (un-normalised) feature value across a decklist."""
    vals = []
    for name in decklist:
        key = name.strip().lower()
        # We need the raw feature — re-extract from a dummy lookup isn't ideal,
        # so we store nothing here. Instead we just use the score_lookup delta
        # as a proxy. For richer breakdowns, extend DeckScorer to cache raw features.
    return float(np.mean(vals)) if vals else 0.0

# Build a list of plain-English factors explaining the prediction, based on the deck stats and scorer insights.
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
# Two-panel chart showing feature weights and explained variance from PCA.
def plot_feature_weights(
    scorer: "DeckScorer",
    output_path: str = "feature_weights.png",
) -> None:
    """
    Two-panel chart showing:
      Left  — PC1 loadings (the weights used to score each card), as a
              horizontal bar chart. Positive = feature raises score,
              negative = feature lowers score.
      Right — Explained variance ratio across all principal components,
              showing how much of the card-pool variance each PC captures.
              The bar for PC1 is highlighted since that is the one we use.
    """
    weights   = scorer._weights
    feat_names = scorer._feature_names
    ev_ratio  = scorer._explained_variance

    # Friendly display names
    label_map = {
        "cmc":            "CMC",
        "price_usd":      "Price (USD)",
        "price_usd_foil": "Foil price (USD)",
        "power":          "Power",
        "toughness":      "Toughness",
        "rarity_score":   "Rarity",
    }
    labels = [label_map.get(f, f) for f in feat_names]

    # Sort loadings for the bar chart (largest absolute value at top)
    order   = np.argsort(np.abs(weights))
    s_labels  = [labels[i] for i in order]
    s_weights = weights[order]
    bar_colors = ["#378ADD" if w >= 0 else "#D85A30" for w in s_weights]

    fig, (ax_load, ax_var) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: PC1 loadings ──────────────────────────────────────────
    bars = ax_load.barh(
        s_labels, s_weights,
        color=bar_colors, edgecolor="white", linewidth=0.4, height=0.6,
    )
    for bar, val in zip(bars, s_weights):
        offset = 0.005 if val >= 0 else -0.005
        ha     = "left" if val >= 0 else "right"
        ax_load.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha=ha, fontsize=8.5, color="#444441",
        )

    ax_load.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax_load.set_xlabel("PC1 loading (weight)", fontsize=10)
    ax_load.set_title("Feature weights from PCA", fontsize=11, fontweight="medium", pad=10)

    pos_patch = mpatches.Patch(color="#378ADD", label="Raises card score")
    neg_patch = mpatches.Patch(color="#D85A30", label="Lowers card score")
    ax_load.legend(handles=[pos_patch, neg_patch], fontsize=8.5,
                   framealpha=0.9, edgecolor="#D3D1C7")

    # ── Right panel: explained variance ──────────────────────────────────
    n_components = len(ev_ratio)
    pc_labels    = [f"PC{i+1}" for i in range(n_components)]
    ev_colors    = ["#7F77DD" if i == 0 else "#D3D1C7" for i in range(n_components)]

    ax_var.bar(pc_labels, ev_ratio * 100, color=ev_colors,
               edgecolor="white", linewidth=0.4)
    for i, v in enumerate(ev_ratio * 100):
        ax_var.text(i, v + 0.4, f"{v:.1f}%", ha="center",
                    fontsize=8, color="#444441")

    ax_var.set_ylabel("Explained variance (%)", fontsize=10)
    ax_var.set_title("Variance explained by each PC", fontsize=11, fontweight="medium", pad=10)
    ax_var.set_ylim(0, max(ev_ratio * 100) * 1.15)

    pc1_patch = mpatches.Patch(color="#7F77DD", label="PC1 — used for scoring")
    ax_var.legend(handles=[pc1_patch], fontsize=8.5,
                  framealpha=0.9, edgecolor="#D3D1C7")

    # ── Shared styling ────────────────────────────────────────────────────
    for ax in (ax_load, ax_var):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D3D1C7")
        ax.spines["bottom"].set_color("#D3D1C7")
        ax.tick_params(colors="#5F5E5A")
        ax.set_facecolor("#FAFAFA")

    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=2.0)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Feature weights plot saved -> %s", output_path)
    except Exception as exc:
        logger.error("Failed to save feature weights plot: %s", exc)
    finally:
        plt.close(fig)

# Horizontal bar chart of every unique card across both decklists, coloured by deck membership and ordered by score.
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

   # These are sample decklists of Dimir Excrutiator and Mono Green Landfall. Both are in the standard meta.
   # Swap them out with other decklists!
    deck_a = [
       # Creatures (12)
       "Superior Spider-Man", "Superior Spider-Man", "Superior Spider-Man", "Superior Spider-Man",
       "Deceit", "Deceit", "Deceit", "Deceit",
       "Doomsday Excruciator", "Doomsday Excruciator", "Doomsday Excruciator", "Doomsday Excruciator",


       # Spells (22)
       "Duress", "Duress", "Duress",
       "Insatiable Avarice", "Insatiable Avarice",
       "Requiting Hex", "Requiting Hex", "Requiting Hex", "Requiting Hex",
       "Bitter Triumph", "Bitter Triumph", "Bitter Triumph", "Bitter Triumph",
       "Day of Black Sun", "Day of Black Sun", "Day of Black Sun",
       "Stock Up", "Stock Up", "Stock Up",
       "Winternight Stories", "Winternight Stories",
       "Deadly Cover-Up",


       # Lands (26)
       "Cavern of Souls", "Cavern of Souls",
       "Gloomlake Verge", "Gloomlake Verge", "Gloomlake Verge", "Gloomlake Verge",
       "Restless Reef", "Restless Reef", "Restless Reef", "Restless Reef",
       "Swamp", "Swamp", "Swamp", "Swamp", "Swamp",
       "Swamp", "Swamp", "Swamp", "Swamp", "Swamp",
       "Undercity Sewers", "Undercity Sewers",
       "Watery Grave", "Watery Grave", "Watery Grave", "Watery Grave",
   ]


    deck_b = [
       # Creatures (24)
       "Llanowar Elves", "Llanowar Elves", "Llanowar Elves", "Llanowar Elves",
       "Sazh's Chocobo", "Sazh's Chocobo", "Sazh's Chocobo", "Sazh's Chocobo",
       "Badgermole Cub", "Badgermole Cub", "Badgermole Cub", "Badgermole Cub",
       "Mossborn Hydra", "Mossborn Hydra", "Mossborn Hydra",
       "Surrak, Elusive Hunter",
       "Icetill Explorer", "Icetill Explorer", "Icetill Explorer", "Icetill Explorer",
       "Mightform Harmonizer", "Mightform Harmonizer", "Mightform Harmonizer", "Mightform Harmonizer",


       # Spells (4)
       "Royal Treatment", "Royal Treatment", "Royal Treatment", "Royal Treatment",


       # Enchantments (6)
       "Meltstrider's Resolve", "Meltstrider's Resolve",
       "Earthbender Ascension", "Earthbender Ascension", "Earthbender Ascension", "Earthbender Ascension",


       # Lands (26)
       "Ba Sing Se", "Ba Sing Se", "Ba Sing Se",
       "Escape Tunnel", "Escape Tunnel", "Escape Tunnel", "Escape Tunnel",
       "Fabled Passage", "Fabled Passage", "Fabled Passage", "Fabled Passage",
       "Forest", "Forest", "Forest", "Forest", "Forest", "Forest", "Forest",
       "Forest", "Forest", "Forest", "Forest", "Forest", "Forest",
       "Promising Vein", "Promising Vein",
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

    plot_feature_weights(scorer, output_path="feature_weights.png")

    plot_deck_comparison(
        deck_a, deck_b, scorer,
        deck_a_label="Dimir Excrutiator (Deck A)",
        deck_b_label="Mono Green Landfall (Deck B)",
        output_path="deck_comparison.png",
    )

#visualization rational

#I opted to visualize the PCA-derived feature weights and the deck comparison in two separate plots to keep each one clear and focused. 
# The first plot shows the feature weights as a horizontal bar chart, which makes it easy to see which features have the biggest impact on card scores and whether they raise or lower the score. 
# The second plot compares the two decks by showing every unique card across both decks, colored by which deck(s) they belong to, and ordered by their PCA score.
#  This allows you to visually identify which high-scoring cards are in each deck and how they contribute to the overall prediction. 
# I chose horizontal bars for readability, especially since card names can be long (i.e. approach of the second sun).
# Decks are defined by color, in this case Dimir Excrutiatoir is blue and mono green landfall is red. These two visualizations allow users to both compare cards between each other across decks,
# as well as understand the underlying features that are drving the score of the card.