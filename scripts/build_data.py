"""Offline build pipeline — run once locally to produce all pre-computed data.

Usage:
    python scripts/build_data.py                  # build all 5 races
    python scripts/build_data.py bahrain_2023     # build one race
    python scripts/build_data.py --skip-whisper   # skip radio transcription

Pipeline order per race:
    1. ingest.py      → per-lap DataFrame (FastF1 + OpenF1)
    2. sentiment.py   → add radio_text + radio_sentiment
    3. momentum.py    → add momentum_score, save shifts JSON
    4. driver_soul.py → add 11 trait scores + 4 predictions (fit across all races)
    5. serialize      → data/cache/<slug>.parquet
"""

import sys
import os
import json
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingest import RACES, build_race_dataframe, save_race, load_race, CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_race(slug: str, skip_whisper: bool = False) -> None:
    log.info(f"=== Building {slug} ===")

    # Step 1: Ingest
    parquet_path = os.path.join(CACHE_DIR, f"{slug}.parquet")
    if os.path.exists(parquet_path):
        log.info(f"  Parquet exists, loading from cache: {parquet_path}")
        df = load_race(slug)
    else:
        log.info("  Step 1/4: Ingesting FastF1 + OpenF1 data...")
        df = build_race_dataframe(slug)
        save_race(df, slug)

    # Step 2: Sentiment (Whisper + TextBlob)
    if not skip_whisper:
        log.info("  Step 2/4: Running Whisper transcription + sentiment scoring...")
        try:
            from models.sentiment import add_sentiment
            race_info = RACES[slug]
            import fastf1
            session = fastf1.get_session(race_info["year"], race_info["round"], "R")
            session.load(messages=True)
            df = add_sentiment(df, session)
            save_race(df, slug)
        except ImportError:
            log.warning("  Whisper/textblob not installed yet — skipping sentiment step.")
        except Exception as e:
            log.warning(f"  Sentiment step failed: {e} — continuing without radio data.")
    else:
        log.info("  Step 2/4: Skipping sentiment (--skip-whisper flag set)")

    # Step 3: Momentum scores + shift detection
    log.info("  Step 3/4: Computing momentum scores and detecting shifts...")
    try:
        from models.momentum import add_momentum, detect_shifts, save_shifts
        df = add_momentum(df)
        shifts = detect_shifts(df)
        shifts_path = save_shifts(shifts, slug)
        log.info(f"  Detected {len(shifts)} momentum shifts → {shifts_path}")
        save_race(df, slug)
    except ImportError:
        log.warning("  ruptures not installed yet — skipping momentum step.")
    except Exception as e:
        log.warning(f"  Momentum step failed: {e}")

    log.info(f"  Race {slug} complete. Columns: {list(df.columns)}")


def fit_driver_soul_all_races(slugs: list) -> None:
    """Fit UMAP + XGBoost across all races, then add predictions to each parquet."""
    log.info("=== Step 4/4: Fitting Driver Soul model across all races ===")
    try:
        from models.driver_soul import fit_and_save, add_predictions_to_race
        import pandas as pd

        # Load all race DataFrames
        dfs = []
        for slug in slugs:
            try:
                df = load_race(slug)
                df["race_slug"] = slug
                dfs.append(df)
            except FileNotFoundError:
                log.warning(f"  Skipping {slug} — parquet not found")

        if not dfs:
            log.error("  No race data found. Run ingestion first.")
            return

        all_data = pd.concat(dfs, ignore_index=True)

        # Fit models
        fit_and_save(all_data)

        # Add predictions back to each race's parquet
        for slug in slugs:
            try:
                df = load_race(slug)
                df = add_predictions_to_race(df)
                save_race(df, slug)
                log.info(f"  Driver Soul predictions added to {slug}")
            except FileNotFoundError:
                pass

    except ImportError:
        log.warning("  umap-learn / xgboost / shap not installed yet — skipping Driver Soul step.")
    except Exception as e:
        log.error(f"  Driver Soul fitting failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="PitMind offline build pipeline")
    parser.add_argument("races", nargs="*", help="Race slugs to build (default: all)")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip Whisper transcription")
    args = parser.parse_args()

    target_slugs = args.races if args.races else list(RACES.keys())
    invalid = [s for s in target_slugs if s not in RACES]
    if invalid:
        log.error(f"Unknown race slugs: {invalid}. Valid: {list(RACES.keys())}")
        sys.exit(1)

    log.info(f"Building {len(target_slugs)} race(s): {target_slugs}")

    # Steps 1-3 per race
    for slug in target_slugs:
        build_race(slug, skip_whisper=args.skip_whisper)

    # Step 4: Driver Soul (fit across all, requires all races to be ingested)
    fit_driver_soul_all_races(target_slugs)

    log.info("=== Build complete ===")


if __name__ == "__main__":
    main()
