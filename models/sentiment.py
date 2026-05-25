"""Radio transcript sentiment pipeline.

Steps:
    1. Extract team radio audio files from FastF1 session.
    2. Transcribe each clip with Whisper (base model).
    3. Score sentiment with TextBlob.
    4. Merge onto the nearest lap row in the DataFrame.
"""

import os
import logging
import tempfile
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DRIVERS = ["VER", "HAM", "LEC", "SAI", "ALO", "PER", "NOR", "RUS"]

# Lazy-loaded globals — imported only when needed so the app doesn't require torch
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        log.info("Loading Whisper base model (first call, may take ~30s)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def get_radio_clips(session) -> list[dict]:
    """
    Extract team radio clips from a FastF1 session.

    Returns list of {driver, lap_approx, audio_bytes} dicts.
    FastF1 provides radio as binary audio (MP3/AAC) via session.radio_messages.
    """
    clips = []
    try:
        radio = session.radio_messages
        if radio is None or len(radio) == 0:
            log.info("No radio messages in this session.")
            return clips

        # FastF1 radio schema: Driver, Time (session-relative), Message (bytes or path)
        # Filter to our tracked drivers
        radio = radio[radio["Driver"].isin(DRIVERS)] if "Driver" in radio.columns else radio

        # Map session time → approximate lap number using session laps
        laps_df = session.laps[["Driver", "LapNumber", "Time"]].dropna().copy()
        laps_df["session_sec"] = laps_df["Time"].dt.total_seconds()

        for _, row in radio.iterrows():
            driver = row.get("Driver", "UNK")
            radio_time = row.get("Time")
            audio = row.get("Recording")  # bytes or None

            if audio is None:
                continue

            radio_sec = radio_time.total_seconds() if hasattr(radio_time, "total_seconds") else 0

            # Approximate lap: nearest lap end time for this driver
            driver_laps = laps_df[laps_df["Driver"] == driver].sort_values("session_sec")
            if len(driver_laps) == 0:
                lap_approx = 0
            else:
                diff = (driver_laps["session_sec"] - radio_sec).abs()
                lap_approx = int(driver_laps.iloc[diff.argmin()]["LapNumber"])

            clips.append({
                "driver": driver,
                "lap_approx": lap_approx,
                "audio_bytes": audio,
                "session_sec": radio_sec,
            })

    except Exception as e:
        log.warning(f"Failed to extract radio clips: {e}")

    log.info(f"Extracted {len(clips)} radio clips")
    return clips


def transcribe_clips(clips: list[dict]) -> list[dict]:
    """Transcribe audio clips with Whisper base and add text field."""
    if not clips:
        return clips

    model = _get_whisper_model()
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, clip in enumerate(clips):
            try:
                # Write audio bytes to temp file
                audio_path = os.path.join(tmpdir, f"clip_{i}.mp3")
                with open(audio_path, "wb") as f:
                    if isinstance(clip["audio_bytes"], (bytes, bytearray)):
                        f.write(clip["audio_bytes"])
                    else:
                        log.debug(f"Clip {i} audio is not bytes: {type(clip['audio_bytes'])}")
                        results.append({**clip, "text": ""})
                        continue

                result = model.transcribe(audio_path, language="en", fp16=False)
                text = result.get("text", "").strip()
                results.append({**clip, "text": text})

                if i % 20 == 0:
                    log.info(f"  Transcribed {i+1}/{len(clips)} clips")

            except Exception as e:
                log.debug(f"Transcription failed for clip {i}: {e}")
                results.append({**clip, "text": ""})

    log.info(f"Transcription complete: {sum(1 for r in results if r['text'])} / {len(results)} clips have text")
    return results


def score_sentiment(clips_with_text: list[dict]) -> list[dict]:
    """Add sentiment score (-1..+1) using TextBlob polarity."""
    from textblob import TextBlob

    results = []
    for clip in clips_with_text:
        text = clip.get("text", "")
        if text:
            polarity = TextBlob(text).sentiment.polarity
        else:
            polarity = 0.0
        results.append({**clip, "sentiment": polarity})
    return results


def merge_onto_laps(df: pd.DataFrame, clips: list[dict]) -> pd.DataFrame:
    """
    Merge radio text + sentiment onto the per-lap DataFrame.
    Each clip is assigned to its driver's nearest lap.
    Multiple clips on the same lap are aggregated (average sentiment, joined text).
    """
    df = df.copy()

    if not clips:
        df["radio_text"] = df["radio_text"].fillna("")
        df["radio_sentiment"] = df["radio_sentiment"].fillna(0.0)
        return df

    clips_df = pd.DataFrame(clips)[["driver", "lap_approx", "text", "sentiment"]]
    clips_df = clips_df.rename(columns={"lap_approx": "lap"})

    # Aggregate multiple clips per (driver, lap)
    agg = clips_df.groupby(["driver", "lap"]).agg(
        radio_text=("text", lambda x: " | ".join(t for t in x if t)),
        radio_sentiment=("sentiment", "mean"),
    ).reset_index()

    # Merge onto main df (left join — most laps have no radio)
    df = df.merge(agg, on=["driver", "lap"], how="left", suffixes=("_old", ""))

    # Keep existing values if new ones are NaN
    if "radio_text_old" in df.columns:
        df["radio_text"] = df["radio_text"].fillna(df["radio_text_old"])
        df.drop(columns=["radio_text_old"], inplace=True)
    if "radio_sentiment_old" in df.columns:
        df["radio_sentiment"] = df["radio_sentiment"].fillna(df["radio_sentiment_old"])
        df.drop(columns=["radio_sentiment_old"], inplace=True)

    df["radio_text"] = df["radio_text"].fillna("")
    df["radio_sentiment"] = df["radio_sentiment"].fillna(0.0)

    return df


def add_sentiment(df: pd.DataFrame, session) -> pd.DataFrame:
    """Full pipeline: extract → transcribe → score → merge. Returns updated df."""
    log.info("Extracting radio clips from FastF1 session...")
    clips = get_radio_clips(session)

    if not clips:
        log.info("No radio clips found — radio columns will be empty.")
        return df

    log.info(f"Transcribing {len(clips)} clips with Whisper base...")
    clips = transcribe_clips(clips)

    log.info("Scoring sentiment with TextBlob...")
    clips = score_sentiment(clips)

    log.info("Merging radio data onto lap DataFrame...")
    df = merge_onto_laps(df, clips)

    radio_count = (df["radio_text"] != "").sum()
    log.info(f"Radio data merged: {radio_count} laps have radio text")
    return df
