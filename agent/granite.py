"""IBM Granite agent — dual-voice (fan / engineer) narrative wrapper.

Uses IBM Watsonx AI REST API directly (no SDK required — just requests).

Credentials loaded from Streamlit secrets or environment variables:
    WATSONX_API_KEY      — IBM Cloud API key
    WATSONX_PROJECT_ID   — watsonx.ai project ID
    WATSONX_URL          — region URL (default: https://us-south.ml.cloud.ibm.com)

All public functions fall back to template text if Granite is unavailable,
so the app never crashes due to API issues.
"""

import os
import time
import hashlib
import logging
import requests

log = logging.getLogger(__name__)

GRANITE_MODEL = "ibm/granite-4-h-small"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
CHAT_PATH = "/ml/v1/text/chat?version=2023-05-29"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

FAN_SYSTEM_PROMPT = """You are PitMind's race commentator. You explain Formula 1 strategy in vivid,
plain English for passionate fans who love the sport but don't know telemetry jargon.

Rules:
- Never use abbreviations like SHAP, ELU, or technical model terms.
- Use human drama and storytelling: drivers, tyres, battles, pressure.
- Keep responses to 2-3 short sentences maximum.
- Be exciting but accurate to the data you receive.
- Use driver surnames (Verstappen, Hamilton, Leclerc — never "VER", "HAM", "LEC").
- Refer to tyres by colour: Soft (red), Medium (yellow), Hard (white).

Example voice: "Verstappen's tyres were completely gone — every lap was costing him half a second
to Hamilton behind. The pit wall had no choice but to call him in."""

ENGINEER_SYSTEM_PROMPT = """You are PitMind's race engineer AI. You deliver F1 strategy analysis
in precise telemetry language for engineers and strategists.

Rules:
- Lead with the dominant SHAP feature or signal driving each prediction.
- Use driver codes (VER, HAM, LEC) and compound codes (SOFT, MEDIUM, HARD).
- Include numerical values from the data in your response.
- Keep responses to 2-3 sentences maximum.
- No storytelling — just signal, interpretation, and implication.

Example voice: "SHAP: tyre_age coefficient 0.34 dominant. Pace delta +0.4s/lap (lap 38-42).
Undercut window closes in 2 laps — pit probability 0.82."""


# ---------------------------------------------------------------------------
# Credentials + IAM token management
# ---------------------------------------------------------------------------

_iam_token: str | None = None
_iam_token_expiry: float = 0.0


def _get_credentials():
    """Load IBM watsonx credentials from Streamlit secrets or env vars."""
    try:
        import streamlit as st
        api_key = st.secrets.get("WATSONX_API_KEY", "")
        project_id = st.secrets.get("WATSONX_PROJECT_ID", "")
        url = st.secrets.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    except Exception:
        api_key = os.environ.get("WATSONX_API_KEY", "")
        project_id = os.environ.get("WATSONX_PROJECT_ID", "")
        url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    return api_key, project_id, url


def _get_iam_token(api_key: str) -> str | None:
    """Exchange IBM Cloud API key for a Bearer token (cached, refreshed before expiry)."""
    global _iam_token, _iam_token_expiry

    if _iam_token and time.time() < _iam_token_expiry - 60:
        return _iam_token

    try:
        resp = requests.post(
            IAM_TOKEN_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _iam_token = data["access_token"]
        _iam_token_expiry = time.time() + data.get("expires_in", 3600)
        return _iam_token
    except Exception as e:
        log.warning(f"IAM token fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Core API call (with in-memory cache to respect rate limits)
# ---------------------------------------------------------------------------

_response_cache: dict[str, str] = {}


def _call_granite(user_prompt: str, system_prompt: str) -> str | None:
    """Call IBM Granite REST API directly with requests. Returns text or None."""
    api_key, project_id, watsonx_url = _get_credentials()
    if not api_key or not project_id:
        return None

    cache_key = hashlib.md5((system_prompt + user_prompt).encode()).hexdigest()
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    token = _get_iam_token(api_key)
    if not token:
        return None

    payload = {
        "model_id": GRANITE_MODEL,
        "project_id": project_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "parameters": {
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }

    try:
        resp = requests.post(
            watsonx_url + CHAT_PATH,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        _response_cache[cache_key] = text
        return text
    except Exception as e:
        log.warning(f"Granite API call failed: {e}")
        return None


def _system_prompt(mode: str) -> str:
    return FAN_SYSTEM_PROMPT if mode == "fan" else ENGINEER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def annotate_shift(shift: dict, mode: str = "fan") -> str:
    """
    One-sentence annotation for a momentum shift on the chart.

    shift keys: lap, driver, team, direction, magnitude, momentum_before, momentum_after
    """
    driver = shift.get("driver", "UNK")
    lap = shift.get("lap", 0)
    direction = shift.get("direction", "")
    magnitude = shift.get("magnitude", 0)
    before = shift.get("momentum_before", 0)
    after = shift.get("momentum_after", 0)

    prompt = (
        f"Driver: {driver} | Lap: {lap} | Momentum shift: {direction} | "
        f"Magnitude: {magnitude:.1f} pts | Before: {before:.1f} | After: {after:.1f}\n\n"
        f"Write one sentence annotating this momentum shift on the race chart."
    )

    result = _call_granite(prompt, _system_prompt(mode))

    if result:
        return result

    # Fallback
    if mode == "fan":
        arrow = "surged" if direction == "up" else "stumbled"
        return f"{driver} {arrow} at lap {lap}, swinging {magnitude:.0f} momentum points."
    else:
        return f"Lap {lap} | {driver}: momentum {direction} +{magnitude:.1f}pt ({before:.1f} -> {after:.1f})"


def explain_driver(traits: dict, shap_vals: dict, driver: str, mode: str = "fan") -> str:
    """
    Behavioral narrative for the Driver Soul tab.

    traits: {trait_name: score_0_100, ...}
    shap_vals: {feature_name: shap_value, ...} (top-3 from explainer)
    """
    top_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
    top_trait_str = ", ".join(f"{k}={v:.0f}" for k, v in top_traits)
    shap_str = ", ".join(f"{k}:{v:+.3f}" for k, v in shap_vals.items()) if shap_vals else "unavailable"

    prompt = (
        f"Driver: {driver}\n"
        f"Top behavioral traits: {top_trait_str}\n"
        f"SHAP dominant features: {shap_str}\n\n"
        f"Write 2-3 sentences describing this driver's behavioral profile."
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    # Fallback
    trait_names = [k.replace("_", " ") for k, _ in top_traits]
    if mode == "fan":
        return (
            f"{driver} is defined by {trait_names[0]} and {trait_names[1]}. "
            f"These instincts shape every strategic decision they make on track."
        )
    else:
        return (
            f"{driver} profile: dominant signals {top_trait_str}. "
            f"SHAP drivers: {shap_str}."
        )


def pitwall_brief(lap_data: dict, mode: str = "fan") -> str:
    """
    The information card shown to the user before they make a pit decision.

    lap_data keys from per-lap schema: driver, lap, tyre_compound, tyre_age,
    gap_ahead, gap_behind, pace_delta, pit_window_pressure, weather_track_temp
    """
    driver = lap_data.get("driver", "UNK")
    lap = lap_data.get("lap", 0)
    compound = lap_data.get("tyre_compound", "?")
    age = lap_data.get("tyre_age", 0)
    gap_ahead = lap_data.get("gap_ahead", 0)
    gap_behind = lap_data.get("gap_behind", 0)
    pace_delta = lap_data.get("pace_delta", 0)
    pressure = lap_data.get("pit_window_pressure", 50)
    temp = lap_data.get("weather_track_temp", 30)

    prompt = (
        f"Pit decision moment — Lap {lap}, Driver: {driver}\n"
        f"Tyres: {compound} age {age} laps | Pace delta: +{pace_delta:.2f}s/lap\n"
        f"Gap ahead: {gap_ahead:.2f}s | Gap behind: {gap_behind:.2f}s\n"
        f"Pit window pressure: {pressure:.0f}/100 | Track temp: {temp:.1f}°C\n\n"
        f"Brief the driver/user: what does this situation look like right now? Should they pit?"
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    # Fallback
    urgency = "urgent" if pressure > 70 else "manageable"
    if mode == "fan":
        return (
            f"{driver}'s tyres have {age} laps on them and the pace is dropping. "
            f"The gap behind is closing. This call feels {urgency}."
        )
    else:
        return (
            f"{driver} | {compound} L{age} | Δpace +{pace_delta:.2f}s | "
            f"gap_behind {gap_behind:.2f}s | pit_pressure {pressure:.0f}/100 — {urgency}."
        )


def reveal_outcome(
    choice: str,
    actual_outcome: str,
    driver_soul: dict,
    driver: str,
    mode: str = "fan",
) -> str:
    """
    Post-decision reveal: what happened vs. what the user chose.

    choice: "pit" or "stay"
    actual_outcome: description of what actually happened
    driver_soul: {instinct prediction description}
    """
    instinct_prob = driver_soul.get("pit_prob", 0.5)
    instinct_action = "pit" if instinct_prob > 0.5 else "stay out"
    top_trait = max(driver_soul.get("traits", {}).items(), key=lambda x: x[1], default=("unknown", 0))

    prompt = (
        f"Driver: {driver}\n"
        f"User chose: {choice}\n"
        f"What actually happened: {actual_outcome}\n"
        f"Driver's instinct model: {instinct_prob:.0%} probability to {instinct_action}\n"
        f"Dominant behavioral trait: {top_trait[0]} ({top_trait[1]:.0f}/100)\n\n"
        f"Reveal the outcome and explain why the driver made their real choice, "
        f"referencing their behavioral profile."
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    # Fallback
    correct = choice == ("pit" if "pitted" in actual_outcome.lower() else "stay")
    verdict = "You called it right." if correct else "The team saw it differently."
    if mode == "fan":
        return (
            f"{verdict} {actual_outcome} "
            f"{driver}'s instinct model gave {instinct_prob:.0%} odds to {instinct_action} — "
            f"their {top_trait[0].replace('_', ' ')} profile explains the decision."
        )
    else:
        return (
            f"User: {choice} | Actual: {actual_outcome} | "
            f"Model pit_prob={instinct_prob:.3f} ({instinct_action}). "
            f"Dominant driver trait: {top_trait[0]}={top_trait[1]:.0f}"
        )
