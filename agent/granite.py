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

# Phase 3 — v3.1 (cache-bust: all narrative functions present)
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

def granite_health_check() -> tuple[bool, str]:
    """Quick ping to verify IBM Watsonx is reachable. Returns (ok, message)."""
    import time
    try:
        api_key, project_id, watsonx_url = _get_credentials()
        if not api_key or not project_id:
            return False, "IAM token unavailable"
        token = _get_iam_token(api_key)
        if not token:
            return False, "IAM token unavailable"
        start = time.time()
        resp = requests.post(
            f"{watsonx_url}/ml/v1/text/chat?version=2024-05-01",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "model_id": "ibm/granite-4-h-small",
                "project_id": project_id,
                "messages": [{"role": "user", "content": "ok"}],
                "parameters": {"max_new_tokens": 3},
            },
            timeout=5,
        )
        latency_ms = int((time.time() - start) * 1000)
        if resp.status_code == 200:
            return True, f"{latency_ms}ms"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)[:40]


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


def track_intel_brief(circuit_data: dict, sector_dominance: dict, mode: str = "fan") -> str:
    """
    Summarize track characteristics and sector advantages for the Track Intel tab.

    circuit_data: from CIRCUIT_PROFILES dict
    sector_dominance: {driver: {"sector1": delta_ms, "sector2": delta_ms, "sector3": delta_ms}}
    """
    track_char = circuit_data.get("track_character", "")
    drs_zones = circuit_data.get("drs_zones", 2)
    overtaking_diff = circuit_data.get("overtaking_difficulty", 50)
    agg_sectors = circuit_data.get("aggression_sectors", [1])

    # Find top sector performers
    sector_leaders = {}
    for sector_key in ["sector1", "sector2", "sector3"]:
        best_driver = None
        best_delta = float("inf")
        for drv, deltas in sector_dominance.items():
            d = deltas.get(sector_key, 0)
            if d < best_delta:
                best_delta = d
                best_driver = drv
        if best_driver:
            sector_leaders[sector_key] = f"{best_driver} ({best_delta:+.0f}ms)"

    leaders_str = ", ".join(f"S{i+1}: {v}" for i, (_, v) in enumerate(sector_leaders.items()))

    prompt = (
        f"Circuit: {track_char}\n"
        f"DRS zones: {drs_zones} | Overtaking difficulty: {overtaking_diff}/100\n"
        f"Aggression pays in: Sector {', '.join(str(s) for s in agg_sectors)}\n"
        f"Sector pace leaders: {leaders_str or 'data unavailable'}\n\n"
        f"Summarize in 2-3 sentences: what does this circuit demand from drivers, "
        f"and which drivers have an advantage based on their sector times?"
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    # Fallback
    diff_word = "easy" if overtaking_diff < 40 else "difficult"
    if mode == "fan":
        return (
            f"This circuit makes overtaking {diff_word} — {drs_zones} DRS zone(s) "
            f"create the main passing opportunities. "
            f"Sector {agg_sectors[0]} aggression is key to gaining positions."
        )
    else:
        return (
            f"Circuit overtaking_difficulty={overtaking_diff}/100 | DRS zones={drs_zones} | "
            f"Aggression sector(s): {agg_sectors}. {leaders_str}."
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


def pitwall_chat(question: str, lap_context: dict, driver: str, mode: str) -> str:
    """Answer a free-text engineer question about the current lap situation."""
    ctx_lines = []
    for k, v in lap_context.items():
        if v is not None and str(v) != "nan":
            ctx_lines.append(f"  {k}: {v}")
    context_str = "\n".join(ctx_lines)

    user_prompt = (
        f"Driver: {driver}\n"
        f"Current lap telemetry:\n{context_str}\n\n"
        f"Question: {question}"
    )
    system_prompt = ENGINEER_SYSTEM_PROMPT if mode == "engineer" else FAN_SYSTEM_PROMPT
    result = _call_granite(user_prompt, system_prompt)
    if result:
        return result
    return f"Analysis unavailable — review tyre age, gap ahead, and pit pressure for {driver} manually."


def error_summary(error_log: list, driver: str, mode: str = "fan") -> str:
    """
    Narrative summary of a driver's incident log for the Pilot Error section.

    error_log: list of dicts with keys lap, error_type, severity, description
    """
    if not error_log:
        if mode == "fan":
            return f"{driver} had a clean race — no notable errors or strategic missteps detected."
        else:
            return f"{driver}: 0 incidents detected. Pace and strategy nominal."

    total = len(error_log)
    types = {}
    for e in error_log:
        t = e.get("error_type", "UNKNOWN")
        types[t] = types.get(t, 0) + 1

    type_str = ", ".join(f"{v}x {k}" for k, v in types.items())
    worst = max(error_log, key=lambda e: e.get("signal_value", 0))

    prompt = (
        f"Driver: {driver}\n"
        f"Total incidents: {total}\n"
        f"Breakdown: {type_str}\n"
        f"Worst incident: Lap {worst.get('lap')} — {worst.get('description')}\n\n"
        f"Write 2-3 sentences summarizing this driver's errors and what they cost them."
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    if mode == "fan":
        return (
            f"{driver} made {total} mistakes this race: {type_str}. "
            f"The worst came on lap {worst.get('lap')}: {worst.get('description')}."
        )
    else:
        return f"{driver} | {total} incidents | {type_str} | Worst: L{worst.get('lap')} {worst.get('error_type')} signal={worst.get('signal_value'):.2f}"


def race_narrative(
    shifts: list,
    flag_periods: list,
    final_positions: dict,
    race_name: str,
    mode: str = "fan",
) -> str:
    """
    Full race narrative summarizing key moments, SC periods, and winner.

    shifts: top momentum shifts [{driver, lap, direction, magnitude}]
    flag_periods: [{flag, lap_start, lap_end}]
    final_positions: {driver: position}
    """
    winner = min(final_positions.items(), key=lambda x: x[1])[0] if final_positions else "Unknown"
    top_shifts = sorted(shifts, key=lambda s: s.get("magnitude", 0), reverse=True)[:5]
    shifts_str = "; ".join(
        f"Lap {s.get('lap')}: {s.get('driver')} {s.get('direction')} {s.get('magnitude', 0):.0f}pts"
        for s in top_shifts
    )
    sc_str = ", ".join(
        f"Laps {p['lap_start']}-{p['lap_end']} ({p['flag'].replace('_', ' ')})"
        for p in flag_periods if p.get("flag") in ("SAFETY_CAR", "VIRTUAL_SAFETY_CAR")
    ) or "None"
    positions_str = ", ".join(f"{d} P{p}" for d, p in sorted(final_positions.items(), key=lambda x: x[1])[:5])

    prompt = (
        f"Race: {race_name}\n"
        f"Winner: {winner}\n"
        f"Final order (top 5): {positions_str}\n"
        f"Safety car / VSC periods: {sc_str}\n"
        f"Top momentum shifts: {shifts_str}\n\n"
        f"Write a 4-6 sentence race narrative covering the key story, pivotal moments, and why {winner} won."
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    sc_note = f" A safety car at {sc_str} reshuffled the order." if sc_str != "None" else ""
    if mode == "fan":
        return (
            f"{winner} took the victory in {race_name}.{sc_note} "
            f"The race hinged on momentum swings at {', '.join('lap ' + str(s.get('lap')) for s in top_shifts[:3])}. "
            f"It was a race decided by strategy, nerve, and tyre management."
        )
    else:
        return (
            f"{race_name} | Winner: {winner} | SC: {sc_str} | "
            f"Key shifts: {shifts_str} | Final: {positions_str}"
        )


def driver_of_race(driver: str, stats: dict, mode: str = "fan") -> str:
    """
    Highlight the standout performer of the race.

    stats: {positions_gained, momentum_gain, error_count, top_trait, top_trait_val}
    """
    gained = stats.get("positions_gained", 0)
    momentum = stats.get("momentum_gain", 0)
    errors = stats.get("error_count", 0)
    top_trait = stats.get("top_trait", "aggression_level")
    top_val = stats.get("top_trait_val", 70)

    prompt = (
        f"Driver of the Race: {driver}\n"
        f"Positions gained from grid: {gained}\n"
        f"Net momentum gain: {momentum:.0f} pts\n"
        f"Incidents: {errors}\n"
        f"Dominant behavioral trait: {top_trait} ({top_val:.0f}/100)\n\n"
        f"Write 2-3 sentences explaining why {driver} was the standout performer this race."
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    if mode == "fan":
        return (
            f"{driver} was the driver of the race — gaining {gained} positions and building "
            f"{momentum:.0f} momentum points while making just {errors} mistake(s). "
            f"Their {top_trait.replace('_', ' ')} drove every key moment."
        )
    else:
        return (
            f"Driver of Race: {driver} | +{gained} pos | momentum_gain={momentum:.0f} | "
            f"incidents={errors} | dominant_trait={top_trait}={top_val:.0f}"
        )


def h2h_narrative(
    driver_a: str,
    driver_b: str,
    stats_a: dict,
    stats_b: dict,
    mode: str = "fan",
) -> str:
    """
    Head-to-head narrative comparing two drivers' season stats.

    stats: dict with keys matching compute_season_stats columns
    """
    def fmt(s: dict) -> str:
        return (
            f"avg_finish={s.get('avg_finish', '?')}, "
            f"avg_positions_gained={s.get('avg_positions_gained', '?')}, "
            f"tyre_preservation={s.get('tyre_preservation', '?')}, "
            f"aggression_level={s.get('aggression_level', '?')}, "
            f"pressure_consistency={s.get('pressure_consistency', '?')}"
        )

    prompt = (
        f"Head-to-head season comparison:\n"
        f"{driver_a}: {fmt(stats_a)}\n"
        f"{driver_b}: {fmt(stats_b)}\n\n"
        f"Write 2-3 sentences comparing these two drivers — who has the edge overall and where does each driver win?"
    )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    a_finish = stats_a.get("avg_finish", 10)
    b_finish = stats_b.get("avg_finish", 10)
    edge = driver_a if (a_finish or 20) < (b_finish or 20) else driver_b
    other = driver_b if edge == driver_a else driver_a
    if mode == "fan":
        return (
            f"Over the season, {edge} has the upper hand in race results. "
            f"But {other} fights back in specific areas — the data tells a story of two very different styles."
        )
    else:
        return (
            f"H2H: {driver_a} avg_finish={a_finish} vs {driver_b} avg_finish={b_finish}. "
            f"Edge: {edge}. Full stats above."
        )


def strategy_recommendation(
    scenarios: list,
    driver: str,
    current_lap: int,
    mode: str = "fan",
) -> str:
    """Narrative explaining the recommended pit strategy scenario.

    scenarios: output of compare_pit_scenarios() — list of scenario dicts each with
               keys: label, pit_lap, projected_position, position_delta, recommendation.
    """
    if not scenarios:
        if mode == "fan":
            return (f"The data couldn't project a clear strategy for {driver} right now "
                    "— the picture should sharpen in a lap or two.")
        return f"{driver}: insufficient scenario data at lap {current_lap} — projection unavailable."

    recommended = next((s for s in scenarios if s.get("recommendation")), scenarios[0])
    label    = recommended["label"]
    proj_pos = recommended["projected_position"]
    delta    = recommended["position_delta"]
    pit_lap  = recommended.get("pit_lap")

    scenario_summary = " | ".join(
        f"{s['label']}→P{s['projected_position']}(Δ{s['position_delta']:+d})"
        for s in scenarios
    )

    if mode == "fan":
        prompt = (
            f"Driver: {driver} | Current lap: {current_lap}\n"
            f"Best strategy: {label}"
            + (f" (pit at lap {pit_lap})" if pit_lap else "") + "\n"
            f"Projected position: P{proj_pos} | Position gain: {delta:+d}\n"
            f"All scenarios: {scenario_summary}\n\n"
            "Write 2-3 exciting sentences explaining the recommended strategy in plain English. "
            "Refer to tyres, track position, and what this means for the race."
        )
    else:
        prompt = (
            f"Driver: {driver} | Lap: {current_lap}\n"
            f"Scenario analysis: {scenario_summary}\n"
            f"Recommended: {label}"
            + (f" | pit_lap={pit_lap}" if pit_lap else "") + "\n"
            f"Projected: P{proj_pos} | Δpos={delta:+d}\n\n"
            "Write 2-3 sentences of engineer-mode analysis: lead with the dominant signal, "
            "state the recommendation and projected outcome numerically."
        )

    result = _call_granite(prompt, _system_prompt(mode))
    if result:
        return result

    # Fallback (Granite unavailable)
    if mode == "fan":
        if delta > 0:
            positions_word = f"{abs(delta)} position{'s' if abs(delta) != 1 else ''}"
            return (f"The data says {label.lower()} is the move — {driver} should gain "
                    f"{positions_word}, projecting to P{proj_pos} over the next 15 laps.")
        elif delta == 0:
            return (f"{label} keeps {driver} at P{proj_pos} "
                    "— no scenario offers a clear advantage right now.")
        else:
            return (f"All scenarios look difficult for {driver} here. "
                    f"{label} is the least costly option, projecting P{proj_pos}.")
    else:
        return (
            f"Scenario analysis: {scenario_summary}. "
            f"Recommendation: {label}"
            + (f" → pit_lap={pit_lap}" if pit_lap else "")
            + f" | projected P{proj_pos} (Δ{delta:+d})."
        )
