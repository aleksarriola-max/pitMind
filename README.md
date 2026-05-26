# PitMind — AI-Powered F1 Strategy Intelligence

> Bridging the gap between the pit wall and the grandstand — powered by IBM Granite AI

[![IBM Granite](https://img.shields.io/badge/IBM-Granite%20AI-0043CE)](https://www.ibm.com/watsonx)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://python.org)

---

## The Problem

Formula 1 is the most data-rich sport on Earth — yet that intelligence is invisible to 99% of fans and inaccessible to smaller racing programs.

During a race, a driver's pit wall processes thousands of telemetry signals every second: tyre degradation curves, gap trajectories, sector delta comparisons, radio sentiment. Decisions made in 3 seconds — "box box" or "stay out" — can win or lose championships.

**Two problems, same root:**

1. **Fans** experience races as a spectacle without understanding the strategy unfolding in real time. They see position changes; they don't see why.
2. **Smaller teams** (junior formula, simulation leagues, educational programs) lack affordable tools to analyze driver behavior and race strategy with the depth of an F1 pit wall.

---

## Our Solution

**PitMind** is an AI-powered F1 analytics platform that gives everyone pit-wall intelligence — from passionate fans to aspiring race engineers.

The app has two modes toggled by a single switch:

| **Fan Mode** | **Engineer Mode** |
|---|---|
| Plain-language storytelling | Raw telemetry numbers |
| "Verstappen's tyres are cooked — the team wants to pit" | "MEDIUM · 31 laps · pace delta +0.42s/lap · pit pressure 84/100" |
| Urgency labels, qualitative gaps | SHAP feature importance, sector heatmaps, prediction trends |
| Granite narrates the story | Granite explains the data |

Both modes are powered by **IBM Granite AI** generating contextual narratives matched to the audience — same model, same data, radically different output.

---

## AI & Technical Approach

### IBM Granite Integration

- **Model:** `ibm/granite-4-h-small` via IBM Watsonx AI REST API
- **8 distinct AI functions**, each with separate Fan and Engineer system prompts:

| Function | What it does |
|---|---|
| `annotate_shift()` | Explains momentum inflection points in fan or engineer voice |
| `pitwall_brief()` | Pre-decision situation briefing for the selected lap |
| `reveal_outcome()` | Post-decision narrative — what the team actually did and why |
| `race_narrative()` | Full race story from momentum shifts + flag periods |
| `driver_of_race()` | Standout performer analysis |
| `explain_driver()` | Behavioral profile from 11-trait fingerprint + SHAP values |
| `h2h_narrative()` | Head-to-head season comparison |
| `error_summary()` | Pilot error and strategic misstep log |

**Dual-voice architecture:** Granite is not a chatbot wrapper — it is the interpretation engine. The same telemetry row produces a fan narrative ("Hamilton was under siege, his tyres giving up lap by lap") and an engineer brief ("pace_delta +0.38s/lap degrading, tyre_age coeff dominant SHAP feature — undercut window closes in 2 laps") from two hardcoded system prompts.

### Data Pipeline

```
FastF1 (official F1 library)    OpenF1 API (real-time)
     ↓ lap times, telemetry          ↓ intervals, race control
     ↓ sector splits, tyres          ↓ weather, flag periods
     ↓ team radio audio
     ↓
  Whisper transcription → TextBlob sentiment scoring
     ↓
  Merged per-lap DataFrame (30+ columns)
     ↓
  Momentum score + shift detection (ruptures PELT)
     ↓
  Driver Soul model (11 traits → UMAP → XGBoost → SHAP)
     ↓
  Pre-computed .parquet cache → sub-second Streamlit loads
```

**5 races, 2025 season:** Bahrain, Monaco, Silverstone, Monza, Abu Dhabi

### Machine Learning Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Driver Soul Model                      │
│                                                           │
│  11 Behavioral Traits (computed from telemetry):          │
│  overtake tendency · tyre preservation · aggression       │
│  restart aggression · pressure consistency · pit comply   │
│  late braking · dirty air tolerance · undercut risk ...   │
│                                                           │
│  UMAP → 2D fingerprint space                              │
│  XGBoost MultiOutputClassifier → 4 real-time predictions │
│    position_gain_prob · position_loss_prob                │
│    pit_prob · incident_risk                               │
│  SHAP TreeExplainer → per-lap feature importance          │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│               Momentum Score (0–100 per lap)              │
│  40% pace delta + 30% gap trajectory +                   │
│  20% radio sentiment + 10% pit window pressure            │
│  Shift detection: sliding-window peak comparison          │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│               Supporting Models                           │
│  Error Detection: 5 types — lockup, pace collapse,       │
│    late pit, SC loss, radio stress                        │
│  Race Forecast: 15-lap position projection from          │
│    pace delta + gap trajectory                            │
│  Sentiment Pipeline: Whisper → TextBlob per lap          │
└─────────────────────────────────────────────────────────┘
```

---

## Application Features (5 Tabs)

### 🌊 Momentum Map
Lap-by-lap momentum score for all drivers with safety car / VSC / yellow flag overlays, momentum shift markers, and three IBM Granite AI narrative cards:
- **Explain This Race** — 4–6 sentence race story
- **Driver of the Race** — model-selected standout performer
- **Top 5 Key Moments** — highest-magnitude shifts with AI annotation

Engineer mode adds: shift magnitude table, per-shift signal causation breakdown (which telemetry signal triggered each shift).

### 🧬 Driver Soul
Each driver gets a behavioral fingerprint — an 11-dimension radar chart built purely from telemetry analysis. No subjective scoring; every trait is a derived signal from lap time variance, braking data, gap dynamics, and radio sentiment.

Live state panel shows trait scores at any selected lap. Engineer mode adds:
- SHAP feature importance bar chart
- Radio sentiment trend across the race
- Incident risk trend line
- Prediction probability trends (position gain/loss/pit across all laps)

### 🔧 Pit Wall Mirror
The interactive centrepiece. The user sees the same data the pit wall sees at a selected lap — tyre compound/age, gap ahead/behind, pace delta, pit window pressure, weather, safety car status.

**Fan mode:** qualitative cards ("tyres getting worn", "DRS range — under attack")
**Engineer mode:** raw numbers + pit strategy model recommendation ("optimal pit in 3 laps — pressure peaks at 84/100")

The user makes their call (**Pit Now** / **Stay Out**), then sees the outcome reveal: what the team actually did, how positions evolved over the next 5 laps, the Driver Soul model's pit probability at that moment, and IBM Granite's narrative of the decision. Engineer mode adds a pit window pressure overlay chart showing the full-race pressure curve with actual pit laps marked.

### 🏁 Track Intel
- Circuit profile (length, DRS zones, overtaking difficulty, track character)
- Interactive circuit layout with live driver position dots (positions estimated from gap/lap_time arc-length math)
- Sector dominance heatmap (who is fastest in S1/S2/S3 vs. field average)
- Grid → finish position analysis (scatter + season average positions gained)
- 15-lap battle forecast (position projection heatmap with DRS range highlighted)
- Aggression zone chart (where in the race do position changes happen?)
- Pilot error log with 5 error types (lockup, pace collapse, late pit, SC loss, radio stress)

### 📊 Driver Stats
Full season leaderboard across 9 metrics, head-to-head radar chart with IBM Granite narrative comparison, and season arc charts showing each driver's performance trajectory across the 5 races.

---

## Why This Matters

**For fans:** F1 broadcasts tell you what happened. PitMind tells you why — in language matched to your expertise level. A casual viewer and an ex-engineer can watch the same lap and each get the insight that's right for them.

**For the sport:** The dual-mode architecture demonstrates that the same AI-powered telemetry analysis that today only exists in team factories can be democratized. Junior series, sim racers, esports leagues, and fan communities can all access this level of strategic intelligence.

**For IBM Granite:** This is AI as an interpretation layer — not a replacement for data, but the translator between raw telemetry and human understanding. Granite's dual-voice capability shows how enterprise AI can serve radically different audiences from one model and one dataset.

---

## Project Structure

```
pitMind/
├── app/
│   ├── main.py              # Streamlit entry point + sidebar
│   ├── momentum_view.py     # Momentum Map tab
│   ├── driver_view.py       # Driver Soul tab
│   ├── pitwall_view.py      # Pit Wall Mirror tab
│   ├── track_view.py        # Track Intel tab
│   └── stats_view.py        # Driver Stats tab
├── agent/
│   └── granite.py           # IBM Granite integration (8 AI functions, 565 lines)
├── models/
│   ├── driver_soul.py       # 11-trait behavioral model + XGBoost + SHAP
│   ├── momentum.py          # Momentum score formula + shift detection
│   ├── sentiment.py         # Whisper transcription + TextBlob scoring
│   ├── error_detection.py   # 5-type pilot error classifier
│   ├── race_forecast.py     # 15-lap position projection
│   └── driver_stats.py      # Season-level aggregations + H2H
├── data/
│   ├── ingest.py            # FastF1 + OpenF1 unified pipeline (22 KB)
│   └── cache/               # Pre-computed .parquet + shift .json files
├── scripts/
│   └── build_data.py        # Offline build orchestrator (ingest → sentiment → momentum → soul)
├── requirements.txt
└── packages.txt             # Streamlit Cloud system deps (ffmpeg)
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/pitMind
cd pitMind

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Configure IBM Watsonx credentials
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
WATSONX_API_KEY = "your-api-key"
WATSONX_PROJECT_ID = "your-project-id"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
EOF

# Run the app (pre-computed data is already in data/cache/)
streamlit run app/main.py
```

The app loads from pre-computed `.parquet` cache files — no data rebuild required to run the demo.

### Rebuilding Race Data (optional)

```bash
python scripts/build_data.py bahrain_2025 --skip-whisper
```

This pulls fresh data from FastF1 + OpenF1, computes momentum scores and behavioral traits, and saves updated cache files.

---

## IBM Watsonx Setup

1. Create an [IBM Cloud account](https://cloud.ibm.com)
2. Create a **Watson Machine Learning** service (free lite tier)
3. Navigate to [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → Create a project
4. IAM → Manage → API keys → Create API key
5. Note your **project ID** (shown in the project URL)
6. Add credentials to `.streamlit/secrets.toml` (see above)

---

## Tech Stack

| Technology | Role |
|---|---|
| [IBM Watsonx AI](https://www.ibm.com/watsonx) — `granite-4-h-small` | AI narrative engine (8 functions, dual-voice) |
| [FastF1](https://github.com/theOehrly/Fast-F1) | Official F1 lap telemetry + team radio |
| [OpenF1 API](https://openf1.org) | Real-time intervals, race control, weather |
| [Streamlit](https://streamlit.io) | Application framework |
| [XGBoost](https://xgboost.ai) | Multi-target behavioral predictions |
| [SHAP](https://shap.readthedocs.io) | Per-lap feature importance |
| [UMAP](https://umap-learn.readthedocs.io) | Driver behavioral fingerprinting |
| [Whisper](https://github.com/openai/whisper) | Team radio audio transcription |
| [TextBlob](https://textblob.readthedocs.io) | Sentiment scoring |
| [Plotly](https://plotly.com) | Interactive charts |

---

*IBM x May Challenge 2026 · Built in 6 days*
