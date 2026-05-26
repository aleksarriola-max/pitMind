# PitMind — Demo Video Script
**IBM x May Challenge 2026 · 3-Minute Walkthrough**

---

## Timing Overview

| Section | Time | Content |
|---------|------|---------|
| Hook | 0:00 – 0:18 | Problem statement — no intro |
| Problem | 0:18 – 0:40 | The data gap in F1 |
| AHA Moment | 0:40 – 1:25 | Fan vs Engineer toggle (centrepiece) |
| Driver Soul | 1:25 – 1:50 | Behavioral fingerprint + engineer depth |
| Momentum Map | 1:50 – 2:15 | IBM Granite live generation |
| Track Intel | 2:15 – 2:40 | Circuit map + battle forecast |
| Vision | 2:40 – 2:55 | Why this scales beyond F1 |
| Close | 2:55 – 3:00 | Sign-off + URL |

---

## Full Script

---

### [0:00 – 0:18] THE HOOK

> *Screen: black. Text appears line by line:*
>
> **"Abu Dhabi 2021. Lap 58 of 58. Red Bull pits Verstappen."**
>
> *(pause 2 seconds)*
>
> **"That decision won a World Championship."**
>
> *(pause 1 second)*
>
> **"98 million fans watched. Almost none of them knew why."**
>
> *Cut to the app.*

**Screen:** Black screen with white text only. No app visible yet.

---

### [0:18 – 0:40] THE PROBLEM

**Screen:** App open — Pit Wall Mirror tab, Fan Mode, VER, Bahrain 2025, lap 28.

> "Formula 1 is the most data-rich sport on Earth. Every lap, a pit wall processes thousands of telemetry signals — tyre degradation, gap trajectories, radio sentiment, sector deltas — to make decisions in under 3 seconds."

> "That intelligence is invisible to fans. And completely out of reach for smaller racing programs."

> "We built PitMind to fix that."

---

### [0:40 – 1:25] THE AHA MOMENT — Fan vs Engineer Toggle

**Screen:** Pit Wall Mirror, Fan Mode, VER, Bahrain 2025, lap 28.

> "This is Fan Mode. Verstappen, lap 28 — his tyres are 24 laps old."

*(Point at the fan status card)*

> "PitMind tells the fan: 'Tyres are very worn — team wants to pit.' Urgency: High. Gap to the car ahead: DRS range — under attack."

> "IBM Granite narrates the situation in plain language. The fan understands the tension without understanding the telemetry."

*(Click 'Pit Now')*

> "The fan makes the call. And then — we show them what actually happened."

*(Outcome reveal appears — hold for 2 seconds)*

> "Now — same lap. Same driver. Same data."

*(Toggle to Engineer Mode in the sidebar)*

> "Engineer Mode."

*(The screen transforms — 8-metric telemetry dashboard appears)*

> "The engineer sees: MEDIUM compound, 24 laps. Pace delta plus 0.38 seconds per lap and degrading. Gap ahead 0.9 seconds — inside DRS. Pit window pressure at 84 out of 100."

*(Point at the pit strategy optimizer)*

> "The model is recommending: pit now. Pressure peaks this lap."

*(Scroll to IBM Granite engineer narrative)*

> "IBM Granite — same model, same race data — switches voice entirely. This is the dual-voice system. One Granite model. Two system prompts. Two audiences."

---

### [1:25 – 1:50] DRIVER SOUL

**Screen:** Switch to Driver Soul tab, same driver and lap.

> "Every driver in our dataset has a behavioral fingerprint."

*(Point at the radar chart)*

> "Eleven traits — overtake tendency, tyre preservation, restart aggression, dirty air tolerance — computed purely from telemetry. No subjective scoring. The model reads the data the way a race engineer would."

*(Switch to Engineer Mode — scroll to show radio sentiment arc and incident risk trend)*

> "In Engineer Mode, you see the full picture: radio sentiment across the race, incident risk trend, and prediction probabilities evolving lap by lap."

*(Tap incident risk chart)*

> "This is when the model thought a crash was most likely. That's not a heatmap — that's XGBoost trained on behavioral traits with SHAP explainability."

---

### [1:50 – 2:15] MOMENTUM MAP — Live IBM Granite Generation

**Screen:** Switch to Momentum Map tab.

> "Every race is tracked as a momentum score — pace delta, gap trajectory, radio sentiment, pit window pressure — combined per lap, per driver."

*(Point at a shift marker on the chart)*

> "When the model detects a momentum shift — IBM Granite annotates it."

*(Click 'Generate race narrative' button)*

> "On demand, Granite generates the full race story."

*(Let the narrative load and display — hold on screen for 3–4 seconds)*

> "Fan mode gets a narrative. Engineer mode gets telemetry language. Same API call. Same race data."

*(Expand a shift causation panel)*

> "Engineer mode also shows exactly which signal triggered each shift — gap, pace, tyre age, radio sentiment — compared to the lap before."

---

### [2:15 – 2:40] TRACK INTEL

**Screen:** Switch to Track Intel tab.

> "Track Intel shows the circuit layout with live driver positions — estimated from gap and lap time data using arc-length path math."

*(Show the circuit map with colored driver dots)*

> "Sector dominance heatmap — who is fastest where. Battle forecast — where position changes will happen over the next 15 laps. Pilot error log."

*(Tap the battle forecast heatmap)*

> "Red means DRS range — a battle is imminent. Green means the gap is safe. The model projects 15 laps forward from any selected lap."

---

### [2:40 – 2:55] THE VISION

**Screen:** Return to Momentum Map or full app view — keep app visible in background.

> "This isn't just a Formula 1 project."

> "Any motorsport with telemetry — junior series, sim racing, esports leagues, fan communities — can have pit wall intelligence at this level."

> "IBM Granite made it possible to serve a 14-year-old fan and a professional race engineer from the same model and the same dataset. That's the real unlock here."

---

### [2:55 – 3:00] CLOSE

**Screen:** App URL displayed prominently.

> "PitMind. F1 intelligence for everyone."

> "Live at pitmind-bvihgej8qmkzrergphk4xb.streamlit.app"

---

## Key Lines to Nail

These are the phrases judges will remember — deliver them clearly and with a beat of silence after each:

1. **"Same model. Two audiences."** *(after the toggle reveal)*
2. **"The fan understands the tension without understanding the telemetry."**
3. **"No subjective scoring — the model reads the data the way a race engineer would."**
4. **"IBM Granite made it possible to serve a 14-year-old fan and a professional race engineer from the same model."**

---

## Pre-Recording Checklist

- [ ] Hard-refresh the app (Ctrl+Shift+R) — confirm race shows **Bahrain 2025**, not 2023
- [ ] Navigate to: Pit Wall Mirror → VER → Lap 28 before pressing record
- [ ] Toggle is set to **Fan Mode** at the start of the recording
- [ ] Reset pit decision (if previously made) — click "Reset decision" button
- [ ] Engineer Mode is visible in the sidebar and responsive
- [ ] Granite narratives are loading (test one call before recording)
- [ ] Screen recording software capturing tab audio (for any ambient sound)
- [ ] Browser zoom at 90% so all panels fit without scrolling mid-sentence

---

## What Makes This a 9.5+

**1. Opens without an introduction.**
The Abu Dhabi hook forces judges to feel the problem before you've said a word about your solution. Most demos open with "Hi, my name is..." — this one doesn't.

**2. The toggle is the centrepiece, not a footnote.**
Most hackathon demos show AI as a sidebar feature. This script makes the fan/engineer switch the central reveal — judges understand in 30 seconds why IBM Granite was the right choice.

**3. Show Granite generating text live.**
Click "Generate race narrative" and let it load on screen in real time. Real API calls in a live demo prove genuine integration — not mocked responses.

**4. "Same model. Two audiences."**
Say this line exactly. It is the thesis of the entire project in four words. Judges will write it down.

**5. The vision at 2:40 scales beyond F1.**
Judges ask "could this be real?" — the junior series / sim racing line answers yes without overpromising.

**6. Narrate the data, not the UI.**
Don't say "here you can see the button." Say what the data *means* while pointing at the UI. That's how engineers present — and these judges are engineers.

---

*PitMind · IBM x May Challenge 2026*
