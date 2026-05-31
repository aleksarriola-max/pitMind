# PitMind — Demo Video Script
**IBM x May Challenge 2026 · 3-Minute Walkthrough**

---

## Timing Overview

| Section | Time | Content |
|---------|------|---------|
| Hook | 0:00 – 0:18 | Problem statement — no intro |
| Problem | 0:18 – 0:40 | The data gap in F1 |
| AHA Moment | 0:40 – 1:30 | Fan vs Engineer toggle — the centrepiece |
| Pit Strategy | 1:30 – 1:55 | Scenarios, trajectory chart, ML confidence |
| Driver Soul | 1:55 – 2:15 | Behavioral fingerprint + engineer depth |
| Momentum Map | 2:15 – 2:35 | IBM Granite live generation |
| Vision | 2:35 – 2:52 | Why this scales beyond F1 |
| Close | 2:52 – 3:00 | Sign-off + URL |

---

## Full Script

---

### [0:00 – 0:18] THE HOOK

> *Screen: black. Text appears line by line:*
>
> **"Abu Dhabi 2025. Final lap. Lando Norris crosses the line."**
>
> *(pause 2 seconds)*
>
> **"World Champion. After 8 years of near misses."**
>
> *(pause 1 second)*
>
> **"The decision that sealed it was made in 3 seconds, 40 laps earlier."**
>
> *Cut to the app.*

**Screen:** Black screen with white text only. No app visible yet.

---

### [0:18 – 0:40] THE PROBLEM

**Screen:** App open — Pit Wall Mirror tab, Fan Mode, NOR, Monaco 2025, lap 30.

> "Formula 1 is the most data-rich sport on Earth. Every lap, a pit wall processes thousands of telemetry signals — tyre degradation, gap trajectories, radio sentiment, sector deltas — to make decisions in under 3 seconds."

> "That intelligence is invisible to fans. And completely out of reach for smaller racing programs."

> "We built PitMind to fix that."

---

### [0:40 – 1:30] THE AHA MOMENT — Fan vs Engineer Toggle

**Screen:** Pit Wall Mirror, Fan Mode, NOR, Monaco 2025, lap 30.

> "This is Fan Mode. Norris, lap 30 at Monaco — his tyres are 28 laps old."

*(Point at the fan status card)*

> "PitMind tells the fan: 'Tyres are very worn — team wants to pit.' Urgency: High. Leclerc is right behind — inside DRS."

> "IBM Granite narrates the situation in plain language. The fan understands the tension without understanding the telemetry."

*(Click 'Pit Now')*

> "The fan makes the call. And then — we show them what actually happened."

*(Outcome reveal appears — hold for 2 seconds)*

> "Now — same lap. Same driver. Same data."

*(Toggle to Engineer Mode in the sidebar)*

> "Engineer Mode."

*(The screen transforms — 8-metric telemetry dashboard appears)*

> "The engineer sees: MEDIUM compound, 28 laps. Pace delta plus 0.41 seconds per lap and degrading. Gap behind Leclerc: 0.7 seconds — inside DRS. Pit window pressure at 87 out of 100."

*(Scroll down to show Driver Soul Model Confidence)*

> "And here — the model's own confidence. 72% pit probability. 31% position gain. 18% incident risk. This is the XGBoost model speaking."

*(Scroll to the 4 scenario cards and trajectory chart)*

> "Four pit timing scenarios — Pit Now, Pit plus 3, Pit plus 5, Stay Out. Each one projected forward 15 laps. And below the cards — a trajectory chart. You can see exactly when Pit Now overtakes the field versus staying out."

*(Point at the car-ahead signal)*

> "The engineer also sees who's ahead: Verstappen — MEDIUM, 31 laps, pit pressure rising 5 per lap. The model is flagging an undercut window."

*(Scroll to IBM Granite engineer narrative)*

> "IBM Granite — same model, same race data — switches voice entirely. This is the dual-voice system. One Granite model. Two system prompts. Two audiences."

---

### [1:55 – 2:15] DRIVER SOUL

**Screen:** Switch to Driver Soul tab, NOR, Monaco 2025, lap 30.

> "Every driver in our dataset has a behavioral fingerprint."

*(Point at the radar chart)*

> "Eleven traits — overtake tendency, tyre preservation, restart aggression, dirty air tolerance — computed purely from telemetry. No subjective scoring. The model reads the data the way a race engineer would."

*(Switch to Engineer Mode — scroll to show Driver Soul Model Confidence and incident risk trend)*

> "In Engineer Mode: the Driver Soul model's live confidence — pit probability, position gain, incident risk — right at the decision moment. Plus the full prediction history across every lap of the race."

*(Tap incident risk chart)*

> "This is when the model thought a crash was most likely. That's XGBoost trained on behavioral traits with SHAP explainability — not a heatmap, a live prediction."

---

### [2:15 – 2:35] MOMENTUM MAP — Live IBM Granite Generation

**Screen:** Switch to Momentum Map tab.

> "Every race is tracked as a momentum score — pace delta, gap trajectory, radio sentiment, pit window pressure — combined per lap, per driver."

*(Point at a shift marker on the chart)*

> "When the model detects a momentum shift — IBM Granite annotates it."

*(Click 'Generate race narrative' button)*

> "On demand, Granite generates the full race story."

*(Let the narrative load and display — hold on screen for 3–4 seconds)*

> "Fan mode gets a narrative. Engineer mode gets telemetry language. Same API call. Same race data."

*(Expand a shift causation panel in Engineer mode)*

> "Engineer mode also shows exactly which signal triggered each shift — gap, pace, tyre age, radio sentiment — compared to the lap before."

---

### [2:35 – 2:52] THE VISION

**Screen:** Return to Momentum Map or full app view — keep app visible in background.

> "This isn't just a Formula 1 project."

> "Any motorsport with telemetry — junior series, sim racing, esports leagues, fan communities — can have pit wall intelligence at this level."

> "IBM Granite made it possible to serve a 14-year-old fan and a professional race engineer from the same model and the same dataset. That's the real unlock here."

---

### [2:52 – 3:00] CLOSE

**Screen:** App URL displayed prominently.

> "PitMind. F1 intelligence for everyone."

> "Live at pitmind-bvihgej8qmkzrergphk4xb.streamlit.app"

---

## Key Lines to Nail

These are the phrases judges will remember — deliver them clearly with a beat of silence after each:

1. **"Same model. Two audiences."** *(after the toggle reveal)*
2. **"The fan understands the tension without understanding the telemetry."**
3. **"No subjective scoring — the model reads the data the way a race engineer would."**
4. **"IBM Granite made it possible to serve a 14-year-old fan and a professional race engineer from the same model."**
5. **"This is the XGBoost model speaking."** *(when pointing at the ML confidence row)*

---

## Pre-Recording Checklist

- [ ] Hard-refresh the app (Ctrl+Shift+R) — confirm race shows **Monaco 2025**, not 2023
- [ ] Navigate to: Pit Wall Mirror → **NOR** → **Lap 30** before pressing record
- [ ] Toggle is set to **Fan Mode** at the start of the recording
- [ ] Reset pit decision if one was previously made
- [ ] Engineer Mode is visible in the sidebar and responsive
- [ ] Granite narratives are loading (do a test call before recording — first call can be slow)
- [ ] Confirm Driver Soul Model Confidence row shows real percentages (not "—")
- [ ] Confirm scenario trajectory chart renders (4 colored lines, y-axis inverted)
- [ ] Browser zoom at 90% so all panels fit without scrolling mid-sentence
- [ ] Screen recording software capturing tab audio

---

## What Makes This a 9.5+

**1. Opens without an introduction.**
The 2025 championship hook forces judges to feel the stakes before you've said a word about your solution. Most demos open with "Hi, my name is..." — this one doesn't.

**2. The toggle is the centrepiece, not a footnote.**
Most hackathon demos show AI as a sidebar feature. This script makes the fan/engineer switch the central reveal — judges understand in 30 seconds why IBM Granite was the right choice.

**3. The trajectory chart is the proof.**
Don't just say "the model recommends Pit Now" — show the 4-line chart where you can *see* when Pit Now gains track position. Visual proof beats verbal claims.

**4. "Same model. Two audiences."**
Say this line exactly. It is the thesis of the entire project in four words. Judges will write it down.

**5. Show Granite generating text live.**
Click "Generate race narrative" and let it load on screen in real time. Real API calls in a live demo prove genuine integration — not mocked responses.

**6. The vision at 2:35 scales beyond F1.**
Judges ask "could this be real?" — the junior series / sim racing line answers yes without overpromising.

**7. Narrate the data, not the UI.**
Don't say "here you can see the button." Say what the data *means* while pointing at the UI. That's how engineers present — and these judges are engineers.

---

*PitMind · IBM x May Challenge 2026*
