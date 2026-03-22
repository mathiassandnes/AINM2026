# NM i AI & DM i AI — Competition History & Strategy Notes

## Challenge Pattern (Consistent Across All Years)

One challenge per AI domain: **RL/Control**, **NLP/Language**, **CV/Vision**

| Year | Competition | Challenge 1 (RL/Control) | Challenge 2 (NLP) | Challenge 3 (CV) |
|------|-----------|--------------------------|-------------------|-------------------|
| 2022 | DM i AI | Robot Robbers | Sentiment Analysis | Pig & Piglet Detection |
| 2023 | DM i AI | Lunar Lander | AI Text Detector | Tumor Segmentation |
| 2024 | DM i AI | Traffic Simulation | — | Cell Classification + CT Inpainting |
| 2025 | NM/DM i AI | Race Car | Emergency Healthcare RAG | Tumor Segmentation |
| 2026 | NM i AI | Warehouse Bot (warmup) | TBD | TBD |

---

## Detailed Challenge Specs (What to Expect)

### RL / Control Challenges

**Robot Robbers (2022)** — 5 robots on 128x128 grid, steal cashbags from 7 AI scrooges, deposit at 3 dropspots. State: 6x10x4 matrix. Quadratic reward for batching deposits (1=1pt, 2=4pt, 3=9pt). Scrooge intercept = -3. Carrying bags slows robots. 2min wall-clock.

**Lunar Lander (2023)** — Gymnasium LunarLander-v2 (discrete). 8-dim observation, 4 actions. 10 games in 2 minutes. 200+ reward/game = success. Standard DQN/PPO benchmark.

**Traffic Simulation (2024)** — SUMO-based traffic light control. Minimize vehicle wait times. Score penalizes waits >90s exponentially: `sum(Qi) + sum(max(0, (90-Qi)^1.5))`. Must respect min green (6s) and transition times. 1s per tick, 10min simulation.

**Race Car (2025)** — 5-lane track, 60 seconds, avoid obstacles. 8 sensors (22.5-degree intervals, 1000px range). Actions: accelerate/decelerate/steer/nothing. Scored by distance. 60 ticks/sec. **No cloud APIs during inference.**

### NLP Challenges

**Sentiment Analysis (2022)** — Predict star ratings (1-5) for 1000 Amazon reviews. MAE scoring. 30s for all reviews. Participants source own training data.

**AI Text Detector (2023)** — Binary classify ~2000 texts as human vs AI. Accuracy scoring. 60s limit. 1133 validation samples, no labels on test. Teams could generate own AI text for training.

**Emergency Healthcare RAG (2025)** — Per medical statement: (1) true/false, (2) classify into 1 of 115 emergency topics. 5s per statement, 24GB VRAM max, **fully offline** (no cloud APIs). 200 training statements (Claude-generated from StatPearls). 115 topic articles provided as retrieval corpus. Recommended: Ollama, llama.cpp, vLLM, HuggingFace Transformers.

### CV Challenges

**Pig & Piglet Detection (2022)** — Object detection: pigs (class 0) vs piglets (class 1). COCO mAP (IoU 0.5-1.0). 10s per image. Only 2 annotated samples provided — need own training data. YOLO format.

**Tumor Segmentation (2023 + 2025)** — Pixel-wise tumor segmentation in MIP-PET images. Dice-Sorensen coefficient. 10s per image. 182 training images + 426 healthy controls. Key difficulty: normal organs (brain, kidneys, bladder, liver) show high FDG uptake mimicking tumors.

**Cell Classification (2024)** — Bone marrow cells: heterogeneous (0) vs homogeneous (1) from fluorescence microscopy. Score = product of per-class accuracies (one bad class zeros score). **Trap: 16-bit training images, 8-bit val/test.** 10s per image.

**CT Inpainting (2024)** — Reconstruct corrupted regions in 256x256 CT slices. MAE scoring (baseline MAE=6). Extra context: corruption mask, tissue type, vertebrae position. 5900 training samples. 10s per image. Cannot use pretrained AutoPET models.

---

## Known Winning Approaches

### Tumor Segmentation (MagnusS0, DM i AI 2023) — Dice 0.84
- **Architecture:** Attention U-Net with self-attention gates
- **Framework:** MONAI (PyTorch medical imaging)
- **Loss:** DiceFocal hybrid (lambda_dice=1, lambda_focal=10)
- **Key insight:** Attention gates discriminate tumors from organs with naturally high glucose uptake

### General Patterns from Winners
1. **Infra ready day 1** — FastAPI endpoint, deployment pipeline, GPU access sorted before challenges drop
2. **Explore multiple approaches in first 1-2 days**, then commit to best
3. **Pretrained models are essential** — competition deliberately gives minimal training data
4. **Don't overfit validation** — final test set is separate and different
5. **External training data** — participants expected to source their own
6. **Submit early** — one-shot final eval, last-minute bugs are fatal
7. **Bit-depth / format traps** — watch for mismatches between train and test data
8. **Time constraints matter** — 5-10s inference limits mean you can't use huge ensembles naively

### What Separates Winners

**Team PER (DM i AI 2024, 1st):**
- Technical setup prepared **before** competition
- Effective task division
- Early exploration, then refine
- Thorough testing

**AlphaGo-Home (DM i AI 2022+2023, 1st both years):**
- Repeat champions — experience compounds
- Explored techniques beyond coursework

**Attention Heads (NM i AI 2025, 1st):**
- CS + Cybernetics/Robotics mix
- No public writeups

---

## Competition Infrastructure

| Aspect | Detail |
|--------|--------|
| Framework | FastAPI + Pydantic DTOs |
| Deployment | Public IP (Azure free, UCloud GPUs, or port-forward) |
| Validation | Unlimited attempts during competition |
| Final eval | **One-shot** against hidden test set |
| GPU access | UCloud: NVIDIA L4 (24GB) or L40 (48GB) |
| Cloud APIs | Prohibited during inference (since 2025) |
| Scoring | F1-style points (since 2025): 25/18/15/12/10/8/6/4/2/1 |

---

## Participant Solution Repos
- [MagnusS0/tumor-segmentation](https://github.com/MagnusS0/tumor-segmentation) — Attention U-Net, Dice 0.84 (2023)
- [adriansousapoza/DMIAI-2023](https://github.com/adriansousapoza/DMIAI-2023) — All 3 challenges in notebooks
- [andersthuesen/DM-i-AI-submissions](https://github.com/andersthuesen/DM-i-AI-submissions) — 2021 solutions
- [amboltio/DM-i-AI-{2022,2023,2024,2025}](https://github.com/amboltio) — Official starter repos

---

## Playbook for NM i AI 2026

Given that Mathi is competing solo (no task division possible), priority is:

1. **First 2 hours per challenge:** Read spec carefully, get baseline running, submit once to validate pipeline
2. **Next 4-6 hours:** Try 2-3 approaches, pick the one that scores best on validation
3. **Final push:** Refine best approach, ensure robust submission
4. **Don't burn out** — diminishing returns after ~6-8 focused hours per challenge
5. **Leverage strengths:** LLM/RAG challenge is your wheelhouse, CV should be comfortable, RL/control is familiar from masters
6. **Use pretrained models aggressively** — no time to train from scratch as a solo competitor
