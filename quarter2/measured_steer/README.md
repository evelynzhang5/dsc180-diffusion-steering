# Round 32: does measured-error adaptive steering beat equally adaptive request adjustment?

**Question.** Round 30 found no reliable one-pass predictor of each region's blur error. This round uses the *measured* error instead, in two passes:
1. Generate once and measure each region's error.
2. Try small positive and negative edits along a candidate direction.
3. Pick each region's strength from the measurements.
4. Compare with an equally adaptive blur-request adjustment, including scene preservation.

A probe that supplies the strength in one pass is only worth training if this passes.

**Why a new direction.** Rounds 29 and 31 already searched per case with the target known, which is an upper bound on any adaptive rule over the same edits:
- round 29: spatial feature smoothing at 6 locations;
- round 31: per-block LoRA scaling.

Neither beat the request by 20 %. So this round tests *channel directions* (classic steering vectors) taken from round 30's activations, which have not been tried.

**Candidate directions (`directions32.py`, from round 30's development cells only).**
- **Data:** 432 cells from 72 generations. Sources: grass, brick, camera, coffee, immunohistochemistry, page. None of this round's cases are among them.
- **Two measured labels per cell:**
  - *size*: log(delivered radius / target);
  - *shape*: primary error against an ideal disk blur at the delivered radius, i.e. the error left after size.
- **Centring:** labels and the cell-pooled features are centred within each generation.
- **Direction:** the mean feature of cells above the label median minus the mean of cells below.
  - This is a difference of means, not the probe readout w.
  - Positive points toward over-blur (size) or larger shape error.
- **Layers:**
  - mid_block, the best exploratory probe layer in round 30 (held-out ρ 0.59);
  - up_blocks.0 (ρ 0.47).
- **Control:** a fixed-seed random direction of the same magnitude.

Per-direction statistics, in-sample on dev and therefore optimistic:

| Layer | Direction | Rank correlation of projection with label | Difference norm / pooled feature norm | cos(size, shape) |
|---|---|---|---|---|
| mid | size | 0.64 | 0.22 | −0.51 |
| mid | shape | 0.67 | 0.23 | |
| up0 | size | 0.66 | 0.16 | −0.56 |
| up0 | shape | 0.73 | 0.15 | |

**Edit.** h′ = h + Σ_k a_k · m_k · u.
- m_k: the area fraction of case cell k on the feature grid.
- u: the direction scaled to the dev high-vs-low difference. a = 1 shifts a cell by the full difference between high- and low-error development cells.
- Edits apply only to the 4 case cells. rocket_0, text_0 and the focus are never edited.

**Configuration and cases.**
- **Configuration:** round 25's baseline: fp32 PISA, original gate, frozen radius curve, seed 0, held-out group 0 / position 0.
- **Cases:** round 29's eight:
  - gravel_0, astronaut_0, chelsea_0, coins_0;
  - each at targets 8 and 11.625 px.
- **Primary error:** round 29's gain/offset-adjusted MAE against the sRGB disk-blurred input. This reference is computable from the input alone, so the adaptive rule is deployable.

**Procedure (`run32.py`, per target; 4 cells corrected at once in every generation).**

| Stage | Generations | What |
|---|---|---|
| base | 1 | Pass 1: round 25 baseline; measure each region |
| R1 | 6 probes + 1 final | Request factors {0.7, 0.85, 0.95, 1.05, 1.15, 1.3}; per cell choose; combined final |
| R+R (comparison) | 6 + 1 | A second request round on R1: {0.9, 0.95, 0.98, 1.02, 1.05, 1.1} |
| S+R (per arm) | 6 + 1 | A steering round on R1's requests: a ∈ {−2, −1, −0.5, 0.5, 1, 2} |
| S (per arm, reported) | 6 + 1 | Steering from the baseline request (steering instead of the request) |

**Choosing.** Each cell picks from its round's start generation plus the 6 probes:
- the lowest primary error among valid outputs, where valid means |r̂ − t| ≤ max(2 px, 20 %) and focus guard ≤ 0.5/255;
- if none is valid, the output whose radius is closest to the target.

There are 6 arms ({mid, up0} × {size, shape, random}), about 203 generations in total.

**Gate (smoke stage; aborts on failure).**
- The baseline reproduces round 25 exactly at both targets.
- A repeat is identical.
- a = 0 at each layer is identical.
- a = 2 along the random direction changes the output at each layer.
- Each hook fires once per generation.

**Rules (fixed before running; `analyze32.py`).**
- **Compared outputs:** each method's final combined generation, one image per method and target.
- **Comparison:** steering S+R vs request R+R. Both start from R1 and get the same 7 extra generations, so a win means steering adds something beyond adapting the request.
- **Unattainable cases are failures:**
  - steering invalid → ratio ∞;
  - request invalid and steering valid → ratio 0.
- **An arm PASSES if all three hold:**
  - median (steering error / request error) ≤ **0.80**;
  - wins ≥ **6 of 8**;
  - MAE change in the unedited cells (rocket_0, text_0) vs the baseline ≤ the request method's + 0.5/255, averaged over the targets.
- **Round PASS:**
  - some size or shape arm passes, **and**
  - the random arm at the same layer does not meet the ratio-and-wins part, so the gain must be specific to the direction.
- **Otherwise FAIL:** "measured-error adaptive steering along these directions does not beat equally adaptive request adjustment".
- **Multiple arms:** four real arms are tested, so a PASS is provisional. It leads to a confirmation round on fresh source images with the arm and the procedure frozen, followed by step 4 (a one-pass probe for the strength).
- **Visual check:** the crop sheet (`results/crops_t11.6.png`, best arm) is inspected for damage before any claim.
- **Reported, not decided on:** S vs R1 (steering instead of a request round); probe responses (error and radius vs a) per arm and cell.

Files: `directions32.py` (run locally on round 30's saved outputs; writes `results/directions32.npz/.json`), `run32.py`, `analyze32.py`, `dsmlp_measured.sh`. Shared modules are copied from round 31.

## Results (Nov 20, 2026; 203 generations at 1.8 s, pod `q2-measured`)

**Gate passed exactly.**
- Round 25 was reproduced at both targets, and the repeat was identical (max difference 0).
- a = 0 was identical at both layers.
- Random a = 2 changed the output (max 3/255 at mid, 5/255 at up0).
- Each hook fired once per generation.

**Decision: FAIL.** No arm comes close.

| Arm | S+R vs R+R: median ratio | Wins | Unedited-cell change (steer / request) | S vs R1: median | Wins |
|---|---|---|---|---|---|
| mid size | 1.01 | 3/8 | 0.21 / 0.20 | 1.06 | 1/8 |
| mid shape | 1.01 | 2/8 | 0.20 / 0.20 | 1.13 | 0/8 |
| mid random | 1.01 | 1/8 | 0.20 / 0.20 | 1.17 | 0/8 |
| up0 size | 0.99 | 5/8 | 0.24 / 0.20 | 1.02 | 3/8 |
| up0 shape | 1.01 | 3/8 | 0.23 / 0.20 | 1.10 | 0/8 |
| up0 random | 0.98 | 5/8 | 0.27 / 0.20 | 1.15 | 1/8 |

Measured-error adaptive steering along these directions does not beat equally adaptive request adjustment.
- Every arm is within ±2 % of a second request round.
- The random direction does as well as the best real one.
- Steering *instead of* a request round is worse in every arm.
- The crop sheet shows no damage; steered and request-adjusted outputs look the same.

**What the probes show (reported; `results/measured_steer_cases.json`).**
- **The size directions control the blur amount, and the shape directions do not fix shape.**
  - up0 size moves the delivered radius smoothly and in one direction in every cell: a = −2 → +2 shifts it by about 1.5–2.7 px.
  - mid size moves it by about 0.5–1.8 px.
  - The random direction barely moves the radius (≤ 0.2 px).
- **But that is the same knob the request already turns.** The error follows the radius: steering toward the requested size helps exactly as much as adjusting the request does. It does not reduce the remaining shape error.
- **The shape directions' best settings are mostly a = ±2, at the edge of the probe range.** Their gains are no larger than the random direction's, and the random direction changes the error (e.g. astronaut and coins improve at a = −2) without moving the radius.
- So no tested direction has a content-specific effect on shape.

**Reading.** The directions round 30's activations provide are blur-*amount* directions, and the request already gives amount control. Combined with rounds 29 (spatial smoothing) and 31 (LoRA scaling), three kinds of internal edit now fail to remove the content-dependent shape error beyond what request adjustment achieves. Step 4 (a one-pass strength probe) is not warranted.
