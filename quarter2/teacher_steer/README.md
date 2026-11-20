# Round 33: can the teacher's activation difference correct the output beyond the request?

**Question.** Rounds 29, 31 and 32 used three kinds of internal edit:
- spatial feature smoothing (round 29);
- per-block LoRA scaling (round 31);
- steering directions from activations (round 32).

All three only re-weighted what the model already does, and none beat request adjustment. This round brings in information about the *correct* output.

A **teacher** run shows the model the answer: the same model and seed, with the case cells already blurred to the target and a zero blur request there, so the model should pass them through. Injecting the teacher-minus-student activation difference into the student asks whether a correct-output signal at an internal location is enough to fix the output. This is privileged feasibility: the teacher needs the target. Only a PASS would justify learning to predict the difference.

**Fixed configuration and cases.**
- **Configuration:** round 25's baseline: fp32 PISA, original gate, frozen radius curve, seed 0, held-out group 0 / position 0.
- **Cases:** round 29's eight:
  - gravel_0, astronaut_0, chelsea_0, coins_0;
  - each at targets 8 and 11.625 px.
- **Primary error:** round 29's gain/offset-adjusted MAE against the sRGB disk-blurred input.

**Teacher.**
- **Input:** the student's input with each of the 4 case cells replaced by the target, i.e. the input disk-blurred at t.
- **Requests:** a zero blur request in those cells; the student's disparity and every other cell's request unchanged.
- **Recording:** activations at each location, from one teacher generation per target.

**Steps (`run33.py`).**

| Step | What | Generations |
|---|---|---|
| Gate | Baseline reproduces round 25 exactly. A repeat is identical. Round 31's 8 request-only winners regenerate exactly. α = 0 at each location reproduces the baseline. Every hook fires once | 25 |
| 1. Pass-through | Teacher output vs the target in the case cells | 2 |
| 2. Injection | h′ = h + α · m · (h_teacher − h), m = area fraction of the case cells. Locations: PISA block output, down_blocks.2, mid_block, up_blocks.0, up_blocks.1 (hidden states only; skip connections stay the student's), and up_blocks.2 as the ceiling. α ∈ {0.25, 0.5, 1, 1.5} | 48 |

**Rules (fixed before running; `analyze33.py`).** Valid means |r̂ − t| ≤ max(2 px, 20 %) and focus guard ≤ 0.5/255.
1. **Pass-through.**
   - The teacher keeps the target if, in ≥ 6 of 8 cases, its output is valid and its error is ≤ 0.5 × the request-only error.
   - Otherwise the round is **PASS-THROUGH FAIL**: the teacher does not encode the target better than the request, and step 2 is not interpreted.
2. **Per candidate location** (all except up_blocks.2):
   - per case, the best valid α (privileged, like the request-only search);
   - unattainable = failure.
   - **A location PASSES if all three hold:**
     - median (injection / request-only) ≤ **0.80**;
     - wins ≥ **6 of 8**;
     - the unedited cells' change (rocket_0, text_0; MAE vs baseline, max over α and targets) ≤ the largest request-only other-cell change + 0.5/255.
3. **Round result:**
   - **PASS** if some location passes, otherwise **FAIL**;
   - up_blocks.2 is reported as the ceiling and cannot pass.
4. **Visual check:** the crop sheet (`results/crops_t11.6.png`) is inspected before any claim.
5. **After a PASS:** a separate round on fresh source images, then a test of whether the difference can be predicted without the target.

**Comparison.** Round 31's request-only search: about 24 requests per case, one cell at a time. Injection gets 4 α values per location, applied to all 4 cells at once. The comparison therefore favours the request.

Files: `run33.py`, `analyze33.py`, `dsmlp_teacher.sh`, and `results/r31_cases.json` (round 31's request-only winners). Shared modules are copied from round 31.

## Results (Nov 20, 2026; 73 generations, pod `q2-teacher`)

**Gate passed exactly.**
- Round 25 was reproduced and the repeat was identical.
- All 8 of round 31's request-only winners regenerated exactly.
- α = 0 reproduced the baseline at every location.

**Step 1, pass-through: passed 8/8.** Given the answer, the model keeps it. The teacher's error is 0.69–2.06, against 1.96–10.20 for the best request-only output, and its radius is within 0.2 px of the target. Coins drops from 10.2 to 0.8.

**Step 2, injection: FAIL.** No location passes.

| Location | Median ratio (bar 0.80) | Wins | Unedited-cell change (request 0.16) |
|---|---|---|---|
| PISA block output | ∞ (4 cases unattainable) | 3/8 | 1.73 |
| down_blocks.2 | 1.00 | 4/8 | 0.33 |
| mid_block | 1.02 | 4/8 | 0.20 |
| up_blocks.0 | 1.12 | 0/8 | 2.08 |
| up_blocks.1 | 1.30 | 2/8 | 3.63 |
| up_blocks.2 (ceiling) | ∞ | 2/8 | 3.61 |

**The design is flawed, so this FAIL does not test the question.** Even the ceiling fails: injecting the teacher's last-block activations at α = 1 makes the output **sharp** (delivered radius 1.2–1.7 px at both targets), not blurred like the teacher's. The same happens at the PISA output (α ≥ 0.5 → 1.3–2.5 px).

BokehDiff predicts its single step relative to its own input. The teacher's input is already blurred, so its activations encode "change nothing here", not "this is the target image". Transplanted onto the sharp student input, that instruction removes the blur. The pass-through gate checked the teacher's *output*, but the signal transferred is the teacher's *input-relative update*, which differs between teacher and student by construction.

At the bottleneck (down_blocks.2, mid_block) the transplant behaves like a mild blur-amount edit: the radius moves toward the target as α grows, and error falls in some cases (gravel 2.38 → 1.87). This matches round 32's size directions and does not beat the request. The crop sheet shows no damage at the best down_blocks.2 setting.

**Reading.**
- An activation difference between runs with different inputs mixes the input change into the signal. For this one-step, input-anchored editor, a usable teacher must share the student's sharp input.
- The only teacher that does is an optimised one: find, by gradient descent through the single UNet step and decoder, the activation edit at a location that makes the student's output match the target.
- That gives the true upper bound on what any steering at that location can achieve, and is the correct version of this feasibility test.
