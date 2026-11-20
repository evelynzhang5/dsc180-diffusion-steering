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
