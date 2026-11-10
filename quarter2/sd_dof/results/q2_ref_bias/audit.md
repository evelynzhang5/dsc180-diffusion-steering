# Blind audit of the automatic composition labels (2026-11-10)

24 photos per attribute, 6 random per cell (CLIP threshold 0.5), shuffled and numbered; labelled by eye from
`audit_<attr>_blind.png` before opening `audit_<attr>_key.json`.

## Portrait (a person is the main subject and fills a large part of the frame)

By eye: portraits #3, #4, #18, #20; borderline #14; the other 19 not.
- CLIP "not portrait" (12): all 12 agree.
- CLIP "portrait" (12): 4 agree, 1 borderline, 7 wrong (snowman, cat, spider, dog, empty road, distant person,
  people in a scene).
- Estimate: about 20-25 true portraits among the 1,200 photos, fewer than ~10 with deep depth of field. Too few for
  balanced / biased / opposed reference sets without reusing the same few photos.

## Close-up (one subject, of any kind, fills a large part of the frame; vs. a wide scene)

By eye: close-up #1, 3, 4, 5, 10, 13, 14, 16, 17, 18, 20, 22; full-frame textures with no single subject
#8, 12, 15, 21 (counted as not close-up); borderline #23 (not).
- Agreement 16 / 24 (Cohen's kappa about 0.33).
- CLIP missed 4 close-ups (#1 woman, #13 branch, #17 leaf, #22 pinecone; probabilities 0.18-0.49).
- CLIP called 4 non-close-ups close-up: 3 textures (#8 fruit, #15 sea foam, #21 rock; all deep depth of field) and
  #19 (small bird against mountains).
- At the extremes it agrees: 6 of 7 with p >= 0.85 correct (the miss is texture #8), 5 of 5 with p <= 0.15 correct.

Counts with confident labels only (close-up p >= 0.85, wide p <= 0.15):

| | close-up | wide |
|---|---|---|
| shallow | 274 | 54 |
| deep | 52 | 427 |

# Close-up rare-group audit (AI audit, 2026-11-10)

Rubric written before review: `rubric.md` (C close-up / W wide / A ambiguous or texture, plus a subject category).
154 photos on 7 shuffled sheets (`audit2_sheet0-6.png`): both rare groups in full and 24 random photos from each
common group, confident CLIP labels only (close-up p >= 0.85, wide p <= 0.15). Labels (`audit2_labels.txt`) were
written by Claude with the group key (`audit2_key.json`) still on DSMLP; the key was copied only afterwards.
Human spot-check: pending (20 photos: 7, 9, 10, 12, 14, 15, 18, 27, 47, 48, 52, 57, 60, 87, 89, 91, 99, 104, 108, 113; 10 of them from the shallow-focus wide group).

| group (confident CLIP label) | photos | AI audit: close-up | wide | ambiguous / texture | confirmed for its group |
|---|---|---|---|---|---|
| shallow focus, wide (rare) | 54 | 17 | 15 | 22 | **15** (12 landscape, 2 interior, 1 urban) |
| deep focus, close-up (rare) | 52 | 32 | 0 | 20 | **32** (16 animal, 9 plant, 5 object, 2 person) |
| shallow focus, close-up (sample) | 24 | 23 | 0 | 1 | 23 of 24 |
| deep focus, wide (sample) | 24 | 0 | 22 | 2 | 22 of 24 |

Log Laplacian variance of the confirmed photos (10 / 50 / 90%): shallow wide 3.89 / 5.31 / 5.64; deep close-up
4.63 / 5.45 / 6.13; shallow close-up 2.90 / 4.06 / 4.65; deep wide 4.36 / 5.50 / 6.16. The confirmed shallow-focus
wide photos are about as sharp as deep-focus photos, so a set that fills its shallow class with them weakens the
shallow-vs-deep sharpness contrast.

Share of close-ups among confident CLIP labels: 274 / 328 = 84% of shallow-focus photos, 52 / 479 = 11% of
deep-focus photos (the 74% / 17% quoted earlier was for the 0.5 threshold over all 600 photos per class).

**Decision: stop the reference-bias experiment on this dataset.** The pilot needed at least 32 confirmed photos in
each rare group (40 per class at an 80% share). Deep close-ups reach exactly 32; shallow-focus wide scenes reach 15,
nearly all landscapes, and they are not blurrier than deep-focus photos, so the opposed set could not keep the
sharpness contrast comparable. Lowering the set size to fit 15 (about 18 per class) or the imposed bias would leave
the opposed condition too small or too weak to be a test.

Portrait note: the portrait design is abandoned on the 24-photo audit (4 of 12 CLIP "portraits" were portraits);
the "20-25 portraits" figure given earlier was a rough extrapolation, not a verified count.
