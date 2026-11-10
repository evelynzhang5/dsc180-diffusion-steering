# Composition rubric for the AI audit (written before review, 2026-11-10)

Each photo gets exactly one label, judged from the 512 px center crop used for the activations:

- **C, close-up**: one identifiable subject (person, animal, plant or flower, object, food, ...) or a tight group
  read as one subject fills much of the frame (roughly a third of the frame or more, or its main extent spans most
  of the frame); the surroundings are context, not the content.
- **W, wide scene**: a scene with no subject that dominates the frame: landscape, street, room, aerial, sky, or a
  subject that is small within a larger scene.
- **A, ambiguous / texture**: full-frame texture or pattern with no single subject (water, foliage, rock, a pile of
  identical items, bokeh only), or a case between C and W that the reviewer cannot call.

Each photo also gets one subject category for variety: person, animal, plant, object, food, landscape, urban,
interior, texture, other.

Review conditions: sheets are numbered and shuffled across all four focus x CLIP groups (the two rare groups in
full, 24 random photos from each common group); focus labels and CLIP probabilities are in a separate key opened
only after all labels are written. The reviewer is Claude (AI audit); a human spot-check follows on a random subset.
Only A and C/W disagreements matter for set construction: rare-group photos labelled A or the opposite class are
dropped.
