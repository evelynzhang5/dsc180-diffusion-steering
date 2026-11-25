"""Round 43 generated scenes (RealVisXL V5.0, same generator and settings as round 40): 16 transparent subjects, each
in front of a TEXTURED background that is visible THROUGH the subject. Thin-walled or flat clear objects only (no
solid crystal balls or thick lenses, whose through-image is strongly refracted). All-in-focus prompts as round 40.
Frozen before any image exists. All scenes are test scenes (nothing is fitted in round 43).
"""
AIF = ", deep depth of field, everything in sharp focus, f/16, highly detailed, natural light, photograph"
AIF_NEG = "bokeh, blurry background, shallow depth of field, defocus, blur, painting, illustration, cartoon, text, watermark"

PROMPTS = [
    "an empty clear drinking glass on a wooden table, a flower garden behind it",
    "an empty wine glass on a cafe table, a busy city street with shops behind it",
    "an empty clear glass vase on a windowsill, leafy trees outside the window",
    "a clear plastic water bottle standing on a park bench, a park with trees and paths behind it",
    "an empty clear glass jar on a kitchen counter, open shelves with colourful packages behind it",
    "a clear acrylic sign with a small printed logo on a bar counter, shelves of bottles behind it",
    "an empty clear glass teapot on a table, a full bookshelf behind it",
    "a clear acrylic display box on an office desk, an open-plan office behind it",
    "an empty clear glass bottle on a beach table, palm trees and a beach hut behind it",
    "an empty clear glass pitcher on a picnic table, a meadow with wildflowers and trees behind it",
    "an empty clear glass mug on a bar counter, a brick wall with framed posters behind it",
    "a transparent clear umbrella held up on a city sidewalk, colourful shop fronts behind it",
    "a clear glass candle holder without a candle on a garden table, a hedge with string lights behind it",
    "an empty clear perfume bottle on a dressing table, a patterned wallpaper wall behind it",
    "a clear plastic cup on a stone wall, a vineyard and hills behind it",
    "an empty clear glass carafe on a restaurant table, a tiled wall with plants behind it",
]


def scenes():
    return [dict(id=f"glass_{k}", category="transparent", split="test", seed=4300 + k, prompt=p + AIF, negative=AIF_NEG)
            for k, p in enumerate(PROMPTS)]
