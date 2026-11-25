"""Round 45 FRESH generated scenes: same 4 initially-sharp categories, generator, settings and prompt suffix as round 40
(prompts40.py), but NEW subjects and backgrounds and new seeds (4500+), written before any round 45 image exists.
6 per category = 24 scenes; all are test scenes (the controllers were fitted on round 40/44 development scenes only).
"""
AIF = ", deep depth of field, everything in sharp focus, f/16, highly detailed, natural light, photograph"
AIF_NEG = "bokeh, blurry background, shallow depth of field, defocus, blur, painting, illustration, cartoon, text, watermark"

SUBJECTS = {
    "hair": [
        "a woman with a messy bun and loose strands in front of a wall of potted plants",
        "a man with long wavy hair in front of a rough concrete wall with posters",
        "a teenager with curly afro hair in front of a chain-link fence and trees",
        "a woman with short tousled grey hair in front of a market stall with fruit",
        "a man with a ponytail and stray hairs in front of a timber barn wall",
        "a girl with pigtails and frizzy hair in front of a hedge of roses",
    ],
    "fur": [
        "a long-haired goat standing in front of a dry stone wall",
        "a fluffy persian cat sitting in front of a knitted blanket pile",
        "a sheep with thick wool in front of a rocky meadow",
        "a husky with thick fur in front of a snowy pine forest",
        "a fluffy hamster on a wooden table in front of a bookshelf",
        "an alpaca with shaggy fleece in front of a wooden fence and hills",
    ],
    "thin": [
        "a garden rake with thin tines leaning against a brick shed",
        "a tripod with thin metal legs standing on a rocky shore",
        "a spider web with dew drops in front of a dense hedge",
        "a wrought iron chair with thin curly legs on a cobblestone terrace",
        "a bunch of thin dried grass stems in a jar in front of a bookshelf",
        "a person holding a thin violin bow in front of a wooden wall",
    ],
    "highlights": [
        "a woman in front of a night market with strings of light bulbs",
        "a polished brass trumpet on a stand in front of a busy bar",
        "a man at night in front of a row of lit shop signs",
        "a crystal chandelier in front of an ornate wallpapered wall",
        "a motorbike with chrome parts in front of a sunlit garage door",
        "a child holding a glowing lantern in front of a crowd at dusk",
    ],
}


def scenes():
    out = []
    for cat, ps in SUBJECTS.items():
        for k, p in enumerate(ps):
            out.append(dict(id=f"f_{cat}_{k}", category=cat, split="test", seed=4500 + len(out), prompt=p + AIF,
                            negative=AIF_NEG))
    return out
