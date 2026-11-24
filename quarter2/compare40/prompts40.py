"""Round 40 generated scenes (RealVisXL V5.0): 5 categories x 8 prompts. In each category the first 2 are development
(calibration) scenes and the other 6 are test scenes. Every scene has one clear subject in front of a TEXTURED
background, so background blur is measurable. Four categories ask for an all-in-focus image; 'existing_blur' asks for
a photo that already has shallow depth of field and is analysed separately (no radius-accuracy analysis).
"""
AIF = ", deep depth of field, everything in sharp focus, f/16, highly detailed, natural light, photograph"
AIF_NEG = "bokeh, blurry background, shallow depth of field, defocus, blur, painting, illustration, cartoon, text, watermark"
SHALLOW = ", shallow depth of field, f/1.8, photograph"
SHALLOW_NEG = "painting, illustration, cartoon, text, watermark"

SUBJECTS = {
    "hair": [
        "portrait of a woman with long curly windswept hair standing in front of a dense ivy-covered brick wall",
        "a man with a messy beard and flyaway hair in front of a crowded bookshelf",
        "a girl with frizzy red hair standing in a field of tall dry grass",
        "an elderly man with wispy white hair in front of a stone wall covered in moss",
        "a woman with braided hair and loose strands in front of a wall of colourful graffiti",
        "a young man with spiky hair standing in front of dense green foliage",
        "a woman with long straight black hair blowing in the wind in front of a cobblestone street",
        "a child with tousled blond hair in front of a wooden fence and leafy hedge",
    ],
    "fur": [
        "a fluffy long-haired cat sitting in front of a pile of autumn leaves",
        "a golden retriever with shaggy fur in front of a brick wall with ivy",
        "a highland cow with long shaggy hair in front of a rocky hillside with heather",
        "an owl with detailed feathers perched in front of tree bark and pine needles",
        "a white fluffy dog sitting on gravel in front of a hedge",
        "a red fox with bushy fur standing in front of dense ferns",
        "a fluffy rabbit sitting in front of a stacked woodpile",
        "a squirrel with a bushy tail on a branch in front of dense leaves",
    ],
    "thin": [
        "a bicycle with thin spokes leaning against a textured stone wall",
        "a wooden chair with thin legs standing in a field of wildflowers",
        "a dandelion seed head in front of a gravel path",
        "a person wearing thin wire-rimmed glasses in front of a bookshelf",
        "a wire birdcage standing in front of a brick wall",
        "a tall thin grass stalk with seeds in front of a pebble beach",
        "a metal music stand with thin legs in front of a wooden panelled wall",
        "a fishing rod and line held by a person in front of reeds by a lake",
    ],
    "highlights": [
        "a woman at night in front of a busy city street with many small lights and neon signs",
        "a chrome espresso machine with specular reflections in a cafe with shelves of cups",
        "a man in front of a christmas tree covered in small fairy lights",
        "a glass of water with sparkling reflections on a table in front of a sunlit garden",
        "a cyclist at dusk in front of street lamps and lit shop windows",
        "a vintage car with shiny chrome bumper in front of a sunlit brick building",
        "a woman holding a sparkler at night in front of a crowd",
        "a dog standing in front of sunlight glittering on a river",
    ],
    "existing_blur": [
        "portrait of a woman with curly hair in a park",
        "a cat sitting on a windowsill",
        "a man in a jacket standing on a city street",
        "a flower in a garden",
        "a dog in a field of grass",
        "a woman reading a book in a cafe",
        "a bird on a branch",
        "a child laughing in a playground",
    ],
}


def scenes():
    out = []
    for cat, ps in SUBJECTS.items():
        for i, p in enumerate(ps):
            shallow = cat == "existing_blur"
            out.append(dict(id=f"{cat}_{i}", category=cat, split="dev" if i < 2 else "test", seed=1000 + len(out),
                            prompt=p + (SHALLOW if shallow else AIF), negative=SHALLOW_NEG if shallow else AIF_NEG))
    return out


if __name__ == "__main__":
    for s in scenes():
        print(s["id"], s["split"], s["seed"], s["prompt"][:80])
