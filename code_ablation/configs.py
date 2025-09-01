from dataclasses import dataclass

@dataclass
class Component:
    key: str
    label: str

COMPONENTS = [
    Component("clip","Visual"),
    Component("color","Color"),
    Component("season","Season"),
    Component("luminance","Luminance"),
    Component("texture","Texture"),
    Component("pattern","Pattern"),
    Component("text","TextPrompt"),
]

VARIANTS = {
    "clip_only": {"clip":1.0},

    "+color": {"clip":1.0, "color":0.5},
    "+luminance": {"clip":1.0, "luminance":0.5},
    "+texture": {"clip":1.0, "texture":0.5},
    "+pattern": {"clip":1.0, "pattern":0.5},
    "+text": {"clip":1.0, "text":0.5},

    "+color+texture": {"clip":1.0, "color":0.4, "texture":0.4},
    "+color+luminance": {"clip":1.0, "color":0.4, "luminance":0.4},
    "+pattern+luminance": {"clip":1.0, "pattern":0.4, "luminance":0.4},
    "+text+color": {"clip":1.0, "text":0.3, "color":0.4},

    "full_ext": {"clip":1.0, "color":0.35, "season":0.15, "luminance":0.25, "texture":0.25, "pattern":0.20, "text":0.15},

    # sanity checks
    "no_clip_sanity": {"color":1.0, "luminance":1.0, "texture":1.0, "pattern":1.0, "text":1.0},
}

SHAPLEY_REFERENCE = "full_ext"
