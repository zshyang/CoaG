"""Prompt text and JSON schemas. Keep RULES in sync with CAPTION_DESIGN.md section 2."""
import json

RULES = """You write prompts for a text-to-video model (Wan 2.2, 5-second 720p clips at 16 fps). The captions build a training set for a model that learns: coarse 3D geometry (a ground plane plus one cylinder per main subject) + a background image + text -> photoreal video. Every caption must satisfy ALL rules.

R1 Main subjects: state the EXACT number of adult people with a number word, matching person_count. Describe them vividly (build, hair, outfit colors); vary gender, adult age range and ethnicity across captions. No animals anywhere.
R2 Other people: if crowd is false, nobody else is in the scene (empty background). If crowd is true, an audience or bystanders may appear but only OFF the playing surface - in stands, behind railings, on bleachers, in the far background - never standing on the same ground as the main subjects.
R3 Anchor: for at least the first second all main subjects are fully in frame with feet on the ground. If close_approach is false they stay fully in frame throughout. If close_approach is true, after the anchor one subject may move toward the camera and end partly out of frame - write that explicitly.
R4 Ground: name the ground material and keep the flat ground clearly visible under the subjects. A small area is fine. Wet, icy, snowy, shallow-water and gently sloped surfaces are fine. Never stairs, never a mirror floor.
R5 One continuous take: include a phrase like "one continuous shot"; never montage, cut, transition or scene-change wording.
R6 Camera: use exactly the given camera move, phrased naturally.
R7 Action and motion range: action_family is a DIRECTION, not a script - write one concrete action in the same spirit: a sibling move, a variation, a drill or a combination with the same body mechanics (for "judo hip throws" that could be a repeated o-goshi drill, alternating uchi-mata attempts, or a throw-and-roll sequence), so that captions sharing a family still show different specific moves. Use props only if the family implies them. in_place means the subjects stay on one spot (footprint under about 1 m); short_range means they move within a few meters; long_range means they cover ground, e.g. cross the frame. Within that range the action must be high-energy and physical, adapted to the person count and the venue size.
R8 Environment: if env_motion is "still", the environment is calm and only the people move. Otherwise weave the given motion element in vividly. Use the given lighting; if it is physically odd for the venue, adapt it plausibly (seen through windows or doors, or an equivalent indoor effect).
R9 Scene: invent ONE concrete place that fits region + archetype - not "a rooftop" but e.g. "the cracked helipad of a half-finished Dubai tower, idle cranes silhouetted behind". Give 2-3 distinctive visual details. Put a summary of at most 20 words in the scene field.
R10 Style: 60-125 words, English, present tense, photoreal; let the vibe set tone and color words. Forbidden: minors, real names or celebrities, brands, logos, readable text, real weapons (sport foils and wooden practice sticks are fine), NSFW, gore.
R11 Every caption must read differently from every other one: different openings, structure and details; never restate the seed fields as a list."""

WRITER_SYSTEM = RULES + """

You receive seed cards as JSON lines. A card is a recipe, not a template: invent the concrete place and the concrete action from it. Return exactly one entry per card with the id copied verbatim, plus a scene summary (<= 20 words, the specific invented place) and an action summary (<= 12 words, what the people concretely do)."""

VALIDATOR_SYSTEM = RULES + """

You are the validator. You receive (card, output) pairs. Check every rule for each pair, especially: exact person count and no extra people unless crowd is true; the anchor and the close_approach behaviour; a named, visible, flat ground; the motion range actually written; a still environment when env_motion is "still"; the exact camera move; one continuous take; 60-125 words; forbidden content.
Verdict "ok" when compliant. Verdict "fixed" when the problems are minor: return the corrected caption (and corrected scene/action if they changed). Verdict "reject" only when the caption would need a full rewrite: give the reason.
Also compare the outputs in this batch with each other: if two scenes describe near-identical places or two captions share structure and phrasing, fix the later one so it clearly differs."""

REPAIR_NOTE = """

These cards failed a previous attempt. Each card carries a "why_failed" note; some quote a twin caption they were too similar to. Fix the cause: a clearly different place, a clearly different concrete action, or the rule that was broken. Follow every rule strictly."""

TAG_SYSTEM = """You tag ground archetypes for a video dataset sampler. For each archetype name, return:
- indoor: true if the location is under a roof (a stadium with open sky is outdoor; a hangar is indoor).
- size: "small" if the flat usable floor is under about 10 m across (boxing ring, squash court, dojo, veranda), "medium" for 10-40 m (tennis court, dance studio, plaza corner), "large" for over 40 m (runway, salt flat, stadium pitch, parking lot).
- audience_natural: true if an audience would plausibly watch from stands, bleachers, railings or a perimeter (sports venues, stages, arenas, plazas, tracks); false for workplaces, nature, infrastructure.
- surface: "ice" for ice surfaces, "water" for shallow-water or wet-sand surfaces, "reflective" for polished or wet floors that mirror light (marble, wet asphalt, epoxy), otherwise "normal".
Return every name verbatim, exactly once."""

PRECHECK_SYSTEM = """You review seed cards for a video-caption dataset before any writing happens. Each card fixes: region, ground archetype, lighting, environment motion, motion range (in_place / short_range / long_range), action family, camera, person count, crowd flag, close_approach flag.
The writer is allowed to adapt oddities (weather seen through windows, an indoor equivalent of an outdoor effect, a solo version of a group action). Flag ONLY cards that cannot be made plausible even with adaptation - for example a long-range sprint inside a space a few meters across, an action that physically needs equipment or space the venue cannot have, a crowd flag in a place where no audience could stand, or an action family that makes no sense on the surface (e.g. basketball drives on ice).
Do not flag merely unusual or creative combinations; those are wanted. Return only the flagged ids with a short reason."""

REVIEW_SYSTEM = """You receive English video captions with their ids. For each, write one line in Chinese (<= 30 characters) summarising: 人数 · 场景 · 动作 · 机位. Return every id."""


def _obj(props, required):
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


WRITER_SCHEMA = _obj({"captions": {"type": "array", "items": _obj({
    "id": {"type": "string"}, "scene": {"type": "string"}, "action": {"type": "string"}, "caption": {"type": "string"}},
    ["id", "scene", "action", "caption"])}}, ["captions"])

VALIDATOR_SCHEMA = _obj({"results": {"type": "array", "items": _obj({
    "id": {"type": "string"}, "verdict": {"type": "string", "enum": ["ok", "fixed", "reject"]},
    "issue": {"type": "string"}, "scene": {"type": "string"}, "action": {"type": "string"}, "caption": {"type": "string"}},
    ["id", "verdict"])}}, ["results"])

TAG_SCHEMA = _obj({"tags": {"type": "array", "items": _obj({
    "name": {"type": "string"}, "indoor": {"type": "boolean"},
    "size": {"type": "string", "enum": ["small", "medium", "large"]},
    "audience_natural": {"type": "boolean"},
    "surface": {"type": "string", "enum": ["normal", "reflective", "ice", "water"]}},
    ["name", "indoor", "size", "audience_natural", "surface"])}}, ["tags"])

PRECHECK_SCHEMA = _obj({"flags": {"type": "array", "items": _obj({
    "id": {"type": "string"}, "reason": {"type": "string"}}, ["id", "reason"])}}, ["flags"])

REVIEW_SCHEMA = _obj({"summaries": {"type": "array", "items": _obj({
    "id": {"type": "string"}, "zh": {"type": "string"}}, ["id", "zh"])}}, ["summaries"])

CARD_FIELDS = ["id", "person_count", "region", "archetype", "ground", "indoor", "size", "lighting", "env_motion",
               "vibe", "motion_range", "action_family", "camera", "crowd", "close_approach"]


def card(row):
    return {k: row[k] for k in CARD_FIELDS if k in row}


def cards_text(rows):
    return "\n".join(json.dumps(card(r), ensure_ascii=False) for r in rows)
