"""Balanced dealing of 2000 seed cards from the axes (CAPTION_DESIGN.md section 4.1).

Deterministic given (axes, archetypes, tags, seed). No LLM calls.
"""
import json
import random
import re
from collections import Counter, defaultdict

RANGES = ["in_place", "short_range", "long_range"]


def _deck(values, n, rng):
    """A list of length n cycling through `values` as evenly as possible, shuffled."""
    if n <= 0:
        return []
    vals = list(values)
    rng.shuffle(vals)
    d = [vals[i % len(vals)] for i in range(n)]
    rng.shuffle(d)
    return d


def _weighted_deck(weights, n, rng):
    """weights: {value: share}. Exact integer counts by largest remainder, shuffled."""
    total = float(sum(weights.values()))
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    rem = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:rem]:
        counts[k] += 1
    d = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(d)
    return d


def deal(axes, archetypes, tags, seed):
    rng = random.Random(seed)
    N = axes["total"]
    regions = list(axes["regions"])
    arch_names = [a["name"] for a in archetypes]
    ground_of = {a["name"]: a["ground"] for a in archetypes}
    tag_of = {t["name"]: t for t in tags}
    missing = [n for n in arch_names if n not in tag_of]
    if missing:
        raise ValueError("archetypes without tags: %s" % missing[:5])
    per_region = N // len(regions)
    if per_region * len(regions) != N or per_region >= len(arch_names):
        raise ValueError("need total = regions x k with k < number of archetypes")
    rng.shuffle(regions)
    rng.shuffle(arch_names)

    rows = []
    for i in range(N):
        a = arch_names[i % len(arch_names)]
        t = tag_of[a]
        rows.append({"id": "c%04d" % i, "region": regions[i // per_region], "archetype": a, "ground": ground_of[a],
                     "indoor": bool(t["indoor"]), "size": t["size"], "audience_natural": bool(t["audience_natural"]),
                     "surface": t["surface"], "low_ceiling": bool(t.get("low_ceiling", False))})

    # person count: independent deck with the exact plan
    plan = {int(k): v for k, v in axes["person_count_plan"].items()}
    assert sum(plan.values()) == N, "person_count_plan must sum to total"
    small_rows = [r for r in rows if r["size"] == "small"]
    big_rows = [r for r in rows if r["size"] != "small"]
    small_w = {k: v for k, v in plan.items() if k <= axes.get("small_venue_max_people", 3)}
    sdeck = _weighted_deck(small_w, len(small_rows), rng)
    used = Counter(sdeck)
    rest = {k: plan[k] - used.get(k, 0) for k in plan}
    if min(rest.values()) < 0:
        raise ValueError("too many small venues for the person-count plan: %s" % rest)
    bdeck = [k for k, c in rest.items() for _ in range(c)]
    rng.shuffle(bdeck)
    for r, c in zip(small_rows, sdeck):
        r["person_count"] = c
    for r, c in zip(big_rows, bdeck):
        r["person_count"] = c

    # lighting: outdoor -> weather deck with a hostile cap; indoor -> indoor lighting deck
    outdoor = [r for r in rows if not r["indoor"]]
    indoor = [r for r in rows if r["indoor"]]
    hostile = [w["name"] for w in axes["weather_outdoor"] if w["hostile"]]
    benign = [w["name"] for w in axes["weather_outdoor"] if not w["hostile"]]
    n_h = int(round(axes["shares"]["hostile_weather_cap"] * len(outdoor)))
    wdeck = _deck(hostile, n_h, rng) + _deck(benign, len(outdoor) - n_h, rng)
    rng.shuffle(wdeck)
    for r, w in zip(outdoor, wdeck):
        r["lighting"] = w
    for r, l in zip(indoor, _deck(axes["lighting_indoor"], len(indoor), rng)):
        r["lighting"] = l

    # environment motion: per stratum, exact still share, elements allowed in that stratum
    still = axes["shares"]["still_env"]
    for stratum, where in ((outdoor, "outdoor"), (indoor, "indoor")):
        allowed = [e["name"] for e in axes["env_motion"] if e["where"] in ("both", where)]
        n_still = int(round(still * len(stratum)))
        deck = ["still"] * n_still + _deck(allowed, len(stratum) - n_still, rng)
        rng.shuffle(deck)
        for r, e in zip(stratum, deck):
            r["env_motion"] = e

    # motion range: small venues never long_range; global shares preserved
    shares = axes["motion_ranges"]
    target = _count_targets(shares, N)
    long_sizes = set(axes.get("long_range_sizes", ["medium", "large"]))
    small = [r for r in rows if r["size"] not in long_sizes]
    big = [r for r in rows if r["size"] in long_sizes]
    small_w = {"in_place": shares["in_place"], "short_range": shares["short_range"]}
    sdeck = _weighted_deck(small_w, len(small), rng)
    used = Counter(sdeck)
    rest = {k: target[k] - used.get(k, 0) for k in RANGES}
    if min(rest.values()) < 0:
        raise ValueError("too few large venues for the requested range shares: %s" % rest)
    bdeck = [k for k, c in rest.items() for _ in range(c)]
    rng.shuffle(bdeck)
    for r, m in zip(small, sdeck):
        r["motion_range"] = m
    for r, m in zip(big, bdeck):
        r["motion_range"] = m

    # action family: within the range; ice/water surfaces only from ice_ok families; ice_only never elsewhere
    # action family: row-level eligibility, least-used family first (keeps counts balanced without strata)
    for r in rows:
        r["hard_ground"] = is_hard(r["ground"], axes)
    fams = axes["action_families"]
    order = list(range(N))
    rng.shuffle(order)
    used = Counter()
    for i in order:
        r = rows[i]
        pool = [f["name"] for f in fams if eligible(f, r)]
        if not pool:
            raise ValueError("no action family fits card %s (%s / %s / %s / %s / %d people)" % (
                r["id"], r["motion_range"], r["archetype"], r["surface"], r["size"], r["person_count"]))
        m = min(used[n] for n in pool)
        r["action_family"] = rng.choice([n for n in pool if used[n] == m])
        used[r["action_family"]] += 1

    for r, c in zip(rows, _deck(axes["cameras"], N, rng)):
        r["camera"] = c
    for r, v in zip(rows, _deck(axes["vibes"], N, rng)):
        r["vibe"] = v

    # flags
    for r in rows:
        r["crowd"] = False
        r["close_approach"] = False
    cand = [r for r in rows if r["audience_natural"]]
    k = min(int(round(axes["shares"]["crowd"] * N)), len(cand))
    for r in rng.sample(cand, k):
        r["crowd"] = True
    cand = [r for r in rows if r["motion_range"] != "in_place"]
    k = min(int(round(axes["shares"]["close_approach"] * N)), len(cand))
    for r in rng.sample(cand, k):
        r["close_approach"] = True
    suffixes = tuple(axes.get("holdout_suffixes", []))
    for r in rows:
        r["holdout"] = r["id"].endswith(suffixes) if suffixes else False
    return rows


def is_hard(ground, axes):
    """True only for surfaces a basketball bounces on."""
    g = ground.lower()
    if any(k in g for k in axes.get("no_bounce_keywords", [])):
        return False
    return any(k in g for k in axes.get("bounce_ground_keywords", []))


def eligible(f, row):
    """Can action family f be dealt onto this card? Every rule the pre-check taught us lives here."""
    if row["motion_range"] not in f["ranges"]:
        return False
    if row["person_count"] < f.get("min_people", 1):
        return False
    surface = row["surface"]
    if surface == "ice":
        if not (f.get("ice_only") or f.get("ice_ok")):
            return False
    elif f.get("ice_only"):
        return False
    if surface in ("ice", "water") and (f.get("needs_hard_surface") or f.get("dry_only")) and not f.get("ice_ok"):
        return False
    if f.get("large_only") and row["size"] != "large":
        return False
    if f.get("needs_hard_surface") and not row.get("hard_ground"):
        return False
    small = row["size"] == "small"
    if small and not f.get("small_ok"):
        return False
    if f.get("needs_space") and small:
        return False
    if f.get("outdoor_only") and row["indoor"]:
        return False
    if f.get("needs_headroom") and row["indoor"] and (row["size"] != "large" or row.get("low_ceiling")):
        return False
    words = set(re.findall(r"[a-z]+", row["ground"].lower()))
    ga, sa = f.get("ground_any"), f.get("surface_any")
    if ga or sa:
        if not ((ga and any(k in words for k in ga)) or (sa and surface in sa)):
            return False
    if f.get("ground_none") and any(k in words for k in f["ground_none"]):
        return False
    an = row["archetype"].lower()
    if f.get("archetype_any") and not any(k in an for k in f["archetype_any"]):
        return False
    if f.get("archetype_none") and any(k in an for k in f["archetype_none"]):
        return False
    return True


def _count_targets(shares, n):
    d = _weighted_deck(shares, n, random.Random(0))
    return dict(Counter(d))


def redeal_row(row, axes, rng, clear_crowd=False):
    """Re-draw the style axes of one card (used after the plausibility pre-check)."""
    if row["indoor"]:
        row["lighting"] = rng.choice(axes["lighting_indoor"])
        allowed = [e["name"] for e in axes["env_motion"] if e["where"] in ("both", "indoor")]
    else:
        row["lighting"] = rng.choice([w["name"] for w in axes["weather_outdoor"]])
        allowed = [e["name"] for e in axes["env_motion"] if e["where"] in ("both", "outdoor")]
    row["env_motion"] = "still" if rng.random() < axes["shares"]["still_env"] else rng.choice(allowed)
    ranges = RANGES if row["size"] in set(axes.get("long_range_sizes", ["medium", "large"])) else ["in_place", "short_range"]
    row["motion_range"] = rng.choice(ranges)
    row["hard_ground"] = is_hard(row["ground"], axes)
    pool = [f["name"] for f in axes["action_families"] if eligible(f, row)]
    if not pool:
        row["motion_range"] = "in_place"
        pool = [f["name"] for f in axes["action_families"] if eligible(f, row)]
    row["action_family"] = rng.choice(pool)
    row["camera"] = rng.choice(axes["cameras"])
    if row["motion_range"] == "in_place":
        row["close_approach"] = False
    if clear_crowd or not row["audience_natural"]:
        row["crowd"] = False
    return row


def report(rows, axes):
    N = len(rows)
    L = ["# Seeds report", "", "Total rows: %d" % N, ""]

    def hist(title, key, fmt=None):
        c = Counter(r[key] for r in rows)
        L.append("## " + title)
        for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            L.append("- %s: %d (%.1f%%)" % (fmt(k) if fmt else k, v, 100.0 * v / N))
        L.append("")

    pairs = Counter((r["region"], r["archetype"]) for r in rows)
    L.append("## Coverage checks")
    L.append("- distinct (region, archetype) pairs: %d of %d rows -> %s" % (len(pairs), N, "OK" if len(pairs) == N else "DUPLICATES"))
    rc = Counter(r["region"] for r in rows)
    L.append("- rows per region: min %d, max %d over %d regions" % (min(rc.values()), max(rc.values()), len(rc)))
    ac = Counter(r["archetype"] for r in rows)
    L.append("- rows per archetype: min %d, max %d over %d archetypes" % (min(ac.values()), max(ac.values()), len(ac)))
    long_sizes = set(axes.get("long_range_sizes", ["medium", "large"]))
    small_long = sum(1 for r in rows if r["size"] not in long_sizes and r["motion_range"] == "long_range")
    L.append("- long_range outside %s venues: %d (must be 0)" % ("/".join(sorted(long_sizes)), small_long))
    fam_by = {f["name"]: f for f in axes["action_families"]}
    bad_fam = sum(1 for r in rows if not eligible(fam_by[r["action_family"]], r))
    L.append("- rows whose action family is not eligible for the card: %d (must be 0)" % bad_fam)
    fc = Counter(r["action_family"] for r in rows)
    L.append("- action families used: %d of %d; per-family min %d, max %d" % (len(fc), len(fam_by), min(fc.values()), max(fc.values())))
    L.append("- small venues with more than %d people: %d (must be 0)" % (axes.get("small_venue_max_people", 3), sum(1 for r in rows if r["size"] == "small" and r["person_count"] > axes.get("small_venue_max_people", 3))))
    bad_crowd = sum(1 for r in rows if r["crowd"] and not r["audience_natural"])
    L.append("- crowd flags outside audience-natural venues: %d (must be 0)" % bad_crowd)
    hostile = set(w["name"] for w in axes["weather_outdoor"] if w["hostile"])
    outdoor = [r for r in rows if not r["indoor"]]
    L.append("- outdoor rows with reconstruction-hostile weather: %d of %d outdoor (%.1f%%)" % (
        sum(1 for r in outdoor if r["lighting"] in hostile), len(outdoor), 100.0 * sum(1 for r in outdoor if r["lighting"] in hostile) / max(1, len(outdoor))))
    refl = sum(1 for r in rows if r["surface"] != "normal")
    L.append("- reflective / ice / water surfaces: %d (%.1f%%)" % (refl, 100.0 * refl / N))
    L.append("- crowd=true: %d; close_approach=true: %d; holdout: %d" % (
        sum(r["crowd"] for r in rows), sum(r["close_approach"] for r in rows), sum(r["holdout"] for r in rows)))
    L.append("")
    hist("Person count", "person_count")
    hist("Motion range", "motion_range")
    hist("Environment motion", "env_motion")
    hist("Lighting / weather", "lighting")
    hist("Action family", "action_family")
    hist("Camera", "camera")
    hist("Vibe", "vibe")
    hist("Venue size", "size")
    hist("Indoor", "indoor", fmt=lambda b: "indoor" if b else "outdoor")
    hist("Surface", "surface")
    return "\n".join(L)


def _ice_ok(fam, axes):
    for f in axes["action_families"]:
        if f["name"] == fam:
            return bool(f["ice_ok"])
    return False


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
