"""Invariant tests for the sampler. Run: python3 tests/test_sampler.py  (no pytest needed)."""
import json, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capgen.sampler import deal, report, RANGES, eligible, is_hard

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
axes = json.load(open(os.path.join(HERE, "axes.json")))
archetypes = json.load(open(os.path.join(HERE, "archetypes.json")))


def fake_tags(seed=1):
    rng = random.Random(seed)
    tags = []
    for a in archetypes:
        tags.append({"name": a["name"], "indoor": rng.random() < 0.3,
                     "size": rng.choice(["small", "medium", "medium", "large", "large", "large", "large"]),
                     "audience_natural": rng.random() < 0.4,
                     "surface": rng.choice(["normal"] * 6 + ["reflective", "ice", "water"]), "low_ceiling": rng.random() < 0.1})
    return tags


def check(rows, tags):
    N = axes["total"]
    tag_of = {t["name"]: t for t in tags}
    fam = {f["name"]: f for f in axes["action_families"]}
    assert len(rows) == N
    assert [r["id"] for r in rows] == ["c%04d" % i for i in range(N)]
    assert len(set((r["region"], r["archetype"]) for r in rows)) == N, "pairs must be unique"
    rc = Counter(r["region"] for r in rows)
    assert set(rc.values()) == {N // len(axes["regions"])}, rc
    ac = Counter(r["archetype"] for r in rows)
    assert max(ac.values()) - min(ac.values()) <= 1, "archetypes must be balanced"
    assert Counter(r["person_count"] for r in rows) == {int(k): v for k, v in axes["person_count_plan"].items()}
    mr = Counter(r["motion_range"] for r in rows)
    assert mr == {"in_place": 600, "short_range": 800, "long_range": 600}, mr
    assert not any(r["size"] not in axes["long_range_sizes"] and r["motion_range"] == "long_range" for r in rows)
    fc = Counter(r["action_family"] for r in rows)
    assert len(fc) >= 0.9 * len(fam), "most families should be used"
    assert sum(1 for r in rows if r["env_motion"] == "still") in (799, 800, 801)
    indoor_l = set(axes["lighting_indoor"]); outdoor_l = set(w["name"] for w in axes["weather_outdoor"])
    for r in rows:
        assert r["lighting"] in (indoor_l if r["indoor"] else outdoor_l)
        e = r["env_motion"]
        if e != "still":
            where = next(x["where"] for x in axes["env_motion"] if x["name"] == e)
            assert where == "both" or where == ("indoor" if r["indoor"] else "outdoor"), (r["indoor"], e)
        f = fam[r["action_family"]]
        assert eligible(f, r), (r["id"], r["action_family"])
        if r["size"] == "small":
            assert r["person_count"] <= 3
        if r["crowd"]:
            assert tag_of[r["archetype"]]["audience_natural"]
        if r["close_approach"]:
            assert r["motion_range"] != "in_place"
    assert sum(r["close_approach"] for r in rows) == 300
    assert sum(r["crowd"] for r in rows) <= 400
    assert sum(r["holdout"] for r in rows) == 40
    cam = Counter(r["camera"] for r in rows)
    assert set(cam.values()) == {N // len(axes["cameras"])}
    outdoor = [r for r in rows if not r["indoor"]]
    hostile = set(w["name"] for w in axes["weather_outdoor"] if w["hostile"])
    share = sum(1 for r in outdoor if r["lighting"] in hostile) / float(len(outdoor))
    assert abs(share - axes["shares"]["hostile_weather_cap"]) < 0.01, share


def main():
    tags = fake_tags()
    rows = deal(axes, archetypes, tags, seed=20260906)
    check(rows, tags)
    rows2 = deal(axes, archetypes, tags, seed=20260906)
    assert json.dumps(rows, sort_keys=True) == json.dumps(rows2, sort_keys=True), "must be deterministic"
    rows3 = deal(axes, archetypes, tags, seed=7)
    assert json.dumps(rows, sort_keys=True) != json.dumps(rows3, sort_keys=True), "seed must matter"
    for s in (2, 3, 4):
        t = fake_tags(s)
        check(deal(axes, archetypes, t, seed=s), t)
    rep = report(rows, axes)
    assert "DUPLICATES" not in rep and "(must be 0)" in rep
    print("sampler tests OK (%d rows, 4 tag sets, determinism)" % len(rows))


if __name__ == "__main__":
    main()
