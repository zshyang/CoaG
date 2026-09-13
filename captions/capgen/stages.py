"""Pipeline stages: tag -> sample -> precheck -> write -> validate -> repair -> review.
Each stage reads/writes files under out/ and is safe to re-run (LLM calls are cached)."""
import json
import os
import random
import re
from collections import Counter, defaultdict

from . import prompts as P
from .sampler import deal, redeal_row, report, write_jsonl, read_jsonl

OUT = "out"
FORBIDDEN = re.compile(r"\b(montage|cut to|cuts to|jump cut|transitions? to|cross-?fade|child|children|kid|kids|toddler|dog|dogs|cat|cats|horse|horses)\b", re.I)
CONTINUOUS = re.compile(r"(continuous|single|unbroken|one) (take|shot)|one continuous", re.I)
STOP = set("a an the of in on at with and by to its from under over near into for as is are after before through across beside behind along while their his her they them it this that".split())


def p(name):
    return os.path.join(OUT, name)


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


# ------------------------------------------------------------------ tag
def stage_tag(llm, archetypes, effort="high"):
    names = [a["name"] for a in archetypes]
    out = llm.call(P.TAG_SYSTEM, "Archetypes (one per line):\n" + "\n".join(names), P.TAG_SCHEMA, effort, tag="tag")
    tags = {t["name"]: t for t in out["tags"]}
    missing = [n for n in names if n not in tags]
    if missing:
        raise RuntimeError("tagger skipped %d names, e.g. %s" % (len(missing), missing[:3]))
    result = [tags[n] for n in names]
    with open("tags.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    c = Counter(); 
    for t in result:
        c["indoor" if t["indoor"] else "outdoor"] += 1; c["size:" + t["size"]] += 1
        c["audience" if t["audience_natural"] else "no-audience"] += 1; c["surface:" + t["surface"]] += 1
    print("[tag] wrote tags.json:", dict(c))
    return result


# --------------------------------------------------------------- sample
def stage_sample(axes, archetypes, tags, seed):
    rows = deal(axes, archetypes, tags, seed)
    write_jsonl(rows, p("seeds.jsonl"))
    rep = report(rows, axes) + "\n\nseed: %d\n" % seed
    with open(p("seeds_report.md"), "w") as f:
        f.write(rep)
    print("[sample] wrote %d seeds -> out/seeds.jsonl, out/seeds_report.md" % len(rows))
    return rows


# ------------------------------------------------------------- precheck
def stage_precheck(llm, axes, rows, seed, rounds=2, batch=100, effort="high"):
    log = []
    for rnd in range(1, rounds + 1):
        jobs = [("precheck-r%d-%02d" % (rnd, i), P.PRECHECK_SYSTEM, P.cards_text(b), P.PRECHECK_SCHEMA, effort)
                for i, b in enumerate(chunks(rows, batch))]
        res = llm.map(jobs)
        flags = {}
        for out in res.values():
            for fl in (out or {}).get("flags", []):
                flags[fl["id"]] = fl["reason"]
        log.append("round %d: %d flagged" % (rnd, len(flags)))
        for fid, reason in sorted(flags.items()):
            log.append("- %s: %s" % (fid, reason))
        if not flags:
            break
        rng = random.Random(seed * 1000 + rnd)
        by_id = {r["id"]: r for r in rows}
        for fid, reason in flags.items():
            if fid in by_id:
                redeal_row(by_id[fid], axes, rng, clear_crowd=("crowd" in reason.lower() or "audience" in reason.lower()))
        rows = [by_id[r["id"]] for r in rows]
    write_jsonl(rows, p("seeds.jsonl"))
    with open(p("seeds_report.md"), "a") as f:
        f.write("\n## Plausibility pre-check\n" + "\n".join(log) + "\n\n" + report(rows, axes).split("## Person count")[0].replace("# Seeds report", "### Coverage after pre-check"))
    print("[precheck]", " | ".join(l for l in log if l.startswith("round")))
    return rows


# ---------------------------------------------------------------- write
def stage_write(llm, rows, batch=40, effort="medium", extra_note="", tagprefix="write"):
    jobs = []
    for i, b in enumerate(chunks(rows, batch)):
        user = "Write exactly one caption per card (%d cards). Return all ids verbatim.\n%s" % (len(b), P.cards_text(b))
        jobs.append(("%s-%03d" % (tagprefix, i), P.WRITER_SYSTEM + extra_note, user, P.WRITER_SCHEMA, effort))
    res = llm.map(jobs)
    outs = {}
    for out in res.values():
        for c in (out or {}).get("captions", []):
            outs[c["id"]] = {"scene": c["scene"].strip(), "action": c["action"].strip(), "caption": c["caption"].strip()}
    return outs


def write_raw(rows, outs, path):
    with open(path, "w") as f:
        for r in rows:
            if r["id"] in outs:
                f.write(json.dumps(dict(r, **outs[r["id"]]), ensure_ascii=False, sort_keys=True) + "\n")


# ------------------------------------------------------------- validate
def mechanical_issue(row, out):
    cap = out["caption"]
    n = len(cap.split())
    if n < 60 or n > 125:
        return "word count %d outside 60-125" % n
    m = FORBIDDEN.search(cap)
    if m:
        return "forbidden word: %s" % m.group(0)
    if not CONTINUOUS.search(cap):
        return "missing continuous-shot phrase"
    if len(out["scene"].split()) > 24:
        return "scene summary longer than 20 words"
    fam = row.get("action_family", "")
    if fam and jaccard(_action_tokens(out["action"]), _action_tokens(fam)) >= 0.7:
        return "action restates the family name; write a specific sibling move, variation or combination"
    return None


_COUNT_WORDS = set("one two three four five six adults adult people dancers dancer man woman men women person performers athletes players fighters".split())


def _action_tokens(s):
    return set(w for w in tokens(s) if w not in _COUNT_WORDS)


def tokens(s):
    return set(w for w in re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split() if w and w not in STOP)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / float(len(a) + len(b) - i)


def near_duplicates(rows, outs, scene_thr=0.6, action_thr=0.45):
    """Global check across all accepted rows. Returns {id: reason} for the later member of each pair."""
    ids = [r["id"] for r in rows if r["id"] in outs]
    by_id = {r["id"]: r for r in rows}
    sc = {i: tokens(outs[i]["scene"]) for i in ids}
    ac = {i: tokens(outs[i]["caption"]) for i in ids}   # action check runs on the full caption, within a family
    ct = {i: tuple(sorted(ac[i])) for i in ids}
    rej = {}
    seen_cap = {}
    for a in ids:
        if ct[a] in seen_cap:
            rej[a] = "caption identical to %s" % seen_cap[ct[a]]
            continue
        seen_cap[ct[a]] = a
    fam = defaultdict(list)
    kept = [i for i in ids if i not in rej]
    for k, a in enumerate(kept):
        for b in kept[:k]:
            if b in rej:
                continue
            if jaccard(sc[a], sc[b]) > scene_thr:
                rej[a] = 'scene too close to %s: "%s"' % (b, outs[b]["scene"])
                break
            if by_id[a]["action_family"] == by_id[b]["action_family"] and jaccard(ac[a], ac[b]) > action_thr:
                rej[a] = 'caption too close to %s (same action family): "%s"' % (b, outs[b]["action"])
                break
    return rej


def stage_validate(llm, rows, outs, batch=40, effort="high", tagprefix="check"):
    """Returns (accepted_outs, rejects{id: reason})."""
    rows_with = [r for r in rows if r["id"] in outs]
    rejects = {r["id"]: "caption was not generated" for r in rows if r["id"] not in outs}
    accepted = {}
    # mechanical first (free)
    pending = []
    for r in rows_with:
        iss = mechanical_issue(r, outs[r["id"]])
        if iss:
            rejects[r["id"]] = iss
        else:
            pending.append(r)
    jobs = []
    for i, b in enumerate(chunks(pending, batch)):
        pairs = "\n".join(json.dumps({"card": P.card(r), "output": outs[r["id"]]}, ensure_ascii=False) for r in b)
        jobs.append(("%s-%03d" % (tagprefix, i), P.VALIDATOR_SYSTEM, "Pairs:\n" + pairs, P.VALIDATOR_SCHEMA, effort))
    res = llm.map(jobs)
    verdicts = {}
    for out in res.values():
        for v in (out or {}).get("results", []):
            verdicts[v["id"]] = v
    for r in pending:
        v = verdicts.get(r["id"])
        o = dict(outs[r["id"]])
        if v is None:
            rejects[r["id"]] = "validator returned no verdict"
        elif v["verdict"] == "ok":
            accepted[r["id"]] = o
        elif v["verdict"] == "fixed" and v.get("caption"):
            o["caption"] = v["caption"].strip()
            if v.get("scene"):
                o["scene"] = v["scene"].strip()
            if v.get("action"):
                o["action"] = v["action"].strip()
            iss = mechanical_issue(r, o)
            if iss:
                rejects[r["id"]] = "after fix: " + iss
            else:
                accepted[r["id"]] = o
        else:
            rejects[r["id"]] = v.get("issue") or "rejected by validator"
    for rid, why in near_duplicates(rows, accepted).items():
        rejects[rid] = why
        accepted.pop(rid, None)
    return accepted, rejects


# --------------------------------------------------------------- repair
def stage_repair(llm, rows, accepted, rejects, rounds=3, batch=40, effort="high"):
    by_id = {r["id"]: r for r in rows}
    history = []
    for rnd in range(1, rounds + 1):
        if not rejects:
            break
        todo = [dict(by_id[i], why_failed=w) for i, w in sorted(rejects.items())]
        cards = [dict(P.card(r), why_failed=r["why_failed"]) for r in todo]
        outs = {}
        jobs = []
        for i, b in enumerate(chunks(cards, batch)):
            user = "Rewrite exactly one caption per card (%d cards). Return all ids verbatim.\n%s" % (
                len(b), "\n".join(json.dumps(c, ensure_ascii=False) for c in b))
            jobs.append(("repair-r%d-%03d" % (rnd, i), P.WRITER_SYSTEM + P.REPAIR_NOTE, user, P.WRITER_SCHEMA, effort))
        res = llm.map(jobs)
        for out in res.values():
            for c in (out or {}).get("captions", []):
                outs[c["id"]] = {"scene": c["scene"].strip(), "action": c["action"].strip(), "caption": c["caption"].strip()}
        acc2, rej2 = stage_validate(llm, [by_id[i] for i in rejects], outs, batch=batch, effort=effort,
                                    tagprefix="recheck-r%d" % rnd)
        # global dedup of the newly accepted against everything already accepted
        merged = dict(accepted)
        merged.update(acc2)
        dup = near_duplicates(rows, merged)
        for rid, why in dup.items():
            if rid in acc2:
                rej2[rid] = why
                acc2.pop(rid)
        accepted.update(acc2)
        history.append("repair round %d: %d in, %d accepted, %d still rejected" % (rnd, len(todo), len(acc2), len(rej2)))
        print("[repair]", history[-1])
        rejects = rej2
    return accepted, rejects, history


# --------------------------------------------------------------- review
def stage_review(llm, rows, accepted, rejects, history, every=100, effort="low"):
    ids = [r["id"] for r in rows if r["id"] in accepted][::every][:20]
    items = "\n".join(json.dumps({"id": i, "caption": accepted[i]["caption"]}) for i in ids)
    out = llm.call(P.REVIEW_SYSTEM, items, P.REVIEW_SCHEMA, effort, tag="review")
    zh = {s["id"]: s["zh"] for s in out["summaries"]}
    by_id = {r["id"]: r for r in rows}
    L = ["# Review sample (every %dth accepted caption)" % every, ""]
    for i in ids:
        L.append("- %s · %s" % (i, zh.get(i, "")))
        L.append("  %s" % accepted[i]["caption"])
    L += ["", "## Totals", "- accepted: %d / %d" % (len(accepted), len(rows)), "- still rejected: %d" % len(rejects)]
    L += ["- " + h for h in history]
    if rejects:
        L += ["", "## Unresolved"] + ["- %s: %s" % (k, v) for k, v in sorted(rejects.items())]
    with open(p("review.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("[review] wrote out/review.md; accepted %d/%d" % (len(accepted), len(rows)))
