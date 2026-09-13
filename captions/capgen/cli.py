"""python3 -m capgen <stage> [--backend claude-code|api] [--seed N] [--workers N]
stages: tag | sample | precheck | write | validate | repair | review | all"""
import argparse
import json
import os
import sys

from . import stages as S
from .llm import LLM, UsageLimit
from .sampler import read_jsonl, write_jsonl


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capgen")
    ap.add_argument("stage", choices=["tag", "sample", "precheck", "write", "validate", "repair", "review", "all"])
    ap.add_argument("--backend", default="claude-code", choices=["claude-code", "api"])
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N seeds (smoke test)")
    a = ap.parse_args(argv)

    os.makedirs(S.OUT, exist_ok=True)
    axes = load_json("axes.json")
    archetypes = load_json("archetypes.json")
    llm = LLM(backend=a.backend, model=a.model, workers=a.workers)
    st = a.stage
    try:
        if st in ("tag", "all"):
            if os.path.exists("tags.json") and st == "all":
                print("[tag] tags.json exists, keeping it")
            else:
                S.stage_tag(llm, archetypes)
        if st in ("sample", "all"):
            tags = load_json("tags.json")
            S.stage_sample(axes, archetypes, tags, a.seed)
        rows = read_jsonl(S.p("seeds.jsonl")) if os.path.exists(S.p("seeds.jsonl")) else None
        if st in ("precheck", "all"):
            rows = S.stage_precheck(llm, axes, rows, a.seed)
        if a.limit and rows:
            rows = rows[:a.limit]
        if st in ("write", "all"):
            outs = S.stage_write(llm, rows)
            S.write_raw(rows, outs, S.p("captions_raw.jsonl"))
            print("[write] %d/%d captions -> out/captions_raw.jsonl" % (len(outs), len(rows)))
        if st in ("validate", "all"):
            raw = read_jsonl(S.p("captions_raw.jsonl"))
            outs = {r["id"]: {"scene": r["scene"], "action": r["action"], "caption": r["caption"]} for r in raw}
            accepted, rejects = S.stage_validate(llm, rows, outs)
            _dump(rows, accepted, rejects, [])
            print("[validate] accepted %d, rejected %d" % (len(accepted), len(rejects)))
        if st in ("repair", "all"):
            accepted, rejects = _load_state(rows)
            accepted, rejects, history = S.stage_repair(llm, rows, accepted, rejects)
            _dump(rows, accepted, rejects, history)
        if st in ("review", "all"):
            accepted, rejects = _load_state(rows)
            history = []
            if os.path.exists(S.p("repair_history.json")):
                history = load_json(S.p("repair_history.json"))
            S.stage_review(llm, rows, accepted, rejects, history)
    except UsageLimit as e:
        print("\n[capgen] STOPPED: %s\nEverything finished so far is cached under out/cache; re-run the same command to continue." % e)
        sys.exit(2)
    finally:
        print("[llm] stats:", llm.stats)


def _dump(rows, accepted, rejects, history):
    with open(S.p("captions.jsonl"), "w") as f:
        for r in rows:
            if r["id"] in accepted:
                f.write(json.dumps(dict(r, **accepted[r["id"]]), ensure_ascii=False, sort_keys=True) + "\n")
    with open(S.p("rejects.json"), "w") as f:
        json.dump(rejects, f, ensure_ascii=False, indent=1, sort_keys=True)
    if history:
        with open(S.p("repair_history.json"), "w") as f:
            json.dump(history, f)


def _load_state(rows):
    acc = {r["id"]: {"scene": r["scene"], "action": r["action"], "caption": r["caption"]}
           for r in read_jsonl(S.p("captions.jsonl"))} if os.path.exists(S.p("captions.jsonl")) else {}
    rej = load_json(S.p("rejects.json")) if os.path.exists(S.p("rejects.json")) else {}
    return acc, rej


if __name__ == "__main__":
    main()
