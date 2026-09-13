#!/usr/bin/env python3
"""Generate raw clips from captions with Veo 3.1 Fast on fal.ai (no audio, 720p).

Usage:
  python3 gen_videos.py --key-file ~/path/with/fal/key --ids c0000-c0009           # pilot
  python3 gen_videos.py --key-file ... --all --concurrency 8                        # full run
Resumable: a clip whose .mp4 already exists is skipped. Each clip gets a sidecar .json
(prompt, seed, params, request id, timings). Never prints the key.

fal queue REST: POST https://queue.fal.run/<model>  -> {request_id, status_url, response_url}
                GET  status_url  until status == COMPLETED ; GET response_url -> {video: {url}}
Auth header: Authorization: Key <FAL_KEY>
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "fal-ai/veo3.1/fast"
QUEUE = "https://queue.fal.run/" + MODEL
PRICE_PER_SEC = 0.10  # 720p/1080p, no audio
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAPTIONS = os.path.join(ROOT, "captions", "out", "captions.jsonl")
OUT = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "runtime", "wan_control_demo", "raw_clips")  # workspace/runtime/
NEGATIVE = "on-screen text, subtitles, watermark, logo, captions, split screen"


def load_key(path):
    txt = open(os.path.expanduser(path)).read()
    m = re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:[0-9a-f]{32}\b", txt)
    if not m:
        sys.exit("no fal key (uuid:hex) found in " + path)
    return m.group(0)


def http(method, url, key, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Key " + key)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError("HTTP %d %s: %s" % (e.code, url, msg))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def generate(row, key, args):
    cid = row["id"]
    mp4 = os.path.join(OUT, cid + ".mp4")
    side = os.path.join(OUT, cid + ".json")
    if os.path.exists(mp4):
        return cid, "skip"
    seed = int(cid[1:])
    payload = {"prompt": row["caption"], "aspect_ratio": "16:9", "duration": args.duration,
               "resolution": args.resolution, "generate_audio": False, "seed": seed,
               "negative_prompt": NEGATIVE, "auto_fix": True}
    t0 = time.time()
    rec = {"id": cid, "model": MODEL, "params": payload, "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        sub = http("POST", QUEUE, key, payload)
        rec["request_id"] = sub.get("request_id")
        status_url, response_url = sub["status_url"], sub["response_url"]
        while True:
            st = http("GET", status_url, key)
            s = st.get("status")
            if s == "COMPLETED":
                break
            if s in ("FAILED", "CANCELLED", "ERROR"):
                raise RuntimeError("queue status %s: %s" % (s, json.dumps(st)[:300]))
            time.sleep(args.poll)
        res = http("GET", response_url, key)
        url = res["video"]["url"]
        urllib.request.urlretrieve(url, mp4 + ".part")
        os.replace(mp4 + ".part", mp4)
        rec.update({"video_url": url, "seconds": round(time.time() - t0, 1), "bytes": os.path.getsize(mp4),
                    "cost_usd": PRICE_PER_SEC * int(args.duration.rstrip("s")), "status": "ok"})
        with open(side, "w") as f:
            json.dump(rec, f, indent=1)
        return cid, "ok"
    except Exception as e:
        rec.update({"status": "error", "error": str(e)[:500], "seconds": round(time.time() - t0, 1)})
        with open(side, "w") as f:
            json.dump(rec, f, indent=1)
        return cid, "error: " + str(e)[:120]


def parse_ids(spec):
    m = re.match(r"^c(\d{4})-c(\d{4})$", spec)
    if m:
        return ["c%04d" % i for i in range(int(m.group(1)), int(m.group(2)) + 1)]
    return spec.split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--ids", help="c0000-c0009 or comma list")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-holdout", action="store_true", help="do not generate the 40 hold-out captions")
    ap.add_argument("--duration", default="6s", choices=["4s", "6s", "8s"])
    ap.add_argument("--resolution", default="720p", choices=["720p", "1080p"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--poll", type=float, default=6.0)
    a = ap.parse_args()
    key = load_key(a.key_file)
    rows = [json.loads(l) for l in open(CAPTIONS)]
    if a.ids:
        want = set(parse_ids(a.ids)); rows = [r for r in rows if r["id"] in want]
    elif not a.all:
        sys.exit("give --ids or --all")
    if a.skip_holdout:
        rows = [r for r in rows if not r.get("holdout")]
    os.makedirs(OUT, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(OUT, r["id"] + ".mp4"))]
    est = len(todo) * PRICE_PER_SEC * int(a.duration.rstrip("s"))
    print("%d clips requested, %d to generate, %s @ %s no audio -> est $%.2f" % (len(rows), len(todo), a.duration, a.resolution, est), flush=True)
    counts = {"ok": 0, "skip": len(rows) - len(todo), "error": 0}
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(generate, r, key, a) for r in todo]
        for fut in as_completed(futs):
            cid, st = fut.result()
            counts["ok" if st == "ok" else "skip" if st == "skip" else "error"] += 1
            print("[%s] %s" % (cid, st), flush=True)
    spent = sum(json.load(open(os.path.join(OUT, f))).get("cost_usd", 0) for f in os.listdir(OUT) if f.endswith(".json"))
    print("done:", counts, "| total spent so far (all clips on disk): $%.2f" % spent)


if __name__ == "__main__":
    main()
