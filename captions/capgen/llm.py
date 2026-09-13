"""One LLM entry point with two interchangeable backends.

claude-code : shells out to the local Claude Code CLI in print mode (uses the Claude Max subscription).
api         : official `anthropic` Python SDK (needs ANTHROPIC_API_KEY or `ant auth login`; Python >= 3.10).

Every call is cached on disk by a hash of (backend, model, system, user, schema, effort), so a
re-run after an interruption only sends what is missing.
"""
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class UsageLimit(RuntimeError):
    """The subscription usage window is exhausted; stop submitting and resume later."""


class LLM:
    def __init__(self, backend="claude-code", model="claude-opus-5", cache_dir="out/cache", workers=4,
                 timeout=1800, verbose=True):
        self.backend = backend
        self.model = model
        self.cache_dir = cache_dir
        self.workers = workers
        self.timeout = timeout
        self.verbose = verbose
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "cache_read": 0}
        os.makedirs(cache_dir, exist_ok=True)
        if backend == "api":
            import anthropic  # noqa: F401  (import error here is the right failure)
            self._client = anthropic.Anthropic()

    # ---------------------------------------------------------------- public
    def call(self, system, user, schema, effort="medium", tag=""):
        key = hashlib.sha256(json.dumps([self.backend, self.model, system, user, schema, effort],
                                        sort_keys=True).encode()).hexdigest()[:24]
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            with self._lock:
                self.stats["cached"] += 1
            return rec["output"]
        t0 = time.time()
        if self.backend == "claude-code":
            output, usage = self._call_claude_code(system, user, schema, effort)
        elif self.backend == "api":
            output, usage = self._call_api(system, user, schema, effort)
        else:
            raise ValueError("unknown backend " + self.backend)
        rec = {"tag": tag, "backend": self.backend, "model": self.model, "effort": effort,
               "usage": usage, "seconds": round(time.time() - t0, 1), "output": output}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, path)
        with self._lock:
            self.stats["calls"] += 1
            self.stats["input_tokens"] += usage.get("input_tokens", 0)
            self.stats["output_tokens"] += usage.get("output_tokens", 0)
            self.stats["cache_read"] += usage.get("cache_read_input_tokens", 0)
        if self.verbose:
            print("[llm] %s done in %ss (out %s tok)" % (tag or key, rec["seconds"], usage.get("output_tokens")), flush=True)
        return output

    def map(self, jobs):
        """jobs: list of (tag, system, user, schema, effort). Returns {tag: output or None}.
        Stops submitting new work on UsageLimit; finished work is already cached."""
        results = {}
        limit_hit = [False]

        def run(job):
            tag, system, user, schema, effort = job
            try:
                return tag, self.call(system, user, schema, effort, tag=tag)
            except UsageLimit as e:
                limit_hit[0] = True
                print("[llm] USAGE LIMIT on %s: %s" % (tag, e), flush=True)
                return tag, None
            except Exception as e:  # keep the batch going; the caller sees None
                print("[llm] FAILED %s: %s" % (tag, e), flush=True)
                return tag, None

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = []
            for job in jobs:
                if limit_hit[0]:
                    break
                futs.append(ex.submit(run, job))
            for fut in as_completed(futs):
                tag, out = fut.result()
                results[tag] = out
        if limit_hit[0]:
            raise UsageLimit("usage window exhausted; re-run the same command later to resume")
        return results

    # -------------------------------------------------------------- backends
    def _call_claude_code(self, system, user, schema, effort):
        cmd = ["claude", "-p", "--tools", "", "--disable-slash-commands", "--no-session-persistence",
               "--effort", effort, "--model", self.model, "--system-prompt", system,
               "--output-format", "json", "--json-schema", json.dumps(schema)]
        last_err = None
        for attempt in range(3):
            try:
                proc = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=self.timeout,
                                      cwd=self.cache_dir)
            except subprocess.TimeoutExpired:
                last_err = "timeout after %ss" % self.timeout
                continue
            try:
                d = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_err = "non-JSON stdout: %s | stderr: %s" % (proc.stdout[:300], proc.stderr[:300])
                time.sleep(5 * (attempt + 1))
                continue
            if d.get("is_error"):
                msg = str(d.get("result", ""))
                if "limit" in msg.lower() or "rate" in msg.lower():
                    raise UsageLimit(msg)
                last_err = msg
                time.sleep(10 * (attempt + 1))
                continue
            out = d.get("structured_output")
            if out is None:
                try:
                    out = json.loads(d.get("result", ""))
                except Exception:
                    last_err = "no structured_output; result=%s" % str(d.get("result"))[:300]
                    continue
            u = d.get("usage", {})
            usage = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                     "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                     "output_tokens": u.get("output_tokens", 0), "cost_usd_list": d.get("total_cost_usd")}
            return out, usage
        raise RuntimeError("claude-code backend failed: %s" % last_err)

    def _call_api(self, system, user, schema, effort):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}, "effort": effort},
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("refusal: %s" % getattr(resp, "stop_details", None))
        text = next(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "output_tokens": u.output_tokens}
        return json.loads(text), usage
