# captions/ — caption generation for the control-video training set

Design and decisions: `CAPTION_DESIGN.md` (read first). Vocabularies: `axes.json`, `archetypes.json`. Claude-produced archetype tags: `tags.json`.

## Run

```bash
cd code/wan_control_demo/captions
python3 tests/test_sampler.py                 # sampler invariants, no LLM
python3 -m capgen tag                         # once: Claude tags the 169 archetypes -> tags.json
python3 -m capgen sample --seed 20260906      # deal 2000 seed cards -> out/seeds.jsonl + out/seeds_report.md
python3 -m capgen precheck                    # Claude flags implausible cards; flagged ones are re-dealt
python3 -m capgen write     --workers 4       # 50 batches x 40 cards -> out/captions_raw.jsonl
python3 -m capgen validate                    # rules + mechanical checks + global dedup -> out/captions.jsonl, out/rejects.json
python3 -m capgen repair                      # up to 3 rewrite rounds for rejects
python3 -m capgen review                      # out/review.md: 20 samples with Chinese one-liners + totals
```

Backends: `--backend claude-code` (default; local Claude Code CLI in print mode, Claude Max subscription, subject to usage windows) or `--backend api` (official `anthropic` SDK; needs Python >= 3.10 and an API key; run via `uv run --python 3.11 --with anthropic python -m capgen ...`).

Every LLM call is cached in `out/cache/` keyed by its full input, so any stage can be re-run after an interruption and only the missing calls are sent. If the usage window runs out the command exits with code 2; run it again later.

Do not run `all` after `precheck`: `sample` would re-deal and overwrite the pre-checked seeds. Run the stages one by one as above.

## Outputs

- `out/seeds.jsonl` — 2000 cards; `id` doubles as the video-generation seed.
- `out/captions.jsonl` — accepted captions with all card fields plus `scene`, `action`, `caption`; rows with `holdout: true` (ids ending in 49/99) are never used for training.
- `out/rejects.json` — anything still failing after repair, with the reason.
- `out/review.md` — the 20-sample review George approves before any video generation.
