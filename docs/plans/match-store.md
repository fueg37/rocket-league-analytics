# Plan: Match Store & Unified Ingest

Status: proposed · Owner: Fueg · Last updated: 2026-10-11

## Why

The app has two analytics pipelines that have already drifted apart:

| | Single match (`_compute_match_analytics`, `app.py:2439`) | Season batch (inline, `app.py:4443-4520`) |
|---|---|---|
| Match ID | `proto.game_metadata.id` | `manager.game.id`, falls back to the filename |
| VAEP | uses the shared `replay_states` encoding | recomputes without `states=` |
| Momentum, win prob, coach report | computed | skipped |
| `Won`, `Timestamp`, `MatchID` | not computed | computed inline |
| Win-prob training rows | not extracted | extracted |
| Duplicate check | by filename, in session only | **after** the full parse |
| What gets persisted | nothing (in-session cache of 12 matches) | one aggregate row per player per match in `career_stats.csv` |

What this costs:

- **Events are thrown away.** Season mode can't do shot maps, heatmaps, or models built from your own shots and saves, because only aggregates are saved.
- **Metric changes never reach old matches.** Old CSV rows keep their old values, and new columns get filled with defaults (`app.py:4524-4537`).
- **Duplicate replays still get parsed.** Each duplicate pays the full parse cost before it's skipped.
- **Every new replay source needs its own ingest code.** After the anti-cheat update we need several sources: manual upload, the Demos folder, later ballchasing and the Stats API.

## The design in one sentence

**A match is computed once by one pipeline, saved as a versioned record of all its event and summary tables, and both app modes are just views that query those records.**

```
Sources ─► Ingest ─► Pipeline ─► Store ─► Views
upload      hash dedupe   one compute fn   plain files:     Single match
Demos dir   parse         (versioned)      manifest.json +  Season
ballchasing                                parquet tables   (later: pages/)
Stats API
```

Rules this design depends on:

1. **The replay file is the source of truth; everything computed from it is a cache.** If a computed table is lost or out of date, it gets recomputed from the replay, never repaired by hand.
2. **One compute function.** Batch ingest is a loop over the same function single-match uses.
3. **Duplicates are rejected before parsing.** Files are identified by the SHA-256 of their bytes (instant). Matches are identified by replay ID (known after parsing). The same match arriving from two sources maps to one record.
4. **Every record stores the `ANALYTICS_VERSION` that produced it.** If it differs from the current version, the record is stale and can be recomputed.
5. **Views never call `calculate_*` functions.** They only read from the store. The one exception is frame-level views (see Phase 4).
6. **No database; the folder layout is the store.** Every fact lives in exactly one file, and anything that looks like an index is computed from those files, never maintained alongside them. There's nothing to get out of sync, and you can inspect, back up or delete data with a file explorer.

## Storage layout

```
data/                              # gitignored; path configurable in settings
  replays/
    <sha256>.replay                # raw file, named by content hash; its existence = "seen"
    <sha256>.error.txt             # present only if parse/compute failed (traceback + source)
  matches/
    <match_id>/
      manifest.json                # match metadata + players + version (see below)
      players.parquet              # per-player summary (today's calculate_final_stats rows + Won/Overtime/Luck)
      shots.parquet                # SHOT_EVENT_COLUMNS
      saves.parquet                # CANONICAL_EVENT_COLUMNS
      passes.parquet
      kickoffs.parquet
      aerials.parquet  recoveries.parquet  defense.parquet  xga.parquet
      vaep.parquet  vaep_summary.parquet  value_reports.parquet
      rotation_timeline.parquet  rotation_summary.parquet  double_commits.parquet
      situational.parquet  momentum.parquet  win_prob.parquet  coach_report.parquet
      wp_training.parquet          # rows from extract_win_prob_training_data
```

**State comes from the files themselves**, so there is no ledger to maintain:

| What's on disk for a replay hash | Meaning |
|---|---|
| `<sha256>.replay` + a manifest whose `sha256` matches | ingested OK |
| `<sha256>.replay` + `<sha256>.error.txt` | failed (shown in the UI with the error; "Retry" deletes the `.txt`) |
| `<sha256>.replay` alone | interrupted mid-ingest, so the next sync retries it automatically |

**Writes are atomic per match.** `write_match` writes everything into `matches/.tmp-<match_id>/` and then renames it into place (if a previous version exists, it's renamed aside first and deleted afterwards). A crash can never leave a half-written match visible to the views.

**Parquet, not CSV,** because it keeps column types (bools, ints, NaN) across a save and reload, and each match's files carry whatever columns the pipeline produced at that version, so there are no schema migrations. A season query is a concat over the selected matches' files. That's fast enough for hundreds to low thousands of matches.

### `manifest.json`

```json
{
  "match_id": "8F3A...",
  "sha256": "c1d2...",                  // null for legacy CSV imports
  "sources": [{"source": "demos_dir", "ref": "8F3A....replay", "seen_at": "2026-10-11T19:02:11"}],
  "played_at": "2026-10-10T21:14:03",   // get_match_timestamp / extract_replay_date
  "playlist": "Ranked Doubles",
  "score": {"Blue": 3, "Orange": 2},
  "overtime": false,
  "fidelity": "replay",                 // 'replay' | 'legacy_summary' | (later) 'stats_api'
  "analytics_version": "2026.10.1",     // null for legacy imports
  "computed_at": "2026-10-11T19:02:15",
  "players": [
    {"key": "steam:7656119...", "name": "Fueg", "team": "Blue", "is_bot": false}
  ],
  "tables": ["players", "shots", "saves", "passes", "..."]
}
```

**The match index** (`list_matches()`) reads every `manifest.json` (tiny files; a few hundred take milliseconds) into one DataFrame. It's wrapped in `st.cache_data`, keyed on the `matches/` folder's modification time, so it only reloads after an ingest. Filters like "matches with my squad" or "stale versions" are pandas filters on that DataFrame.

## Module layout

```
store/
  __init__.py
  layout.py      # path helpers: replay_path(sha), match_dir(id), error_path(sha)
  writer.py      # save_replay(bytes) -> sha, write_match(record) (atomic), mark_failed(sha, err)
  reader.py      # list_matches() -> DataFrame, load_match(id), load_table(name, match_ids)
ingest/
  __init__.py
  sources.py     # ReplayCandidate + UploadSource, DemosFolderSource
  run.py         # ingest(candidates, compute_fn, data_dir) -> IngestReport
analytics/
  version.py     # ANALYTICS_VERSION = "2026.10.1"
settings.py      # loads settings.toml: data_dir, demos_dir, identity
```

Core types:

```python
@dataclass(frozen=True)
class ReplayCandidate:
    source: str            # 'upload' | 'demos_dir' | ...
    source_ref: str        # filename / path / remote id
    read_bytes: Callable[[], bytes]

@dataclass(frozen=True)
class MatchRecord:
    manifest: dict                    # becomes manifest.json
    tables: dict[str, pd.DataFrame]   # name -> table, written as parquet
```

`ingest.run.ingest()` does: read bytes → hash → skip if `replays/<sha>.replay` exists and isn't in the "interrupted" state → save the file → parse → `compute_fn(...) -> MatchRecord` → `write_match` (or `mark_failed`). A failure in one file is recorded as its `.error.txt` and never stops the rest of the batch. If the parsed `match_id` already has a manifest from a different file (the same match from another source), the new source is appended to `sources` and nothing is recomputed. Calling ingest again on the same folder does nothing new.

## Identity & settings

`settings.toml` (gitignored, with a checked-in `settings.example.toml`):

```toml
data_dir  = "data"
demos_dir = "~/Documents/My Games/Rocket League/TAGame/Demos"

[identity]
me    = "steam:7656119..."              # player_key
squad = ["steam:...", "epic:..."]       # regular teammates
```

This replaces the hardcoded `"Fueg"` / `"Zelli197"` (`app.py:3127`, `4551`). Views filter by `player_key`, so name changes no longer split your history. Until `identity.me` is set, the app shows a one-time picker built from `match_players`.

## Phases

Each phase can be merged on its own and leaves the app working.

### Phase 0: Spike to check assumptions (no product code)
- [ ] Save a replay from the current game version and confirm `carball` still parses it. **If it doesn't, stop and evaluate a parser replacement (e.g. `subtr-actor`) first.** Only the parse step is affected, because the pipeline's input is the parsed replay.
- [ ] Confirm `proto.game_metadata.id == manager.game.id` on a few replays. The match ID must stay the same across old and new code.
- [ ] Measure parse time and computed-table size per match to sanity-check the parquet-per-match layout.
- [ ] Add `carball` (and its pinned dependencies) to `requirements.txt`. It's missing today.

### Phase 1: Store
- [ ] `store/` package (layout, writer, reader); `pyarrow` added to requirements.
- [ ] Tests: write/read round-trip; rewriting a match replaces it completely (no leftover tables); a simulated crash mid-write leaves no visible partial match; the stale filter on `list_matches()` works; an interrupted replay is picked up again.

### Phase 2: One pipeline, one ingest path
- [ ] Extend `_compute_match_analytics` to return a `MatchRecord`. Move the batch-only work into it (`Won`, `Timestamp`, `MatchID` on player rows, win-prob training rows, kickoff `MatchID`) so nothing exists only in batch mode.
- [ ] `ingest/run.py` takes `compute_fn` as a parameter. The pipeline still lives in `app.py` until the page split; passing it in avoids a circular import and keeps ingest testable with a fake.
- [ ] Batch upload becomes `ingest([UploadSource(f) ...], compute_fn, store)`. **Delete the inline batch pipeline** (`app.py:4443-4520`).
- [ ] Win-prob model training reads `wp_training` across the store instead of only the current batch.
- [ ] Tests: ingesting the same file twice parses once; a broken file gets a `.error.txt` and the rest of the batch continues; the same match from two sources produces one match with two `sources`.

### Phase 3: Season mode reads the store
- [ ] Season dataframe = `load_table("players", match_ids)` joined with the `list_matches()` index. Remove the column-default backfill list.
- [ ] Session detection uses the manifest's `played_at`. Remove the "every 5 matches" fallback and the bare `except:`. Matches without a date sort last and are labelled as such.
- [ ] **Migration:** a one-time `migrate_legacy_csv()` imports `career_stats.csv` / `career_kickoffs.csv` as `fidelity='legacy_summary'` matches: a manifest plus `players.parquet` / `kickoffs.parquet`, with no replay, no events and a null version. Re-ingesting the real replay later replaces the legacy record with a full one because the `match_id` is the same. Views that need events quietly leave out legacy matches and show a count ("12 legacy matches have summary stats only").
- [ ] Remove `load_data` / `save_data` / `DB_FILE` / `KICKOFF_DB_FILE`.

### Phase 4: Single-match mode reads the store
- [ ] Match picker = `list_matches()` (all of history, not just the session). Upload still works and goes through `ingest`.
- [ ] Remove `match_store` / `match_order` / `_evict_oldest_match` / `MAX_STORED_MATCHES`; the store replaces them.
- [ ] **Frame-level views** (Shot Viewer, Tactical 3D, heatmaps, export panels) need `game_df`/`proto`, which are too big to store. They re-parse `replays/<sha256>.replay` on demand through the existing `@st.cache_resource` `get_parsed_replay_data`. Every other tab reads computed tables only.

### Phase 5: Demos-folder source
- [ ] `DemosFolderSource` lists `*.replay` in `demos_dir`; hashing makes repeat scans cheap.
- [ ] Sidebar shows "Sync replays (N new)", and new files are ingested on app start. An optional `watchdog` watcher can come later if wanted.
- [ ] Settings UI for `demos_dir` and identity.

### Phase 6: Recompute
- [ ] `ANALYTICS_VERSION` gets bumped in the same PR as any change to metric logic (add this rule to CLAUDE.md).
- [ ] Sidebar: "N matches computed with an older analytics version · Recompute". Uses the stored replay and the same ingest path, with progress shown.

## Later (enabled by this, not part of it)

- **Page split:** move the pipeline into `analytics/pipeline.py`, with `pages/match.py` and `pages/season.py` as thin views. Ingest then imports the pipeline directly and the `compute_fn` parameter goes away.
- **More sources:** ballchasing (`BallchasingSource` via API token) and a Stats API recorder (`fidelity='stats_api'`, upgraded automatically when the replay for the same match is ingested).
- **Models trained on your own data:** fit xG / expected saves from the season's `shots` / `saves` tables; trends with confidence intervals via `analytics/stats_uncertainty.py`.

## Risks

| Risk | Mitigation |
|---|---|
| carball can't parse post-anti-cheat replays | Phase 0 checks this first; the design isolates parsing |
| Replay ID differs between `proto` and `manager` | Phase 0 checks this; if it differs, choose one and remap legacy rows during migration |
| Parquet file count grows (~20 files per match) | Fine for thousands of matches; if it ever isn't, add a rebuildable season cache (one parquet per table, regenerated from the per-match files). It's still a derived cache, never the source of truth |
| Index scan slows as manifests grow | Cached until the next ingest; a few thousand small JSON files still load in well under a second |
| Re-parsing for frame-level views is slow | Cached per session; only the views that need frames pay for it |

## Open questions

1. Do you still have the original `.replay` files for the matches in `career_stats.csv`? If so, migration is just "ingest the folder" and legacy records are rare.
2. Is adding `pyarrow` acceptable? It's the only new required dependency, and it's what lets pandas read and write parquet.
3. Should bots be stored and filtered at view time (proposed), or dropped at ingest as batch mode does today?
