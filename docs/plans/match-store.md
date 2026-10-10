# Plan: Match Store & Unified Ingest

Status: proposed · Owner: Fueg · Last updated: 2026-10-10

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
upload      hash dedupe   one compute fn   SQLite index     Single match
Demos dir   parse         (versioned)      + per-match      Season
ballchasing                                parquet tables   (later: pages/)
Stats API
```

Rules this design depends on:

1. **The replay file is the source of truth; everything computed from it is a cache.** If a computed table is lost or out of date, it gets recomputed from the replay, never repaired by hand.
2. **One compute function.** Batch ingest is a loop over the same function single-match uses.
3. **Duplicates are rejected before parsing.** Files are identified by the SHA-256 of their bytes (instant). Matches are identified by replay ID (known after parsing). The same match arriving from two sources maps to one record.
4. **Every record stores the `ANALYTICS_VERSION` that produced it.** If it differs from the current version, the record is stale and can be recomputed.
5. **Views never call `calculate_*` functions.** They only read from the store. The one exception is frame-level views (see Phase 4).

## Storage layout

```
data/                         # gitignored; path configurable in settings
  store.sqlite                # index + ledger (small, transactional)
  replays/<sha256>.replay     # raw files, content-addressed
  matches/<match_id>/         # derived tables, one parquet file per table
    players.parquet           # per-player summary (today's calculate_final_stats rows + Won/Overtime/Luck)
    shots.parquet             # SHOT_EVENT_COLUMNS
    saves.parquet             # CANONICAL_EVENT_COLUMNS
    passes.parquet
    kickoffs.parquet
    aerials.parquet  recoveries.parquet  defense.parquet  xga.parquet
    vaep.parquet  vaep_summary.parquet  value_reports.parquet
    rotation_timeline.parquet  rotation_summary.parquet  double_commits.parquet
    situational.parquet  momentum.parquet  win_prob.parquet  coach_report.parquet
    wp_training.parquet       # rows from extract_win_prob_training_data
```

Why both SQLite and parquet: SQLite (built into Python) handles the parts that need atomic updates: the list of files ingested, failures, and versions. Parquet files per match avoid schema migrations, because each match's tables have whatever columns the pipeline produced at that version. A season query is a concat over the selected matches. That's fast enough for hundreds to low thousands of matches; DuckDB can read the same files later if that changes.

### SQLite schema

```sql
CREATE TABLE replay_files (          -- every file ever seen; the duplicate ledger
  sha256       TEXT PRIMARY KEY,
  match_id     TEXT,                 -- NULL until parsed successfully
  source       TEXT NOT NULL,        -- 'upload' | 'demos_dir' | 'ballchasing' | 'legacy_csv'
  source_ref   TEXT,                 -- original filename / path / ballchasing id
  first_seen   TEXT NOT NULL,
  status       TEXT NOT NULL,        -- 'ok' | 'parse_failed' | 'compute_failed'
  error        TEXT
);

CREATE TABLE matches (
  match_id          TEXT PRIMARY KEY,
  sha256            TEXT REFERENCES replay_files(sha256),  -- NULL for legacy CSV rows
  played_at         TEXT,            -- ISO-8601; get_match_timestamp / extract_replay_date
  playlist          TEXT,
  blue_goals        INTEGER,
  orange_goals      INTEGER,
  overtime          INTEGER,
  analytics_version TEXT,            -- NULL for legacy rows
  computed_at       TEXT,
  fidelity          TEXT NOT NULL    -- 'replay' | 'legacy_summary' | (later) 'stats_api'
);

CREATE TABLE match_players (         -- small join table so "matches with X" needs no parquet scan
  match_id     TEXT REFERENCES matches(match_id),
  player_key   TEXT,                 -- str(p.id.id), the platform ID; never the display name
  display_name TEXT,
  team         TEXT,                 -- 'Blue' | 'Orange'
  is_bot       INTEGER,
  PRIMARY KEY (match_id, player_key)
);
```

## Module layout

```
store/
  __init__.py
  schema.py      # DDL, open_store(path), migrations (versioned via PRAGMA user_version)
  writer.py      # write_match(record), mark_file(...)
  reader.py      # list_matches(filters), load_match(match_id), load_table(name, match_ids)
ingest/
  __init__.py
  sources.py     # ReplayCandidate + UploadSource, DemosFolderSource
  run.py         # ingest(candidates, compute_fn, store) -> IngestReport
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
    match_id: str
    meta: dict             # columns of `matches`
    players: list[dict]    # rows of `match_players`
    tables: dict[str, pd.DataFrame]   # name -> table, written as parquet
```

`ingest.run.ingest()` does: read bytes → hash → skip if `replay_files` already has it → save the file to `replays/` → parse → `compute_fn(...) -> MatchRecord` → write it → record status. A failure in one file gets recorded in the ledger and never stops the rest of the batch. Calling ingest again on the same folder does nothing new.

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
- [ ] `store/` package with schema, writer, reader; `pyarrow` added to requirements.
- [ ] Tests: write/read round-trip; rewriting a match replaces it completely (no leftover tables); stale query by version.

### Phase 2: One pipeline, one ingest path
- [ ] Extend `_compute_match_analytics` to return a `MatchRecord`. Move the batch-only work into it (`Won`, `Timestamp`, `MatchID` on player rows, win-prob training rows, kickoff `MatchID`) so nothing exists only in batch mode.
- [ ] `ingest/run.py` takes `compute_fn` as a parameter. The pipeline still lives in `app.py` until the page split; passing it in avoids a circular import and keeps ingest testable with a fake.
- [ ] Batch upload becomes `ingest([UploadSource(f) ...], compute_fn, store)`. **Delete the inline batch pipeline** (`app.py:4443-4520`).
- [ ] Win-prob model training reads `wp_training` across the store instead of only the current batch.
- [ ] Tests: ingesting the same file twice parses once; a broken file is recorded as `parse_failed` and the rest of the batch continues.

### Phase 3: Season mode reads the store
- [ ] Season dataframe = `load_table("players", match_ids)` joined with `matches` metadata. Remove the column-default backfill list.
- [ ] Session detection uses `matches.played_at`. Remove the "every 5 matches" fallback and the bare `except:`. Matches without a date sort last and are labelled as such.
- [ ] **Migration:** a one-time `migrate_legacy_csv()` imports `career_stats.csv` / `career_kickoffs.csv` as `fidelity='legacy_summary'` matches (no replay, no events, version NULL). Re-ingesting the real replay later replaces the legacy record with a full one because the `match_id` is the same. Views that need events quietly leave out legacy matches and show a count ("12 legacy matches have summary stats only").
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
| Parquet file count grows (~20 files per match) | Fine for thousands of matches; switch to one file per table partitioned by match if it becomes a problem |
| Re-parsing for frame-level views is slow | Cached per session; only the views that need frames pay for it |

## Open questions

1. Do you still have the original `.replay` files for the matches in `career_stats.csv`? If so, migration is just "ingest the folder" and legacy records are rare.
2. Is adding `pyarrow` acceptable? (It's the only new required dependency.)
3. Should bots be stored and filtered at view time (proposed), or dropped at ingest as batch mode does today?
