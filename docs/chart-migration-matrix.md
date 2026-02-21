# Chart Migration Matrix (Single Match + Season Batch)

This file is the **single source of truth** for chart migration planning and execution. Every chart task must map to one or more rows here. Mark status as complete (`[x]`) only when migrated and validated.

## Global invariants (non-negotiable)

1. **Shared theme tokens only**
   - Use centralized color/spacing/typography tokens (`TEAM_COLORS`, canonical semantic palette, spacing scale).
   - No hard-coded one-off colors at chart call sites.
2. **Canonical chart grammar only**
   - Allowed canonical chart families: `metric`, `bar`, `line/area`, `scatter`, `heatmap`, `table`, `composition`, `flow`, `timeline`.
   - Each existing chart must resolve to exactly one canonical grammar type (composed views may include multiple canonical subcharts).
3. **Deterministic sorting**
   - Every categorical chart must declare sort key + tie-break rule (typically `value desc`, then `name asc`).
   - Player/team ordering must be stable across reruns.
4. **Unified tooltip language**
   - Tooltips use consistent terms, units, precision, and sentence casing.
   - Prefer `Metric: value unit` format; avoid mixed abbreviations across tabs.
5. **No inline style literals in chart call sites**
   - Move visual styling to shared presets/theme adapters.
   - Chart calls should specify semantic intent (`preset='hero'`) rather than raw style dicts.

## Canonical interaction baseline (applies unless overridden per-row)

- Sorting: deterministic, explicit, and stable.
- Hover: concise value-first labels with units + consistent decimal precision.
- Focus behavior: selected player/team persists across tabs where logically valid.
- Keyboard/screen-reader fallback: chart title + one-sentence text summary below chart.

---

## Single Match flow (interactive tabs)

| Status | Chart ID | Current location in `app.py` | Current chart type | Analytic intent | Target canonical chart type | Required data fields + transformations | Style tier | Interaction rules | Accessibility notes | Priority / risk |
|---|---|---|---|---|---|---|---|---|---|---|
| [x] | SM-KO-01 | `with t1` Kickoff Analysis, `Kickoff Win Rate (Selected)` gauge | Indicator gauge | Comparison (performance vs 100%) | `metric` (KPI dial/progress) | `kickoff_df[Player, Result]` → filter `focus_players`; `win_rate = wins/total*100` | hero | Hover on gauge shows exact `%`; focus tracks selected players | Ensure green/red not sole signal; add numeric text; minimum 4.5:1 contrast for gauge text | P1 / Low |
| [ ] | SM-KO-02 | `with t1` Kickoff Analysis, `Kickoff Outcomes` | Pitch scatter | Distribution (end locations by result) | `scatter` (spatial) | `kickoff_df[End_X, End_Y, Result, Player]` filtered by focus; grouped by result | support | Legend toggles result classes; hover `Player + Result`; deterministic legend order Win/Loss/Neutral | Marker shapes + colors for colorblind fallback; avoid dense overlap via jitter/opacity strategy | P1 / Medium |
| [ ] | SM-MN-01 | `with t2` Match Narrative, team tug-of-war overview | Diverging horizontal bar | Comparison (Blue vs Orange by stat) | `bar` (diverging comparison) | Team aggregates from `df` for Goals/Shots/Saves/etc.; normalize by stat total into signed fractions | hero | Fixed stat order; hover shows both raw values + share; optional sort by abs delta | Diverging palette with neutral baseline line; labels readable at small widths | P1 / Medium |
| [ ] | SM-MN-02 | `with t2` Match Narrative, `🏆 Win Probability` | Area + line | Trend / expected-vs-actual (model confidence over time) | `line/area` | `win_prob_df[Time, WinProb]`; add 50% reference, OT marker at 300s when needed | hero | Hover `Time`, `Blue Win %`; synchronized x-domain with xG/pressure charts | Add textual fallback for model unavailable; ensure fill opacity doesn't hide line | P1 / Medium |
| [ ] | SM-MN-03 | `with t2` Match Narrative, `Cumulative xG Over Time` | Multi-line with goal markers | Trend + expected-vs-actual | `line` (multi-series with events) | `shot_df` cumulative xG by `Team, Time`; goal event markers from shot results; OT line optional | hero | Team toggles; hover includes cum xG and event annotations; stable team order Blue/Orange | Marker shapes for goals; high-contrast team colors on dark bg | P1 / Medium |
| [ ] | SM-MN-04 | `with t2` Match Narrative, `🌊 Pressure Index` | Positive/negative area strip | Trend / momentum comparison | `line/area` | `momentum_series` indexed by time; clip to ±; optional goal markers | support | Linked x-range with WinProb/xG; hover returns signed pressure value | Do not rely on hue alone for side ownership; add axis annotations (Blue pressure / Orange pressure) | P2 / Medium |
| [ ] | SM-SM-01 | `with t3` Shot Map tab | Pitch scatter (shots/goals/chances) | Distribution + composition | `scatter` (spatial events) | `shot_df[X,Y,Team,Result,BigChance,Player]`; split shot/goal/big chance traces | hero | Hover standardized: shooter, team, outcome, xG; legend toggles by outcome | Large markers require overlap strategy; add non-color symbol encoding | P1 / Medium |
| [ ] | SM-SV-01 | `with t3b` Frozen Frame Shot Viewer | Event scatter snapshot | Comparison (player/ball positions at shot time) | `scatter` (annotated snapshot) | frame-level positional slices from `game_df` + selected shot metadata | detail | Scrubber/selector drives frame; hover includes speed/boost if available | Ensure text labels not overcrowded; provide table fallback for exact coordinates | P2 / High |
| [ ] | SM-PM-01 | `with t4` Pass Map | Directed scatter/lines network on field | Flow/composition | `flow` (network) | `pass_df[From,To,StartX,StartY,EndX,EndY,Team,Complete]`; aggregate thickness by count | support | Hover: passer→receiver, count, completion%; stable node ordering by team then name | Edge contrast and line width minimums; arrowheads visible on dark bg | P2 / High |
| [ ] | SM-HM-01 | `with t5` Heatmaps | 2D contour density | Distribution (positional occupancy) | `heatmap` (spatial density) | player positions from `game_df[pos_x,pos_y,pos_z]`; filter airborne/ground per design; sampling | support | Player selector persists from sidebar/export; hover may be disabled for perf | Perceptually uniform sequential palette; avoid red-green gradients | P2 / Medium |
| [ ] | SM-AD-01 | `with t8` Advanced, `Aerial Hits` | Grouped bar | Ranking/comparison | `bar` | `aerial_df[Name,Team,Aerial Hits]`; sort desc by hits then name | support | Click legend filters team; hover includes per-game normalization if available | Keep label angle readable; provide numeric data table toggle | P2 / Low |
| [ ] | SM-AD-02 | `with t8` Advanced, `Time Airborne (s)` | Grouped bar | Ranking/comparison | `bar` | `aerial_df[Name,Team,Time Airborne (s)]`; sort desc | support | Shared sorting with Aerial Hits for consistency | Unit suffix seconds in labels/tooltips | P2 / Low |
| [ ] | SM-AD-03 | `with t8` Advanced, `Avg Time to Supersonic After Hit` | Grouped bar | Ranking (lower is better) | `bar` | `recovery_df[Name,Team,Avg Recovery (s)]`; sort asc (explicit inversion) | support | Hover clarifies “lower is better”; deterministic asc order | Add icon/text cue for inverse metric to avoid misread | P2 / Medium |
| [ ] | SM-AD-04 | `with t8` Advanced, `Fast Recovery Rate (< 1s)` | Grouped bar | Ranking/comparison | `bar` | `recovery_df[Name,Team,Recovery < 1s %]`; sort desc | support | Shared player order option with SM-AD-03 | Percent formatting with 0–100 bounds and one decimal | P2 / Low |
| [ ] | SM-AD-05 | `with t8` Advanced, `Pass Flow Network` | Sankey | Flow/composition | `flow` | `pass_df` aggregated by pair and team; source-target-index mapping; filter top links | hero | Node focus highlights incoming/outgoing links; hover standardized with attempts/completion | Ensure colorblind-safe link palette + sufficient node label contrast | P1 / High |
| [ ] | SM-AD-06 | `with t8` Advanced, `Shadow Defense Time %` | Grouped bar | Ranking/comparison | `bar` | `defense_df[Name,Team,Shadow %]`; sort desc | support | Team toggle + deterministic sort | `%` labels always visible for top bars | P2 / Low |
| [ ] | SM-AD-07 | `with t8` Advanced, `Total Pressure Time (s)` | Grouped bar | Ranking/comparison | `bar` | `defense_df[Name,Team,Pressure Time (s)]`; sort desc | support | Hover includes share of team total | Seconds unit and readable axis ticks | P2 / Low |
| [ ] | SM-AD-08 | `with t8` Advanced, `Expected Goals Against` | Grouped bar | Expected-vs-actual proxy (defensive shot quality conceded) | `bar` | `xga_df[Name,Team,xGA]`; sort desc | support | Hover with shots faced context | Clarify that higher xGA conceded is worse | P2 / Medium |
| [ ] | SM-AD-09 | `with t8` Advanced, `Avg Distance to Shot` | Grouped bar | Comparison (defensive spacing) | `bar` | `xga_df[Name,Team,Avg Dist to Shot]`; sort asc or desc per interpretation (must fix invariant) | detail | Hover includes distance units and count of defending events | Include unit (uu/meters) consistently | P3 / Medium |
| [ ] | SM-AD-10 | `with t8` Advanced, `Total VAEP` | Sorted bar | Ranking (value added) | `bar` | `vaep_summary[Name,Team,Total_VAEP]`; filter non-null; sort desc | hero | Click player to cross-filter scatter if enabled | Diverging color for positive/negative VAEP values | P1 / Medium |
| [ ] | SM-AD-11 | `with t8` Advanced, `VAEP consistency` scatter | Scatter | Comparison/distribution | `scatter` | `vaep_summary[Name,Avg_VAEP,Total_VAEP,Team]`; optional bubble by actions count | detail | Hover full stat card; axis zero-lines | Minimum marker size + text alternative list | P3 / Medium |
| [ ] | SM-AD-12 | `with t8` Advanced, `Total Save Impact` | Sorted bar | Ranking (save difficulty handled) | `bar` | `xs_summary` filtered `SaveEvents>0`; `Total_SaveImpact`; sort desc | support | Stable filter criteria + deterministic sorting | Explain SDI + Expected Save Probability in tooltip | P2 / Medium |
| [ ] | SM-AD-13 | `with t8` Advanced, `Avg Save Difficulty (SDI)` | Sorted bar | Comparison efficiency | `bar` | `xs_summary` filtered `SaveEvents>0`; `Avg_SaveDifficulty`; sort desc | detail | Linked ordering option with SM-AD-12 | Percent/decimal precision fixed (2 dp) | P3 / Low |
| [ ] | SM-RT-01 | `with t9` Rotation, `Time Spent as 1st/2nd Man` | Stacked bar | Composition/comparison | `bar` (stacked composition) | `rotation_summary[Name,Team,Time_1st%,Time_2nd%]`; normalize to 100 where needed | hero | Hover shows role split + double commits; sort by Time_1st% desc | Stack colors must remain distinguishable in CVD modes | P1 / Medium |
| [ ] | SM-RT-02 | `with t9` Rotation, `Rotation Timeline` (per team) | Heatmap | Timeline/distribution | `timeline` (heatmap lane) | `rotation_timeline[Team,Player,Time,Role]`; map role→ordinal; sample if >5k rows | support | Team switch, zoom x-axis, hover `time/player/role` | Keep colorbar labels textual (`1st`,`2nd`); avoid red/green pair | P2 / High |
| [ ] | SM-RT-03 | `with t9` Rotation, `Double Commit Locations` | Pitch scatter (X markers) | Distribution/risk hotspots | `scatter` (spatial incidents) | `double_commits_df[BallX,BallY,Team,Player1,Player2,Time]` | support | Hover incident detail; legend grouped by team with counts | X markers need high stroke contrast against field | P2 / Medium |
| [ ] | SM-TA-01 | `with t10` Tactical Replay Viewer | Animated 3D scatter + trajectory | Timeline/comparison | `timeline` (animated spatial scene) | frame arrays from `game_df`/`proto`: player+ball xyz, boost, frame time; stride + window transforms | hero | Play/pause, slider scrub, deterministic frame stride; hover player+boost | Provide reduced-motion fallback (static frame); ensure controls keyboard reachable | P1 / High |

## Single Match flow (export/composite panels)

| Status | Chart ID | Current location in `app.py` | Current chart type | Analytic intent | Target canonical chart type | Required data fields + transformations | Style tier | Interaction rules | Accessibility notes | Priority / risk |
|---|---|---|---|---|---|---|---|---|---|---|
| [ ] | SM-EX-01 | `build_export_shot_map()` used by `t7` Composite Export | Pitch scatter | Distribution + composition | `scatter` | `shot_df` shot/goal/big-chance splits | detail | Non-interactive export; preserve legend semantics in caption | Ensure print-safe palette and marker distinction | P2 / Low |
| [ ] | SM-EX-02 | `build_export_heatmap()` used by `t7` Composite Export | 2D contour heatmap | Distribution | `heatmap` | sampled `game_df[player][pos_x,pos_y,pos_z]` | detail | Non-interactive | Palette must stay interpretable in grayscale printouts | P2 / Medium |
| [ ] | SM-EX-03 | `build_export_scoreboard()` used by `t7` Composite Export | Table | Ranking/comparison summary | `table` | `df` sorted by score, derived luck%, team group separator rows | support | Non-interactive; deterministic row order by team then score desc | Table text contrast and legible minimum font size | P1 / Low |
| [ ] | SM-EX-04 | `build_export_xg_timeline()` used by `t7` Composite Export | Line + markers | Trend expected-vs-actual | `line` | cumulative xG by team + goal markers + OT marker | support | Non-interactive | Ensure markers visible when printed small | P1 / Low |
| [ ] | SM-EX-05 | `build_export_win_prob()` used by `t7` Composite Export | Area + line | Trend | `line/area` | `calculate_win_probability(...)` output + 50% ref + OT marker | support | Non-interactive | Add explicit axis labels in export image | P1 / Medium |
| [ ] | SM-EX-06 | `build_export_zones()` used by `t7` Composite Export | Grouped bar | Composition/comparison | `bar` | `df` positional % fields (`Pos_*`, wall/corner/carry) for selected players | detail | Non-interactive | Small-text bar labels require concise naming | P2 / Medium |
| [ ] | SM-EX-07 | `build_export_pressure()` used by `t7` Composite Export | Area strip + event markers | Trend / momentum | `line/area` | `momentum_series` + goal events from `proto` | detail | Non-interactive | Fill opacity and symbols should remain distinguishable | P2 / Low |

## Season Batch flow (interactive tabs)

| Status | Chart ID | Current location in `app.py` | Current chart type | Analytic intent | Target canonical chart type | Required data fields + transformations | Style tier | Interaction rules | Accessibility notes | Priority / risk |
|---|---|---|---|---|---|---|---|---|---|---|
| [ ] | SB-PF-01 | `with t1` Performance Trends, `metric over Time` | Multi-line + markers | Trend/comparison | `line` | `hero_df[GameNum,metric,Won,Overtime]`; rolling mean; optional teammate series | hero | Metric selector drives chart; legend toggles; deterministic game order asc | Ensure marker shape encodes win/loss beyond color | P1 / Medium |
| [x] | SB-KO-01 | `with t2` Season Kickoff Meta, hero win-rate gauge | Indicator gauge | Comparison | `metric` | `season_kickoffs[Player,Result]` filtered hero → rate | support | Hover exact `%`; no ambiguous scaling | Numeric label + contrast-compliant gauge text | P2 / Low |
| [ ] | SB-KO-02 | `with t2` Season Kickoff Meta, `Win Rate by Spawn Location` | Bar | Ranking/comparison | `bar` | group by `Spawn`, aggregate win rate % | support | Deterministic spawn ordering (canonical kickoff slot order) | Axis labels horizontal/rotated for readability | P2 / Medium |
| [ ] | SB-KO-03 | `with t2` Season Kickoff Meta, spawn map scatter | Spatial scatter | Distribution/comparison | `scatter` | `season_kickoffs` spawn end positions + result classification | detail | Hover spawn/result/time-to-hit; optional filtering by outcome | Use symbol + color for outcomes | P3 / Medium |
| [x] | SB-PS-01 | `with t3` Positional Tendencies, `Avg Positioning` | Pie | Composition | `composition` (100% stacked bar preferred) | `hero_df[Pos_Def,Pos_Mid,Pos_Off]` means | support | If retained as pie: fixed segment order; else migrate to stacked bar | Avoid pie-only encoding; labels with percentages directly on chart | P2 / Medium |
| [ ] | SB-PS-02 | `with t3` Positional Tendencies, `Granular Zones` | Bar | Composition/comparison | `bar` | zone averages from hero positional columns; melt to tidy format | support | Deterministic zone order Def/Mid/Off/Wall | Colorblind-safe categorical palette | P2 / Low |
| [ ] | SB-PS-03 | `with t3` Positional Tendencies, `Comparison` | Grouped bar | Comparison | `bar` | hero + teammate zone means; align zones and players | support | Shared y-scale and consistent player ordering | Team/player colors with accessible contrast | P2 / Medium |
| [x] | SB-RD-01 | `with t4` Player Comparison Radar | Radar polar | Comparison/profile | `composition` (radar or normalized bar panel) | averages across core stats for hero vs teammate; normalize scale where required | hero | Hover per axis metric; consistent axis max across players | Radar labels can crowd; provide tabular fallback | P1 / High |
| [ ] | SB-IN-01 | `with t8` Career Insights, `Stat Correlation with Winning` | Bar (pos/neg) | Ranking (predictiveness) | `bar` | correlations between numeric stats and `Won` binary; drop NaN; sort desc | support | Hover includes correlation and sample size | Use diverging palette + textual sign (+/-) | P2 / Medium |
| [x] | SB-SI-01 | `with t9` Situational, `Goal Distribution by Period` | Pie | Composition/distribution | `composition` (bar preferred) | counts of `Goals_First_Half`, `Goals_Second_Half`, `Goals_Last_Min` | support | Fixed period ordering; hover count + share | Consider replacing pie with bar for label density | P2 / Low |
| [ ] | SB-SI-02 | `with t9` Situational, `Late-Game Scoring Trend` | Line | Trend | `line` | rolling average over `Goals_Last_Min` by `GameNum` | support | Window-size control + deterministic smoothing method | Label rolling window in subtitle for clarity | P2 / Low |
| [ ] | SB-SI-03 | `with t9` Situational, `Goals by Game State` | Bar | Composition/comparison | `bar` | aggregates of `Goals_When_Leading/Trailing/Tied` | support | Fixed game-state order Leading/Tied/Trailing (or strategic order) | Contrast-safe bars + direct labels | P2 / Low |
| [ ] | SB-SI-04 | `with t9` Situational, `Comeback vs Blown Lead Trends` | Dual line | Trend/comparison | `line` | rolling rates for `Comeback_Win`, `Blown_Lead` by game index | detail | Legend toggles series; synchronized y-axis percentages | Distinct dashes/markers in addition to color | P3 / Medium |
| [x] | SB-SI-05 | `with t9` Situational, `Save Distribution by Timing` | Pie | Composition/distribution | `composition` (bar preferred) | counts for save timing buckets from `Saves_Last_Min` and complements | detail | Fixed period ordering + hover counts | Replace/augment pie with bar for readability | P3 / Medium |
| [ ] | SB-SE-01 | `with t6` Session Analytics, `Session Performance Overview` | Combo bar+line dual-axis | Trend/comparison | `line + bar` composite | `summary_df[Session,Win Rate %,Avg Rating]` from detected sessions | hero | Session sort asc; hover includes games/wins context | Dual-axis needs explicit axis color/labels to avoid ambiguity | P1 / Medium |

## Season Batch flow (export dashboard components)

| Status | Chart ID | Current location in `app.py` | Current chart type | Analytic intent | Target canonical chart type | Required data fields + transformations | Style tier | Interaction rules | Accessibility notes | Priority / risk |
|---|---|---|---|---|---|---|---|---|---|---|
| [ ] | SB-EX-01 | `with t7` Season Export, subplot `Rating Over Time` | Line | Trend | `line` | `hero_df[GameNum,Rating]` | detail | Non-interactive export | High-contrast line on dark background | P2 / Low |
| [ ] | SB-EX-02 | `with t7` Season Export, subplot `Positioning` | Bar | Composition | `bar` | means of `Pos_Def/Pos_Mid/Pos_Off` | detail | Non-interactive export | Distinct category colors + legend/caption mapping | P2 / Low |
| [ ] | SB-EX-03 | `with t7` Season Export, subplot `Win Rate by Session` | Bar | Comparison | `bar` | `summary_df[Session,Win Rate %]` if sessions exist | detail | Non-interactive export | Session labels legible in dense seasons | P3 / Low |
| [x] | SB-EX-04 | `with t7` Season Export, subplot `Radar` | Radar polar | Comparison/profile | `composition` | averaged categories `Rating,Goals,Assists,Saves,Shots,xG` | detail | Non-interactive export | Add textual stat table below export in UI for accessibility | P3 / Medium |

---

## Migration execution protocol

1. For every chart PR/task:
   - Locate row(s) by Chart ID.
   - Implement using canonical grammar + global invariants.
   - Validate style tier and interaction/accessibility requirements.
2. Update row `Status` from `[ ]` to `[x]` only when:
   - visual parity/intent parity confirmed,
   - invariants pass,
   - regression checks pass.
3. If scope changes, append new rows (never silently repurpose IDs).

## Director Mode canonical tracks (timeline-first shell)

| Status | Chart ID | Location | Canonical type | Intent | Migration protocol |
|---|---|---|---|---|---|
| [x] | DM-FLD-01 | `app.py` Director Mode field replay layer | `timeline + scatter` | Primary spatial replay synchronized to global timeline cursor | Uses shared `timeline_state` (`current_time`, `selected_event_id`, `playback_speed`, `selected_scenario`) and canonical event markers. |
| [x] | DM-WP-02 | `app.py` Director Mode track stack row 1 | `line/area` | Win probability trajectory with ranked director event markers | Reuses `charts/win_probability.py` semantics and overlays canonical marker grammar (team/intent color, event shape, confidence opacity). |
| [x] | DM-MINI-03 | `app.py` Director Mode track stack rows 2-4 | `line` mini-tracks | xG, VAEP, pressure synchronized on shared x-axis | All tracks consume synchronized timeline grid and vertical cursor for deterministic cross-track time locking. |
| [x] | DM-NAR-04 | `analytics/narrative_engine.py` + `app.py` Narrative Studio | `timeline-linked claims` | Claim navigation from narrative to canonical event queue | Each claim stores `canonical_event_id`; click navigation jumps timeline and selected event state. |

### Legacy tab-specific chart migration status protocol

- Old tab-specific charts under single-match flow are now treated as **secondary deep-dive blocks** rendered beneath Director Mode.
- New feature work must target Director Mode IDs (`DM-*`) first, then optionally backport to legacy sections.
- Mark legacy rows complete only after (a) semantic parity in Director Mode and (b) explicit accessibility text-summary fallback under the track.
