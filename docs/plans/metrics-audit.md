# Metrics Audit: What to Keep, Fix, Rebuild, or Delete

Status: proposed · Last updated: 2026-10-11 · Companion to [`match-store.md`](match-store.md)

This audit assumes none of the existing work is correct by default. Every number the app shows was traced to the code that produces it. Line references are to `app.py` unless noted. Findings marked **bug** are cases where the code doesn't do what its name or docstring says. All of them were confirmed by reading the code and, for library behavior, carball 0.7.5's source. None were confirmed by running a replay, which Phase 0 of the match-store plan does.

## TL;DR

1. **Several headline stats are wrong because of bugs, not modeling choices.** The app reads carball's hit flags under names that don't exist, so carball's physics-based shot, goal and save detection is never used. The goal width constant is twice the real value. Frame count divided by 30 is used as the game clock. Shots that miss the net are counted as saves.
2. **Every "model" is hand-tuned, with weights nobody has checked.** That covers xG, xGOT, save difficulty, win probability, "VAEP" and the coach report. `fit_value_model()` exists and is never called. With about 20 replays (roughly 100 goals) there isn't enough data to *fit* replacements either, so the rebuild uses physics instead of curve fitting.
3. **A few features compute impressive-looking output from inputs that can't support it.** The counterfactual coach report and Partnership Intelligence are the main ones. They should be deleted, not tuned.
4. **carball already computes a lot of good stats that the app ignores** (boost economy, positioning tendencies, possession time, ball carries, demos), while the app reimplements weaker versions of some of them.

The fix is not 25 separate patches. Most of the analytics should be rebuilt around **one physical question: "who can reach the ball first, and where is it going?"** That needs no training data, and most metrics follow from it (see *The rebuild*).

---

## Part 1: Bugs (fix regardless of anything else)

### B1. carball's hit flags are read under the wrong names, so they never fire
`calculate_shot_data` reads `getattr(hit, 'is_shot', False)` / `'is_goal'` (`app.py:1582-1583`), and `analytics/save_metrics.py:118` reads `'is_save'`. carball's `Hit` proto fields are `shot`, `goal`, `save`, `pass_`, `assist`, `aerial`, `dribble`, `clear` (set in `carball/analysis/events/hit_detection/hit_analysis.py`). The `getattr` default hides the mismatch, so all three are always `False`. Consequences:
- **carball's shot detector is thrown away.** It simulates the ball's flight (`BallSimulator.get_is_shot()`) and only flags a touch as a shot if the ball would actually go in. The app uses its own rule instead: any touch in the attacking half moving toward goal faster than 1200 uu/s (`app.py:1594-1600`). That rule also catches clears, passes and wild hits, which inflates shot counts.
- **Explicit save attribution never runs.** Every save falls back to "nearest defender", even though the code and docs say explicit attribution comes first.

### B2. Goal width is doubled, and there are two different xG implementations
`constants.GOAL_HALF_W = 1784`. The real half-width is **893** uu; carball's `STANDARD_GOAL_WIDTH_HALF = 893`, and `analytics/shot_quality.py` also defaults to 893. `app.py` defines its own `calculate_xg_probability` / `calculate_xgot_probability` (`app.py:1484-1530`) using the wrong constant. Those are the versions actually called; the module's versions are dead code. So:
- **On target** accepts shots up to 891 uu wide of the post.
- **The xG angle term** sees a goal twice as wide.
- **xGOT's placement term** treats a shot into the real corner as "near the middle".
- The 2D goal-mouth chart's axes are also twice as wide.

### B3. Frame count is used as the game clock
Time is computed as `frame / 30` everywhere. Replay frames include goal explosions and the 3-second kickoff countdowns, and the replay's real frame rate varies. carball provides the actual clock in `game_df['game']` (`seconds_remaining`, `is_overtime`, `time`; see `carball/json_parser/frame_parser.py:65`), and the app never reads it. Consequences:
- **Win probability jumps to 100/0 before the match is over.** With `t >= 300 and diff != 0 → 100/0` (`app.py:1197`, `1371`), a match with a few goals passes "300 s" of frames with maybe 30–60 s still on the real clock, and the curve goes to certainty while the game is live.
- **OT is flagged in regulation matches.** `is_ot = duration > 305` in both win-prob functions is true for most regulation matches with goals. `detect_overtime` Method 3 (`app.py:1018-1026`) estimates time spent on goal resets, but only runs when the metadata check fails.
- **Win-prob training data is mislabeled.** `extract_win_prob_training_data` stops at frame 9000 (`app.py:1253`), which drops the last real minute of regulation, the most informative part. Its `time_remaining` labels are also skewed.
- **"Last minute" and "first half" are wrong.** `Goals_Last_Min`, `Goals_First_Half` and similar (`app.py:2194-2196`) use frame-based halves.
- **The value model's clock is wrong.** `SecondsRemaining` in the possession-value states (`analytics/possession_value.py`) is "frames until the replay ends ÷ 30", not the game clock.

### B4. Shots that miss the net are counted as saves
`build_save_events` treats every shot with `Result == "Shot"` as saved (`analytics/save_metrics.py`) and credits it to the nearest defender. A 30%-off-target shot that rolls wide becomes a "save" with a difficulty score. With B1, nothing filters this.

### B5. Long-range goals are probably recorded as plain shots
A touch only counts as the goal-scoring hit if it's within 10 frames of the goal (falling back to 30 frames, `app.py:1556-1570`). A shot from 3,000 uu at 2,000 uu/s takes about 45 frames to arrive, so it's labeled `"Shot"`. Because of B1, carball's `goal` flag doesn't catch it either. Combined with B4, **such a goal is then credited as a save by the nearest defender.** This needs confirming on a real replay in Phase 0, but the logic is clear.

### B6. Kickoffs record only one of the two kickers
`calculate_kickoff_stats` keeps the single player closest to center (`app.py:1414-1428`). Every kickoff has one kicker per team, so the other team's kicker, and their spawn and outcome, are never recorded. Career kickoff stats cover about half the real kickoffs, and only from one side's point of view.

### B7. Rotation stats break in 2v2
- **Rotation breaks need 3 players.** `RotationBreaks` requires at least 3 players (`app.py:2147`), so it's **always 0 in doubles**. Your default focus pair suggests that's most of your games.
- **Double commits only count in the attacking half** (`app.py:2136`), which skips the defensive double commits that actually concede goals.

### B8. Smaller definition errors
| Stat | Problem |
|---|---|
| `Time Airborne (s)` (`app.py:1802`) | Car z > 300 counts **wall driving** as airborne |
| `Avg Recovery (s)` (`app.py:1828`) | "Time from **any** touch until supersonic", capped at 5 s. Dribbles, flicks and passes all score 5 s. Not a recovery measure |
| `Possession` (`app.py:2329`) | Share of the match's touches, not possession. carball computes real possession time |
| `Goals Prevented` (`app.py:2433`) | On-target shots faced minus goals conceded, i.e. roughly saves, under a misleading name. "Prevented" should mean xG faced minus goals conceded (R3) |
| `xGA` (`app.py:1954`) | Credits each shot's xG to whoever is nearest **the ball** when the shot happens. That's usually the player challenging the shooter, not the one responsible |
| `Pressure Time (s)` | Actually shadow-defense time; nothing measures pressure |
| `Saves_Last_Min` (`app.py:2251-2264`) | **Made up**: total saves × the share of the opponent's saved shots that came late. No real per-save timing |
| `Carry_Time` | Within 300 uu in 2D (ignores height) and moving the same direction. Catches a car driving under an aerial ball; carball has proper ball-carry detection |

---

## Part 2: Delete

These don't need fixing. Their inputs can't support what they claim to show.

### D1. Counterfactual coach report (`analytics/counterfactuals.py`, about 470 lines)
- **The "simulation" just replays hand-picked constants.** It "simulates" tactical actions by nudging the ball's Y position and the possession/pressure features by fixed amounts per action (`attack_bias`, `pressure_delta` in `ACTION_LIBRARY`), adding noise, and scoring the result with the **never-fitted** linear value model. The ranking of "what you should have done" is fully determined by those constants. Nothing about the actual replay can change which action looks best in a given situation.
- **The output looks authoritative but isn't.** It's presented as advice with confidence scores, which is the most misleading output in the app.

**Replace with:** *Goal autopsies* (see N1). Factual and checkable.

### D2. Partnership Intelligence / chemistry (`analytics/chemistry.py`, `partnership_*`)
`build_streams_from_match_stats` builds fake "frame" and "event" streams **from per-match totals**:
- "handoff success" = whether either player had an assist
- "pressure release" = whether average "recovery" (B8) was under 1.4 s
- `RotationRole = "Defense" if Time_1st% > 38`. First man is the player closest to the ball, so this label is **backwards**.

Shrinkage, confidence intervals and driver explanations are then layered on top. The careful statistics can't fix fabricated inputs.

**Replace with:** an honest pair table (see N6).

### D3. Fake per-event uncertainty
- **Single events get invented confidence intervals.** Each shot's xG/xGOT and each save's SDI gets an interval from a formula like `spread = 0.08·(1−xG)+0.02` (`shot_quality.py`, `save_metrics.py`). A single event's estimate has no sample to bootstrap; these intervals are made up.
- **What to keep:** bootstrap intervals on **aggregates** (`stats_uncertainty.py`) are legitimate and should stay.

### D4. Duplicate and legacy columns
- **One save number under three names.** `ExpectedSaveProb = 1 − SDI` and `SaveImpact = SDI`.
- **Legacy aliases.** `Total_xS`, `Avg_xS`, `Hard_Saves`, `Saves_Nearby`, plus the `VAEP` naming for something that isn't VAEP.
- **Duplicate xG/xGOT implementations.** The app and `analytics/shot_quality.py` each have their own (B2).

The store makes renames free, because there's no old CSV schema to stay compatible with.

### D5. The current `Rating` and `Luck %`
- **Rating** `= 5 + goals + 0.75·assists + 0.6·saves + 0.4·key passes + 0.2·shots − 0.3·team goals conceded`. Arbitrary weights, inflated saves (B4), and defense judged purely on team results.
- **Luck** only measures one direction: it returns 0 whenever you scored nothing, so it can never show bad luck. It also ignores the opponent's finishing.

**Replace both** with value-based equivalents (R5, R6).

---

## Part 3: The rebuild, physics first

### Constraints that shape it
- **About 20 replays, all 2v2, growing slowly** (manual saves). That's roughly 200 shots and 100 goals, and maybe a few hundred more matches a year.
- **That's too little to fit statistical models.** A fitted xG model needs thousands of shots, and a fitted "chance of scoring in the next 10 s" model needs thousands of goals. Fitting on 100 goals would produce numbers that look precise and mean nothing. That rules out training a model on your own games for now.
- **But Rocket League is deterministic physics, and the replay records every car and the ball 30 times a second.** Football analytics needs data because it can't simulate players; we mostly can.

### The core: reachability
For any moment in a match:

1. **Where is the ball going?** Project its flight with gravity and bounces. carball already ships a `BallSimulator` (it's what its shot detection uses), or we write a small one.
2. **How fast can each car get to a point on that path?** A simple kinematic estimate from each car's position, velocity, facing and boost: accelerate toward the point (faster with boost), up to max speed.
3. **The margin:** each team's earliest arrival time minus the ball's arrival time, both for the ball's path and for the goal line.

That single computation (call it `reach(state)`) answers most of the questions the app asks, with no training data and a handful of physically meaningful constants: car acceleration, boost acceleration, max speed, and one "reaction slack" value for human imperfection. Every one of those constants is either a known game value or tunable by watching a few clips.

### What follows from it

| Metric | New definition | Replaces |
|---|---|---|
| **R1. xG (one shot-quality number)** | For each shot (carball's physics-based shot flag, B1): P(the ball crosses the goal line before any defender can reach its path), turned into a probability by a soft margin (logistic on "defender arrival − ball arrival"). A clear shot with nobody able to reach it ≈ 0.95; a shot the keeper can meet with half a second to spare ≈ 0.05. | Hand-weighted angle × distance formula **and** the separate xGOT |
| **R2. Merge xGOT into xG** | In Rocket League the touch *is* the shot. After contact, the trajectory and the defenders' positions determine the outcome, which is exactly what R1 measures. A separate "pre-shot" xG would need a model of where a player *could* have hit the ball, and 20 replays can't support that. One honest number is better than two dubious ones | xGOT |
| **R3. Saves & save value** | Saves come from carball's `save` flag (a defending touch on a real shot). Save value = the xG of the shot saved. Goals conceded above expected = goals conceded − xG faced. All the save metrics collapse into these | SDI / ExpectedSaveProb / SaveImpact |
| **R4. Control** | At every moment, which team gets to the ball first: Blue control / Orange control / **contested** (arrival times within ~0.3 s) | The sigmoid "possession belief" in `possession_value.py`, and touch-share `Possession` |
| **R5. Threat** | For the team in control: how dangerous the moment is if they shoot now, i.e. R1 evaluated at the ball's current position against the defenders' reach to the goal. Not a fitted probability, so it's labeled a **threat index**, not a percentage | The unfitted linear "VAEP" state value |
| **R6. Touch value** | Each touch's change in threat for your team minus change in threat for the opponent (R5, before vs. ~0.5 s after). Shown as attacking and defensive totals and per-touch averages. **No composite "Rating"**: with 20 matches a single number would mostly be noise | "VAEP", `Rating` |
| **R7. xWin (replaces Luck)** | Simulate the match 10,000 times from both teams' shot xGs (R1). "You won a match you'd win 22% of the time from the chances created." Correct in both directions | One-sided Luck % |
| **R8. Win probability** | A clock model. Goals arrive at a roughly constant rate, so from score difference *d* and **real** time remaining *t*, P(win) comes from the Skellam distribution (difference of two Poisson counts), and ties at 0:00 go to OT at 50/50. It needs **one** parameter, the goal rate, which 20 matches (~100 goals) already estimates to within about ±10%. Optionally nudged by R5 threat in the moment. Handles the lead × time interaction that the current linear model can't | Hand-tuned sigmoid + linear logistic regression |
| **R9. Momentum** | Rolling difference of team threat (R5), smoothed | The squared ball-Y heuristic |

**Calibrating later:**
- **What to tune:** only the few scalars (the margin softness in R1, the reaction slack, the goal rate in R8). Never a big model.
- **When:** once you have ~150+ matches, tune them so that total xG across your matches matches total goals, and so that a calibration plot (predicted xG vs. actual scoring rate, binned) roughly lines up.
- **Faster option:** pull public 2v2 replays from ballchasing at your rank and calibrate on those. This is optional.
- **Versioning:** each tuning is an `ANALYTICS_VERSION` bump, and the store's recompute updates every match.

**Small samples, shown honestly:**
- **Show sample sizes and intervals everywhere.** Every season aggregate shows *n* and a bootstrap interval.
- **Smooth trend lines** (e.g. rolling 5 matches with a band) instead of plotting game-to-game noise.
- **Gate rate stats on volume.** Any rate stat with fewer than ~10 underlying events (e.g. "50/50 win %" from 6 challenges) is greyed out with "not enough data yet".

## Part 4: New stats worth adding

### Use what carball already computes
`proto.players[i].stats` already contains these, and the app ignores nearly all of them. Exact field names to be confirmed in Phase 0:
- **Boost:** small and large pads collected, **stolen** pads, time at 0 boost, time at 100, average boost level, and wasted boost (only `boost_usage` and `wasted_usage` are used today).
- **Positional tendencies:** time on the ground / low air / high air (this fixes B8's airborne stat), time in each third, **time behind vs. in front of the ball**, time as the most-back or most-forward player, and time closest to the ball.
- **Possession time** (fixes the `Possession` stat), **ball carries and flicks** (replaces `Carry_Time`), **demos and bumps**, and **hit counts** by type (pass / dribble / aerial / clear).

### Derived, high-value additions
- **N1. Goal autopsies** (replaces D1). For every goal conceded, a factual snapshot 5 s before and at the shot:
  - who was last back, and how far from the net
  - each defender's boost
  - whether a double commit happened
  - whether both teammates were ahead of the ball
  - the touch with the most negative value (R6) in the lead-up

  Then aggregate over the season: "38% of goals conceded came after both of you were in front of the ball; 21% after you had under 20 boost when last back." This is concrete coaching from real data.
- **N2. Possession chains:** sequences of same-team touches until the other team touches. Chain outcomes (shot, goal, turnover), turnovers by field zone, chain length, and the time from winning the ball to a shot.
- **N3. 50/50s / challenges:** contested moments (R4) that end in a touch from each side. Who wins them (which team controls the ball afterwards), split by first-man role and by boost level.
- **N4. Proper xA (expected assists):** the xG of the *resulting* shot credited to the passer. Today's xA is the xG at the receiver's touch.
- **N5. Kickoffs, rebuilt:** both kickers (B6), spawn, first-touch timing difference, and outcome defined by **which team has control (R4) and threat (R5) 3 s after the first touch** rather than "ball Y > 1000 after 3 s". Also goals within 10 s of kickoff, for and against.
- **N6. Duo report** (replaces D2). You almost always play 2s with the same teammate, so pairing tables would mostly show one pair with tiny samples. A side-by-side of **you vs. your teammate in the same matches** is more useful: first-man share, touch value (R6), challenges won, boost starved time, who was last back on goals conceded (N1), and who double committed. Same matches, same opponents, so the comparison is fair.
- **N7. 2v2 rotation:**
  - first/second man at every moment (closer to the ball, goal-side considered)
  - double commits anywhere on the field (both committing to the same ball), with what happened next
  - **"both in front of the ball" time**, the classic 2s mistake, and how often it led to a shot against
  - last-back distance from your own net when the opponent gets control
  - spacing (distance between you and your teammate)
- **N8. Recovery, redefined:** after a touch made in the air or on a wall, the time until the car is back on its wheels and driving toward a useful position. If carball's data can't measure this reliably, drop it rather than ship a wrong number.

---

## Part 5: Keep (mostly as-is)

- **Chart system:** tokens, theme, formatters, factory. It's sound; only some chart inputs change.
- **Uncertainty on aggregates:** bootstrap intervals and reliability by sample size (`stats_uncertainty.py`).
- **Speed-unit rule:** uu/s internally, mph for display only.
- **Shadow-defense time:** a reasonable heuristic, renamed honestly. It may be absorbed by the positional tendencies later.
- **Session detection:** fine once it runs on real timestamps (match-store Phase 3).
- **Kickoff coordinate normalization:** mirroring Orange's kickoff coordinates so both teams attack the same direction is correct.

---

## Order of work (combined with the match-store plan)

1. **Store first, as a pure refactor** (match-store Phases 0–4). The parity check must compare like for like, so **no metric changes go into the refactor**.
2. **Bug-fix release** (`ANALYTICS_VERSION` bump): B1 (flag names), B2 (goal width, single xG implementation), B3 (real clock), B4/B5 (saves and goals from carball flags), B6 (both kickers). Recompute everything. Expect shot counts to drop, saves to drop sharply, some shots to become goals, and the win-prob curve to stop jumping to certainty early.
3. **Deletions:** D1–D5. Removes about 1,000 lines plus the two misleading tabs.
4. **carball stats** (Part 4, first list). Cheap and immediately useful.
5. **Reachability core + R1–R9.** No data requirement; tune the few constants later as replays accumulate.
6. **N1–N8**, starting with goal autopsies (N1), 2v2 rotation (N7) and the duo report (N6), which give the most coaching value for doubles.

## Decisions

- **2v2 only.** Every analysis assumes two players per team; the 1s/3s code paths and generality get removed rather than maintained. Non-2v2 replays are skipped at import (see `match-store.md`).
- **Physics first, fit later.** About 20 replays can't support fitted models. The rebuild uses reachability physics with a few named constants, tuned only once there are enough matches.
- **xGOT is merged into xG** (R2), and there's **no composite Rating** (R6).
