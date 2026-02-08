import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import base64
import tempfile
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# --- 1. SETUP & IMPORTS ---
try:
    import carball
    from carball.json_parser.game import Game
    from carball.analysis.analysis_manager import AnalysisManager
    SPROCKET_AVAILABLE = True
except ImportError as e:
    SPROCKET_AVAILABLE = False
    IMPORT_ERROR = str(e)

KALEIDO_AVAILABLE = False
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    pass

PIL_AVAILABLE = False
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    pass

# Load pitch background image as base64 for Plotly
PITCH_IMAGE_B64 = None
_pitch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple-pitch.png")
if os.path.exists(_pitch_path):
    with open(_pitch_path, "rb") as _f:
        PITCH_IMAGE_B64 = "data:image/png;base64," + base64.b64encode(_f.read()).decode()

st.set_page_config(page_title="RL Pro Analytics", layout="wide", page_icon="🚀")
st.title("🚀 Rocket League Pro Analytics (Final Version)")

# --- 2. PERSISTENCE CONFIG ---
DB_FILE = "career_stats.csv"
KICKOFF_DB_FILE = "career_kickoffs.csv"
REPLAY_FPS = 30  # Standard replay frame rate

# --- 3. HELPER: DATABASE MANAGEMENT ---
def load_data():
    """Loads existing career data from CSVs."""
    stats_df = pd.DataFrame()
    kickoff_df = pd.DataFrame()
    
    if os.path.exists(DB_FILE):
        try: stats_df = pd.read_csv(DB_FILE)
        except Exception as e: logger.warning("Failed to load %s: %s", DB_FILE, e)

    if os.path.exists(KICKOFF_DB_FILE):
        try: kickoff_df = pd.read_csv(KICKOFF_DB_FILE)
        except Exception as e: logger.warning("Failed to load %s: %s", KICKOFF_DB_FILE, e)
        
    return stats_df, kickoff_df

def save_data(new_stats, new_kickoffs):
    """Appends new data to CSVs, handling duplicates by MatchID."""
    # 1. Main Stats
    if not new_stats.empty:
        if os.path.exists(DB_FILE):
            existing = pd.read_csv(DB_FILE)
            existing['MatchID'] = existing['MatchID'].astype(str)
            new_stats['MatchID'] = new_stats['MatchID'].astype(str)
            
            existing_ids = set(existing['MatchID'].unique())
            new_stats = new_stats[~new_stats['MatchID'].isin(existing_ids)]
            combined = pd.concat([existing, new_stats], ignore_index=True)
        else:
            combined = new_stats
        combined.to_csv(DB_FILE, index=False)

    # 2. Kickoff Stats
    if not new_kickoffs.empty:
        if os.path.exists(KICKOFF_DB_FILE):
            existing_k = pd.read_csv(KICKOFF_DB_FILE)
            existing_k['MatchID'] = existing_k['MatchID'].astype(str)
            new_kickoffs['MatchID'] = new_kickoffs['MatchID'].astype(str)
            
            existing_ids = set(existing_k['MatchID'].unique())
            new_kickoffs = new_kickoffs[~new_kickoffs['MatchID'].isin(existing_ids)]
            combined_k = pd.concat([existing_k, new_kickoffs], ignore_index=True)
        else:
            combined_k = new_kickoffs
        combined_k.to_csv(KICKOFF_DB_FILE, index=False)

# --- 4. VISUALIZATION HELPERS ---
def get_field_layout(title=""):
    """Returns a Plotly layout dict with the pitch image as background."""
    layout = dict(
        title=title,
        xaxis=dict(range=[-4700, 4700], visible=False, fixedrange=True),
        yaxis=dict(range=[-6200, 6200], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
    )
    if PITCH_IMAGE_B64:
        layout['images'] = [dict(
            source=PITCH_IMAGE_B64,
            xref="x", yref="y",
            x=-4700, y=6200,
            sizex=9400, sizey=12400,
            sizing="stretch",
            opacity=1.0,
            layer="below"
        )]
    else:
        # Fallback: draw field with shapes if image not found
        layout['plot_bgcolor'] = '#1a241a'
        layout['shapes'] = [
            dict(type="rect", x0=-4096, y0=-5120, x1=4096, y1=5120, line=dict(color="rgba(255,255,255,0.8)", width=2)),
            dict(type="rect", x0=-893, y0=5120, x1=893, y1=6000, line=dict(color="#ff9900", width=2)),
            dict(type="rect", x0=-893, y0=-6000, x1=893, y1=-5120, line=dict(color="#007bff", width=2)),
            dict(type="line", x0=-4096, y0=0, x1=4096, y1=0, line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dot")),
            dict(type="circle", x0=-1000, y0=-1000, x1=1000, y1=1000, line=dict(color="rgba(255,255,255,0.5)", width=2))
        ]
    return layout

# --- 5. PARSING ENGINE (CACHED) ---
@st.cache_resource(show_spinner=False)
def get_parsed_replay_data(file_bytes, file_name):
    if not SPROCKET_AVAILABLE:
        return None, None, None, None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".replay") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        manager = carball.analyze_replay_file(tmp_path)
        game_df = manager.get_data_frame()
        proto = manager.get_protobuf_data()
        return manager, game_df, proto, None
    except Exception as e:
        logger.error("Failed to parse replay %s: %s", file_name, e)
        return None, None, None, str(e)
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass

# --- 6. MATH: PROBABILITY (xG) ---
def calculate_xg_probability(shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y):
    dist = np.sqrt(shot_x**2 + (target_y - shot_y)**2)
    vec_l = np.array([-893 - shot_x, target_y - shot_y])
    vec_r = np.array([893 - shot_x, target_y - shot_y])
    try:
        norm_l = np.linalg.norm(vec_l)
        norm_r = np.linalg.norm(vec_r)
        if norm_l == 0 or norm_r == 0: return 0
        unit_l = vec_l / norm_l
        unit_r = vec_r / norm_r
        dot = np.dot(unit_l, unit_r)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
    except (ValueError, FloatingPointError): angle = 0
    base_xg = (angle * 0.85) * np.exp(-0.00045 * dist)
    speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    speed_factor = 1.0 + (speed - 1400) / 2000.0
    speed_factor = max(0.5, min(speed_factor, 1.5))
    height_factor = 1.15 if shot_z > 150 else 1.0
    xg = base_xg * speed_factor * height_factor
    return min(max(xg, 0.01), 0.99)

# --- 6b. MATH: LUCK % (POISSON BINOMIAL) ---
def calculate_luck_percentage(shot_df, team, actual_goals):
    """Computes Luck % using Poisson Binomial distribution.
    Each shot has a different xG probability. We compute P(scoring >= actual_goals)
    from those individual probabilities, then Luck = (1 - P) * 100 for winners."""
    team_shots = shot_df[shot_df['Team'] == team] if not shot_df.empty else pd.DataFrame()
    if team_shots.empty or actual_goals == 0:
        return 0.0
    probs = team_shots['xG'].values.tolist()
    n = len(probs)
    # Iterative Poisson Binomial: dp[k] = P(exactly k goals)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    for p in probs:
        p = max(0.001, min(p, 0.999))
        new_dp = [0.0] * (n + 1)
        for k in range(n + 1):
            new_dp[k] += dp[k] * (1 - p)
            if k + 1 <= n:
                new_dp[k + 1] += dp[k] * p
        dp = new_dp
    # P(scoring >= actual_goals)
    p_at_least = sum(dp[actual_goals:])
    luck = (1.0 - p_at_least) * 100
    return round(max(0.0, min(luck, 100.0)), 1)

# --- 6c. MATH: OVERTIME DETECTION (FIXED) ---
def detect_overtime(manager):
    """
    Returns True if match went to overtime.
    Uses carball's internal analysis which accounts for pauses/replays.
    """
    # Method 1: Check Python Object (Best)
    if hasattr(manager.game, 'overtime_seconds'):
        return manager.game.overtime_seconds > 0
    
    # Method 2: Check Protobuf Metadata
    proto = manager.get_protobuf_data()
    if hasattr(proto, 'game_stats') and hasattr(proto.game_stats, 'overtime_seconds'):
        return proto.game_stats.overtime_seconds > 0
        
    return False

def get_match_timestamp(proto):
    """Extract match timestamp from replay metadata."""
    ts = ""
    try:
        if hasattr(proto, 'game_metadata'):
            meta = proto.game_metadata
            if hasattr(meta, 'time') and meta.time:
                ts = str(meta.time)
            elif hasattr(meta, 'date') and meta.date:
                ts = str(meta.date)
            elif hasattr(meta, 'id') and meta.id:
                ts = str(meta.id)
    except:
        pass
    return ts

# --- 7. MATH: MOMENTUM & WIN PROBABILITY ---
def calculate_contextual_momentum(manager, game_df, proto):
    if 'ball' not in game_df: return pd.Series()
    try:
        numeric_cols = game_df.select_dtypes(include=[np.number]).columns
        df_numeric = game_df[numeric_cols].copy()
        df_numeric.index = pd.to_timedelta(df_numeric.index / REPLAY_FPS, unit='s')
        df_resampled = df_numeric.resample('1s').mean()
    except Exception:
        df_resampled = game_df.iloc[::REPLAY_FPS, :]

    if df_resampled.empty or 'ball' not in df_resampled: return pd.Series()

    ball = df_resampled['ball']
    normalized_y = ball['pos_y'] / 5120.0
    threat = np.sign(normalized_y) * (normalized_y ** 2)

    top_level_cols = df_resampled.columns.levels[0] if isinstance(df_resampled.columns, pd.MultiIndex) else df_resampled.columns
    blue_players = [p.name for p in proto.players if not p.is_orange and p.name in top_level_cols]
    orange_players = [p.name for p in proto.players if p.is_orange and p.name in top_level_cols]
    
    dist_to_orange = pd.DataFrame(index=df_resampled.index)
    for p in orange_players:
        if 'pos_x' in df_resampled[p]:
            dist_to_orange[p] = np.sqrt((ball['pos_x'] - df_resampled[p]['pos_x'])**2 + (ball['pos_y'] - df_resampled[p]['pos_y'])**2)
    
    dist_to_blue = pd.DataFrame(index=df_resampled.index)
    for p in blue_players:
        if 'pos_x' in df_resampled[p]:
            dist_to_blue[p] = np.sqrt((ball['pos_x'] - df_resampled[p]['pos_x'])**2 + (ball['pos_y'] - df_resampled[p]['pos_y'])**2)

    if not dist_to_orange.empty and not dist_to_blue.empty:
        min_orange = dist_to_orange.min(axis=1)
        min_blue = dist_to_blue.min(axis=1)
        # When blue is attacking (threat > 0), check if blue players are near the ball
        # When orange is attacking (threat < 0), check if orange players are near the ball
        relevant_dist = np.where(threat > 0, min_blue, min_orange)
        final_threat = np.where(relevant_dist > 2000, threat * 0.2, threat)
    else:
        final_threat = threat

    time_seconds = np.arange(len(final_threat))
    if hasattr(df_resampled.index, 'total_seconds'):
        time_seconds = df_resampled.index.total_seconds().values
    return pd.Series(final_threat * 100, index=time_seconds).rolling(window=10, center=True).mean().fillna(0)

def calculate_win_probability(manager):
    """Calculates win probability for Blue Team over time. Overtime-aware.
    Uses a moderate logistic model so the line actually swings with goals."""
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    max_frame = game_df.index.max()
    match_duration_s = max_frame / 30.0
    is_ot = match_duration_s > 305
    frames = np.arange(0, max_frame, 30)
    seconds = frames / 30.0

    """Calculates win probability for Blue Team over time.

    Uses a sigmoid model calibrated for RL's high-scoring, fast-paced nature:
    probability shifts gradually with goal differential and accelerates
    as time runs out. A 1-goal lead at midgame ≈ 60%, only reaching 80%+
    in the final 30 seconds.
    """
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    max_frame = game_df.index.max()
    frames = np.arange(0, max_frame, REPLAY_FPS)
    seconds = frames / float(REPLAY_FPS)

    # Build player team lookup — g.player_id.is_orange is unreliable
    # (it's a reference ID, not the full player object), so we look up
    # the scorer's team from proto.players instead
    player_teams = {}
    for p in proto.players:
        player_teams[str(p.id.id)] = "Orange" if p.is_orange else "Blue"

    blue_goals = []
    orange_goals = []
    # Build player ID -> team lookup for reliable team detection
    _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            frame = getattr(g, 'frame_number', getattr(g, 'frame', 0))

            scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
            team = _pid_team.get(scorer_pid, "Orange" if getattr(g.player_id, 'is_orange', False) else "Blue")

            pid = str(getattr(g.player_id, 'id', ''))
            team = player_teams.get(pid, "Blue")
            frame = min(frame, max_frame)
            if team == "Blue": blue_goals.append(frame)
            else: orange_goals.append(frame)

    blue_goals.sort()
    orange_goals.sort()
    probs = []

    match_length = 300.0  # standard match length in seconds

    for f, t in zip(frames, seconds):
        b_score = sum(1 for gf in blue_goals if gf <= f)
        o_score = sum(1 for gf in orange_goals if gf <= f)
        diff = b_score - o_score

        if t >= 300 and diff == 0:
            # In overtime or heading to OT: 50/50 next-goal-wins

        time_remaining = max(match_length - t, 0.0)

        if t >= match_length and diff == 0:
            p = 0.5
        elif t >= 300 and diff != 0:
            # Overtime with a lead — game is essentially over
            p = 0.95 if diff > 0 else 0.05
        else:
            time_remaining = max(300 - t, 1.0)
            time_fraction = time_remaining / 300.0  # 1.0 at start, 0.0 at end
            # Moderate scaling: bigger swings as time runs out, but not absurdly so
            # At start (time_fraction=1): k ~= 0.8 per goal diff
            # At end (time_fraction~0.01): k ~= 3.0 per goal diff
            k = 0.8 + 2.2 * (1.0 - time_fraction)
            x = diff * k
            p = 1 / (1 + np.exp(-x))
        probs.append(p * 100)

    return pd.DataFrame({'Time': seconds, 'WinProb': probs, 'IsOT': is_ot})
            # Time pressure: base of 0.7 gives visible ~67% for a 1-goal lead
            # early on, ramping to ~90%+ in the final minute via quadratic curve
            time_fraction = 1.0 - (time_remaining / match_length)
            time_pressure = 0.7 + 1.8 * (time_fraction ** 2.0)
            x = diff * time_pressure
            p = 1 / (1 + np.exp(-x))
        probs.append(p * 100)

    return pd.DataFrame({'Time': seconds, 'WinProb': probs})

# --- 8. MATH: KICKOFFS ---
def calculate_kickoff_stats(manager, player_map, match_id=""):
    game = manager.game
    game_df = manager.get_data_frame()
    proto = manager.get_protobuf_data()
    kickoff_list = []
    
    kickoff_frames = getattr(game, 'kickoff_frames', [])
    all_hits = proto.game_stats.hits
    
    goal_frames = []
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            f = getattr(g, 'frame_number', getattr(g, 'frame', None))
            if f: goal_frames.append(f)

    for k_frame in kickoff_frames:
        kicker_name = "Unknown"
        kicker_team = "Unknown"
        spawn_loc = "Center"
        try:
            start_row = game_df.loc[k_frame]
            min_dist = 99999
            for pid, pname in player_map.items():
                if pname in start_row:
                    p_data = start_row[pname]
                    dist = np.sqrt(p_data['pos_x']**2 + p_data['pos_y']**2)
                    if dist < min_dist:
                        min_dist = dist
                        kicker_name = pname
                        if abs(p_data['pos_x']) < 50: spawn_loc = "Center"
                        elif abs(p_data['pos_x']) > 1500: spawn_loc = "Diagonal"
                        else: spawn_loc = "Off-Center"
            for p in proto.players:
                if p.name == kicker_name:
                    kicker_team = "Orange" if p.is_orange else "Blue"
                    break
        except (KeyError, IndexError) as e:
            logger.debug("Kickoff kicker lookup failed at frame %s: %s", k_frame, e)

        first_touch_frame = -1
        time_to_hit = 0.0
        boost_at_hit = 0
        for hit in all_hits:
            if hit.frame_number > k_frame:
                first_touch_frame = hit.frame_number
                time_to_hit = (first_touch_frame - k_frame) / float(REPLAY_FPS)
                break
        
        outcome = "Neutral"
        end_x, end_y = 0, 0
        if first_touch_frame != -1 and kicker_name != "Unknown":
            try:
                if kicker_name in game_df:
                    boost_row = game_df[kicker_name].loc[first_touch_frame]
                    if 'boost' in boost_row and pd.notna(boost_row['boost']):
                        boost_at_hit = int(boost_row['boost'])
            except (KeyError, IndexError, ValueError):
                pass
            
            check_frame = min(first_touch_frame + 90, game_df.index.max())
            try:
                ball_end = game_df['ball'].loc[check_frame]
                end_x, end_y = ball_end['pos_x'], ball_end['pos_y']
                if kicker_team == "Blue":
                    if end_y > 1000: outcome = "Win" 
                    elif end_y < -1000: outcome = "Loss"
                elif kicker_team == "Orange":
                    if end_y < -1000: outcome = "Win" 
                    elif end_y > 1000: outcome = "Loss"
            except (KeyError, IndexError):
                pass

        kickoff_goal = False
        for gf in goal_frames:
            if k_frame < gf <= k_frame + 150: 
                kickoff_goal = True
                break

        if kicker_name != "Unknown":
            kickoff_list.append({
                "MatchID": str(match_id), "Frame": k_frame, "Player": kicker_name, "Team": kicker_team,
                "Spawn": spawn_loc, "Time to Hit": round(time_to_hit, 2), "Boost": boost_at_hit,
                "Result": outcome, "Goal (5s)": kickoff_goal, "End_X": end_x, "End_Y": end_y
            })
    return pd.DataFrame(kickoff_list)

# --- 9. MATH: SHOTS & PASSING ---
def calculate_shot_data(manager, player_map):
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    hits = proto.game_stats.hits
    shot_list = []


    # Build a tight goal frame map: only the LAST hit before each goal gets credit
    # Map each goal to the exact scorer frame from metadata
    _pid_team_shot = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    goal_scorer_frames = {}  # frame -> team of scorer

    # Tag frames near known goals as metadata-confirmed goals
    # (45 frames = 1.5s before through 15 frames after the goal event)
    metadata_goal_frames = set()
 
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            f = getattr(g, 'frame_number', getattr(g, 'frame', None))
            if f:

                scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
                scorer_team = _pid_team_shot.get(scorer_pid, "Blue")
                goal_scorer_frames[f] = scorer_team

    # For each goal, find the last hit within 10 frames before the goal frame (the actual scoring touch)
    goal_hit_frames = set()
    for goal_frame in goal_scorer_frames:
        best_hit = None
        for hit in hits:
            if hit.player_id and goal_frame - 10 <= hit.frame_number <= goal_frame:
                best_hit = hit.frame_number
        if best_hit is not None:
            goal_hit_frames.add(best_hit)
        else:
            # Widen to 30 frames if no hit found very close
            for hit in hits:
                if hit.player_id and goal_frame - 30 <= hit.frame_number <= goal_frame:
                    best_hit = hit.frame_number
            if best_hit is not None:
                goal_hit_frames.add(best_hit)

                for i in range(f - 45, f + 15): metadata_goal_frames.add(i)
 

    for hit in hits:
        frame = hit.frame_number
        if not hit.player_id: continue
        # Trust carball's shot/goal detection as primary signal
        is_lib_shot = getattr(hit, 'is_shot', False)
        is_lib_goal = getattr(hit, 'is_goal', False)
        is_goal_hit = frame in goal_hit_frames
        is_physics_shot = False
        ball_pos, ball_vel = None, None

        if 'ball' in game_df and frame in game_df.index:
            try:
                ball_data = game_df['ball'].loc[frame]
                ball_pos = (ball_data['pos_x'], ball_data['pos_y'], ball_data['pos_z'])
                ball_vel = (ball_data['vel_x'], ball_data['vel_y'], ball_data['vel_z'])
                # Physics fallback: only count as a shot if the ball is in the
                # attacking third AND moving fast toward goal. This catches real
                # shots carball misses without flagging mid-field clears.
                pid = str(hit.player_id.id)
                shooter_team = "Unknown"
                for p in proto.players:
                    if str(p.id.id) == pid:
                        shooter_team = "Orange" if p.is_orange else "Blue"
                        break
                direction_sign = 1 if shooter_team == "Blue" else -1

                # Tighter physics check: ball moving fast toward goal AND in attacking half
                ball_toward_goal = ball_vel[1] * direction_sign > 0
                fast_enough = abs(ball_vel[1]) > 1200
                in_attacking_half = (ball_pos[1] * direction_sign) > 0
                if ball_toward_goal and fast_enough and in_attacking_half:
                    is_physics_shot = True
            except: pass

        # Only count if library says it's a shot/goal, or tight physics check passes
        if is_lib_shot or is_lib_goal or is_goal_hit or is_physics_shot:

                in_attacking_third = (ball_pos[1] * direction_sign) > 1700
                fast_toward_goal = (ball_vel[1] * direction_sign > 0) and (abs(ball_vel[1]) > 1400)
                if in_attacking_third and fast_toward_goal:
                    is_physics_shot = True
            except (KeyError, IndexError):
                pass

        if is_lib_shot or is_lib_goal or is_meta_goal or is_physics_shot:
 
            pid = str(hit.player_id.id)
            player_name = player_map.get(pid, "Unknown")
            shooter_team = "Unknown"
            for p in proto.players:
                if str(p.id.id) == pid:
                    shooter_team = "Orange" if p.is_orange else "Blue"
                    break
            target_y = 5120 if shooter_team == "Blue" else -5120

            if ball_pos and ball_vel:
                xg = calculate_xg_probability(ball_pos[0], ball_pos[1], ball_pos[2], ball_vel[0], ball_vel[1], ball_vel[2], target_y)
                is_big_chance = False
                defender_dist = 99999
                if xg > 0.40:
                    orange_players = [p.name for p in proto.players if p.is_orange]
                    blue_players = [p.name for p in proto.players if not p.is_orange]
                    defenders = orange_players if shooter_team == "Blue" else blue_players
                    try:
                        frame_data = game_df.loc[frame]
                        for d_name in defenders:
                            if d_name in frame_data:
                                d_pos = frame_data[d_name]
                                dist = np.sqrt((ball_pos[0]-d_pos['pos_x'])**2 + (ball_pos[1]-d_pos['pos_y'])**2)
                                if dist < defender_dist: defender_dist = dist
                    except (KeyError, IndexError):
                        pass
                    if defender_dist > 500: is_big_chance = True
                result = "Goal" if (is_lib_goal or is_goal_hit) else "Shot"
                shot_list.append({
                    "Player": player_name, "Team": shooter_team, "Frame": frame,
                    "xG": round(xg, 2), "Result": result, "BigChance": is_big_chance,
                    "X": ball_pos[0], "Y": ball_pos[1],
                    "Speed": int(np.sqrt(ball_vel[0]**2 + ball_vel[1]**2 + ball_vel[2]**2))
                })

    if shot_list:
        raw_df = pd.DataFrame(shot_list)

        # Deduplicate: within 1-second windows, keep highest xG per team
        raw_df['TimeGroup'] = (raw_df['Frame'] // 30)
        # Prefer goals over shots, then highest xG
        raw_df['_sort'] = raw_df['Result'].map({'Goal': 0, 'Shot': 1})
        final_df = raw_df.sort_values(['_sort', 'xG'], ascending=[True, False]).drop_duplicates(subset=['Team', 'TimeGroup'])
        final_df = final_df.drop(columns=['_sort'])

        # Dedup shots within 0.5s windows per player
        raw_df['TimeGroup'] = (raw_df['Frame'] // 15)
        shots_only = raw_df[raw_df['Result'] == 'Shot'].sort_values('xG', ascending=False).drop_duplicates(subset=['Player', 'TimeGroup', 'Result'])
        # Dedup goals using proximity — a single goal event can trigger
        # multiple detections across nearby frames. Keep the highest-xG
        # entry per cluster of goals within 90 frames (3s) of each other.
        goals_raw = raw_df[raw_df['Result'] == 'Goal'].sort_values('Frame').copy()
        goals_deduped = []
        last_kept_frame = -9999
        for _, row in goals_raw.iterrows():
            if row['Frame'] - last_kept_frame > 90:
                goals_deduped.append(row)
                last_kept_frame = row['Frame']
            else:
                # Same goal event — keep higher xG
                if row['xG'] > goals_deduped[-1]['xG']:
                    goals_deduped[-1] = row
        goals_only = pd.DataFrame(goals_deduped) if goals_deduped else pd.DataFrame(columns=raw_df.columns)
        final_df = pd.concat([shots_only, goals_only], ignore_index=True)
 
        return final_df
    return pd.DataFrame(columns=["Player", "Team", "Frame", "xG", "Result", "BigChance", "X", "Y"])

def calculate_advanced_passing(manager, player_map, shot_df, max_time_diff=2.0):
    proto = manager.get_protobuf_data()
    hits = proto.game_stats.hits
    game_df = manager.get_data_frame()
    team_map = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    pass_events = [] 
    last_hitter_id = None
    last_hit_time = -999
    last_hit_frame = 0
    shot_frames_by_team = {}
    if not shot_df.empty:
        for team_name in shot_df['Team'].unique():
            shot_frames_by_team[team_name] = set(shot_df[shot_df['Team'] == team_name]['Frame'].tolist())
    
    for hit in hits:
        if not hit.player_id: continue
        curr_id = str(hit.player_id.id)
        curr_time = hit.frame_number / float(REPLAY_FPS)
        curr_frame = hit.frame_number
        if last_hitter_id is not None and curr_id != last_hitter_id:
            if team_map.get(curr_id) == team_map.get(last_hitter_id):
                if (curr_time - last_hit_time) < max_time_diff:
                    sender = player_map.get(last_hitter_id, "Unknown")
                    receiver = player_map.get(curr_id, "Unknown")
                    team = team_map.get(curr_id, "Unknown")
                    xa_val = 0.0
                    try:
                        if curr_frame in game_df.index:
                            b_data = game_df['ball'].loc[curr_frame]
                            target_y = 5120 if team == "Blue" else -5120
                            xa_val = calculate_xg_probability(b_data['pos_x'], b_data['pos_y'], b_data['pos_z'], b_data['vel_x'], b_data['vel_y'], b_data['vel_z'], target_y)
                    except (KeyError, IndexError):
                        pass
                    is_key_pass = False
                    team_shots = shot_frames_by_team.get(team, set())
                    for sf in team_shots:
                        if curr_frame <= sf <= curr_frame + 90:
                            is_key_pass = True
                            break
                    try:
                        if last_hit_frame in game_df.index and curr_frame in game_df.index:
                            s_pos = game_df[sender].loc[last_hit_frame]
                            r_pos = game_df[receiver].loc[curr_frame]
                            pass_events.append({
                                'Sender': sender, 'Receiver': receiver, 'Team': team, 'xA': xa_val, 'KeyPass': is_key_pass,
                                'x1': s_pos['pos_x'], 'y1': s_pos['pos_y'], 'x2': r_pos['pos_x'], 'y2': r_pos['pos_y']
                            })
                    except (KeyError, IndexError):
                        pass
        last_hitter_id = curr_id
        last_hit_time = curr_time
        last_hit_frame = curr_frame
    return pd.DataFrame(pass_events)

# --- 10. MATH: AGGREGATE ---
def calculate_final_stats(manager, shot_df, pass_df):
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    stats = []
    total_hits = len(proto.game_stats.hits)
    player_hits = {}
    for h in proto.game_stats.hits:
        if h.player_id:
            pid = str(h.player_id.id)
            player_hits[pid] = player_hits.get(pid, 0) + 1
            
    # Pre-compute goals conceded per team for rating penalty
    goals_by_team = {"Blue": 0, "Orange": 0}
    if hasattr(proto, 'players'):
        for p in proto.players:
            t = "Orange" if p.is_orange else "Blue"
            goals_by_team[t] += p.goals
    goals_conceded = {"Blue": goals_by_team["Orange"], "Orange": goals_by_team["Blue"]}

    if hasattr(proto, 'players'):
        for player in proto.players:
            name = player.name
            team = "Orange" if player.is_orange else "Blue"
            p_shots = shot_df[shot_df['Player'] == name] if not shot_df.empty else pd.DataFrame()
            xg_sum = p_shots['xG'].sum() if not p_shots.empty else 0
            big_chances = len(p_shots[p_shots['BigChance'] == True]) if not p_shots.empty else 0
            p_passes = pass_df[pass_df['Sender'] == name] if not pass_df.empty else pd.DataFrame()
            xa_sum = p_passes['xA'].sum() if not p_passes.empty else 0
            key_passes = len(p_passes[p_passes['KeyPass'] == True]) if not p_passes.empty else 0
            hits = player_hits.get(str(player.id.id), 0)
            poss_pct = round((hits / total_hits) * 100, 1) if total_hits > 0 else 0
            conceded = goals_conceded.get(team, 0)
            raw_rating = 5.0 + (player.goals * 1.0) + (player.assists * 0.75) + (player.saves * 0.6) + (key_passes * 0.4) + (player.shots * 0.2) - (conceded * 0.3)
            final_rating = max(1.0, min(10.0, raw_rating))
            p_data = {
                "Name": name, "Team": team, "Goals": player.goals, "Assists": player.assists, "Saves": player.saves,
                "Shots": player.shots, "Score": player.score, "xG": round(xg_sum, 2), "xA": round(xa_sum, 2),
                "Big Chances": big_chances, "Key Passes": key_passes, "Possession": poss_pct,
                "Rating": round(final_rating, 1), "IsBot": getattr(player, 'is_bot', False),
                "Boost Used": 0, "Wasted Boost": 0, "Avg Speed": 0, "Time Supersonic": 0,
                "Pos_Def": 0, "Pos_Mid": 0, "Pos_Off": 0,
                "Wall_Time": 0, "Corner_Time": 0, "On_Wall_Time": 0, "Carry_Time": 0,
                "Overtime": False, "Luck": 0.0, "Timestamp": ""
            }
            if hasattr(player, 'stats') and hasattr(player.stats, 'boost'):
                p_data['Boost Used'] = int(getattr(player.stats.boost, 'boost_usage', 0))
                p_data['Wasted Boost'] = int(getattr(player.stats.boost, 'wasted_usage', 0))
            if name in game_df:
                pdf = game_df[name]
                if 'vel_x' in pdf.columns:
                    velocities = pdf[['vel_x', 'vel_y', 'vel_z']].to_numpy()
                    speeds = np.linalg.norm(velocities, axis=1)
                    p_data['Avg Speed'] = int(np.nanmean(speeds))

                    p_data['Time Supersonic'] = round(np.sum(speeds >= 2200) / 30.0, 2)
                if 'pos_y' in pdf.columns and 'pos_x' in pdf.columns:
                    x_pos = pdf['pos_x'].to_numpy()

                    p_data['Time Supersonic'] = round(np.sum(speeds >= 2200) / float(REPLAY_FPS), 2)
                if 'pos_y' in pdf.columns:
 
                    y_pos = pdf['pos_y'].to_numpy()
                    z_pos = pdf['pos_z'].to_numpy() if 'pos_z' in pdf.columns else np.zeros_like(y_pos)
                    total_f = len(y_pos)
                    if total_f > 0:
                        if team == "Blue":
                            def_cnt = np.sum(y_pos < -1700)
                            off_cnt = np.sum(y_pos > 1700)
                        else:
                            def_cnt = np.sum(y_pos > 1700)
                            off_cnt = np.sum(y_pos < -1700)
                        mid_cnt = total_f - def_cnt - off_cnt
                        p_data['Pos_Def'] = round((def_cnt/total_f)*100, 1)
                        p_data['Pos_Mid'] = round((mid_cnt/total_f)*100, 1)
                        p_data['Pos_Off'] = round((off_cnt/total_f)*100, 1)
                        # Granular zones
                        near_wall = np.abs(x_pos) > 3596  # within 500 of side wall
                        near_goal_end = np.abs(y_pos) > 4620  # within 500 of back wall
                        on_wall_height = z_pos > 200
                        p_data['Wall_Time'] = round((np.sum(near_wall) / total_f) * 100, 1)
                        p_data['Corner_Time'] = round((np.sum(near_wall & near_goal_end) / total_f) * 100, 1)
                        p_data['On_Wall_Time'] = round((np.sum(near_wall & on_wall_height) / total_f) * 100, 1)
                # Carry time: player within 300 units of ball moving in same direction
                if 'ball' in game_df and 'vel_x' in pdf.columns:
                    try:
                        ball_df = game_df['ball']
                        common_idx = pdf.index.intersection(ball_df.index)
                        if len(common_idx) > 100:
                            p_x = pdf.loc[common_idx, 'pos_x'].to_numpy()
                            p_y = pdf.loc[common_idx, 'pos_y'].to_numpy()
                            b_x = ball_df.loc[common_idx, 'pos_x'].to_numpy()
                            b_y = ball_df.loc[common_idx, 'pos_y'].to_numpy()
                            dist_to_ball = np.sqrt((p_x - b_x)**2 + (p_y - b_y)**2)
                            p_vx = pdf.loc[common_idx, 'vel_x'].to_numpy()
                            p_vy = pdf.loc[common_idx, 'vel_y'].to_numpy()
                            b_vx = ball_df.loc[common_idx, 'vel_x'].to_numpy()
                            b_vy = ball_df.loc[common_idx, 'vel_y'].to_numpy()
                            dot_product = p_vx * b_vx + p_vy * b_vy
                            carry_frames = np.sum((dist_to_ball < 300) & (dot_product > 0))
                            p_data['Carry_Time'] = round((carry_frames / len(common_idx)) * 100, 1)
                    except:
                        pass
            stats.append(p_data)
    return pd.DataFrame(stats)

# --- 10b. EXPORT PANEL BUILDERS ---
def render_panel_to_image(fig, width, height, scale=2):
    """Render a Plotly figure to a PIL Image via kaleido."""
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return PILImage.open(io.BytesIO(img_bytes))

def build_export_shot_map(shot_df, proto):
    """Shot map on pitch background for export."""
    fig = go.Figure()
    fig.update_layout(get_field_layout(""))
    fig.update_layout(title=None, margin=dict(l=0, r=0, t=0, b=0))
    if not shot_df.empty:
        _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
        for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
            t_shots = shot_df[(shot_df['Team'] == team) & (shot_df['Result'] == 'Shot')]
            t_goals = shot_df[(shot_df['Team'] == team) & (shot_df['Result'] == 'Goal')]
            if not t_shots.empty:
                fig.add_trace(go.Scatter(x=t_shots['X'], y=t_shots['Y'], mode='markers',
                    marker=dict(size=10, color=color, opacity=0.6, line=dict(width=1, color='white')),
                    name=f'{team} Shot', showlegend=False))
            if not t_goals.empty:
                fig.add_trace(go.Scatter(x=t_goals['X'], y=t_goals['Y'], mode='markers',
                    marker=dict(size=16, color=color, symbol='star', line=dict(width=2, color='white')),
                    name=f'{team} Goal', showlegend=False))
        big_chances = shot_df[shot_df['BigChance'] == True]
        if not big_chances.empty:
            fig.add_trace(go.Scatter(x=big_chances['X'], y=big_chances['Y'], mode='markers',
                marker=dict(size=25, color='rgba(0,0,0,0)', line=dict(width=2, color='yellow')),
                showlegend=False))
    return fig

def build_export_heatmap(game_df, player_name):
    """Player heatmap on pitch background for export."""
    fig = go.Figure()
    fig.update_layout(get_field_layout(""))
    fig.update_layout(title=None, margin=dict(l=0, r=0, t=0, b=0))
    if player_name in game_df:
        p_frames = game_df[player_name]
        if 'pos_z' in p_frames.columns:
            valid_pos = p_frames[p_frames['pos_z'] > 0]
            sampled = valid_pos.iloc[::3]
            fig.add_trace(go.Histogram2dContour(
                x=sampled['pos_x'], y=sampled['pos_y'],
                colorscale=[[0, 'rgba(0,0,0,0)'], [0.15, 'rgba(0,80,0,0.25)'],
                            [0.3, 'rgba(0,160,0,0.4)'], [0.5, 'rgba(80,200,0,0.5)'],
                            [0.7, 'rgba(200,220,0,0.6)'], [0.85, 'rgba(255,200,0,0.65)'],
                            [1.0, 'rgba(255,255,50,0.75)']],
                contours=dict(coloring='fill', showlines=False),
                ncontours=20, showscale=False, hoverinfo='skip'
            ))
    return fig

def build_export_scoreboard(df, shot_df, is_overtime):
    """Scoreboard table for export with stats and luck."""
    blue_goals = int(df[df['Team']=='Blue']['Goals'].sum())
    orange_goals = int(df[df['Team']=='Orange']['Goals'].sum())
    blue_luck = calculate_luck_percentage(shot_df, "Blue", blue_goals) if not shot_df.empty else 0
    orange_luck = calculate_luck_percentage(shot_df, "Orange", orange_goals) if not shot_df.empty else 0
    ot_str = " (OT)" if is_overtime else ""
    # Build header row
    header_vals = ['', 'Rating', 'G', 'A', 'S', 'Sh', 'xG', 'Poss%']
    blue_rows = []
    orange_rows = []
    for _, p in df[df['Team']=='Blue'].sort_values('Score', ascending=False).iterrows():
        blue_rows.append([p['Name'], f"{p['Rating']:.1f}", int(p['Goals']), int(p['Assists']),
                         int(p['Saves']), int(p['Shots']), f"{p['xG']:.2f}", f"{p['Possession']:.0f}%"])
    for _, p in df[df['Team']=='Orange'].sort_values('Score', ascending=False).iterrows():
        orange_rows.append([p['Name'], f"{p['Rating']:.1f}", int(p['Goals']), int(p['Assists']),
                           int(p['Saves']), int(p['Shots']), f"{p['xG']:.2f}", f"{p['Possession']:.0f}%"])
    # Combine into table columns
    all_rows = blue_rows + [['─'*6, '─'*4, '─', '─', '─', '─', '─'*4, '─'*4]] + orange_rows
    cols = list(zip(*all_rows)) if all_rows else [[] for _ in header_vals]
    # Team colors for cell font
    n_blue = len(blue_rows)
    n_orange = len(orange_rows)
    name_colors = ['#4da6ff'] * n_blue + ['#888'] + ['#ffb347'] * n_orange
    cell_colors = ['#4da6ff'] * n_blue + ['#888'] + ['#ffb347'] * n_orange
    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color='#2a2a2a', font=dict(color='white', size=13),
                    align='center', line_color='#444'),
        cells=dict(values=list(cols), fill_color='#1e1e1e',
                   font=dict(color=[name_colors] + [cell_colors] * (len(header_vals)-1), size=12),
                   align='center', line_color='#333', height=28)
    )])
    # Add score annotation
    score_text = f"<b>{blue_goals}</b>  -  <b>{orange_goals}</b>{ot_str}"
    fig.add_annotation(x=0.5, y=1.12, xref='paper', yref='paper', text=score_text,
                      font=dict(size=28, color='white'), showarrow=False)
    # Luck annotation
    luck_text = f"<span style='color:#4da6ff'>Luck: {blue_luck}%</span>    <span style='color:#ffb347'>Luck: {orange_luck}%</span>"
    fig.add_annotation(x=0.5, y=1.03, xref='paper', yref='paper', text=luck_text,
                      font=dict(size=13, color='white'), showarrow=False)
    fig.update_layout(
        paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
        margin=dict(l=10, r=10, t=80, b=10), font=dict(color='white')
    )
    return fig

def build_export_xg_timeline(shot_df, game_df, proto, is_overtime):
    """Cumulative xG timeline for export."""
    fig = go.Figure()
    if not shot_df.empty:
        sorted_shots = shot_df.sort_values('Frame').copy()
        sorted_shots['Time'] = sorted_shots['Frame'] / 30.0
        _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
        meta_goals = {"Blue": [], "Orange": []}
        if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
            for g in proto.game_metadata.goals:
                gf = getattr(g, 'frame_number', getattr(g, 'frame', 0))
                scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
                gteam = _pid_team.get(scorer_pid, "Blue")
                meta_goals[gteam].append(gf / 30.0)
        match_end = game_df.index.max() / 30.0
        for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
            team_shots = sorted_shots[sorted_shots['Team'] == team]
            if not team_shots.empty:
                times = [0] + team_shots['Time'].tolist() + [match_end]
                cum_xg = [0] + team_shots['xG'].cumsum().tolist()
                cum_xg.append(cum_xg[-1])
                fig.add_trace(go.Scatter(x=times, y=cum_xg, mode='lines', name=f"{team} xG",
                    line=dict(color=color, width=3, shape='hv'), showlegend=True))
            if meta_goals[team]:
                goal_times = sorted(meta_goals[team])
                goal_cum = []
                for gt in goal_times:
                    prior = team_shots[team_shots['Time'] <= gt]['xG'].sum() if not team_shots.empty else 0
                    goal_cum.append(prior)
                fig.add_trace(go.Scatter(x=goal_times, y=goal_cum, mode='markers', name=f"{team} ⚽",
                    marker=dict(size=12, color=color, symbol='star', line=dict(width=2, color='white')), showlegend=False))
    if is_overtime:
        fig.add_vline(x=300, line_dash="dash", line_color="rgba(255,204,0,0.5)")
    fig.update_layout(
        title=dict(text="Cumulative xG", font=dict(size=14, color='white')),
        xaxis=dict(title=dict(text="Time (s)", font=dict(size=10)), showgrid=False, color='#888'),
        yaxis=dict(title=dict(text="xG", font=dict(size=10)), showgrid=True, gridcolor='rgba(255,255,255,0.08)', color='#888'),
        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font=dict(color='white', size=10),
        margin=dict(l=40, r=10, t=35, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9))
    )
    return fig

def build_export_win_prob(manager, is_overtime):
    """Win probability chart for export."""
    win_prob_df = calculate_win_probability(manager)
    fig = go.Figure()
    if not win_prob_df.empty:
        fig.add_trace(go.Scatter(x=win_prob_df['Time'], y=win_prob_df['WinProb'], fill='tozeroy',
            mode='lines', line=dict(width=0), fillcolor='rgba(0, 123, 255, 0.25)', showlegend=False))
        fig.add_trace(go.Scatter(x=win_prob_df['Time'], y=[100]*len(win_prob_df), fill='tonexty',
            mode='none', fillcolor='rgba(255, 153, 0, 0.25)', showlegend=False))
        fig.add_trace(go.Scatter(x=win_prob_df['Time'], y=win_prob_df['WinProb'], mode='lines',
            line=dict(color='white', width=2), name='Win Prob', showlegend=False))
        fig.add_shape(type="line", x0=win_prob_df['Time'].min(), y0=50, x1=win_prob_df['Time'].max(), y1=50,
            line=dict(color="gray", width=1, dash="dot"))
    if is_overtime:
        fig.add_vline(x=300, line_dash="dash", line_color="rgba(255,204,0,0.5)")
    fig.update_layout(
        title=dict(text="Win Probability", font=dict(size=14, color='white')),
        yaxis=dict(title=dict(text="Blue Win %", font=dict(size=10)), range=[0, 100], showgrid=False, color='#888'),
        xaxis=dict(title=dict(text="Time (s)", font=dict(size=10)), showgrid=False, color='#888'),
        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font=dict(color='white', size=10),
        margin=dict(l=40, r=10, t=35, b=30)
    )
    return fig

def build_export_zones(df, focus_players):
    """Positional zone bar chart for export."""
    fig = go.Figure()
    players_to_show = focus_players if focus_players else df['Name'].tolist()[:2]
    colors = ['#007bff', '#ff9900', '#00cc96', '#AB63FA']
    for i, pname in enumerate(players_to_show[:3]):
        p_row = df[df['Name'] == pname]
        if p_row.empty:
            continue
        p = p_row.iloc[0]
        zones = ['Def', 'Mid', 'Off', 'Wall', 'Corner', 'On Wall', 'Carry']
        vals = [p.get('Pos_Def', 0), p.get('Pos_Mid', 0), p.get('Pos_Off', 0),
                p.get('Wall_Time', 0), p.get('Corner_Time', 0), p.get('On_Wall_Time', 0), p.get('Carry_Time', 0)]
        fig.add_trace(go.Bar(x=zones, y=vals, name=pname, marker_color=colors[i % len(colors)],
                            opacity=0.85))
    fig.update_layout(
        title=dict(text="Positional Zones (%)", font=dict(size=14, color='white')),
        barmode='group',
        xaxis=dict(showgrid=False, color='#888', tickfont=dict(size=9)),
        yaxis=dict(title=dict(text="%", font=dict(size=10)), showgrid=True, gridcolor='rgba(255,255,255,0.08)', color='#888'),
        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font=dict(color='white', size=10),
        margin=dict(l=35, r=10, t=35, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9))
    )
    return fig

def build_export_pressure(momentum_series, proto):
    """Pressure index strip for export."""
    fig = go.Figure()
    if not momentum_series.empty:
        x_time = momentum_series.index
        y_values = momentum_series.values
        fig.add_trace(go.Scatter(x=x_time, y=y_values.clip(min=0), fill='tozeroy', mode='none',
            fillcolor='rgba(0, 123, 255, 0.6)', showlegend=False))
        fig.add_trace(go.Scatter(x=x_time, y=y_values.clip(max=0), fill='tozeroy', mode='none',
            fillcolor='rgba(255, 153, 0, 0.6)', showlegend=False))
        # Goal markers from proto
        _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
        _pid_name = {str(p.id.id): p.name for p in proto.players}
        if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
            for g in proto.game_metadata.goals:
                gf = getattr(g, 'frame_number', getattr(g, 'frame', 0))
                scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
                gteam = _pid_team.get(scorer_pid, "Blue")
                time_sec = gf / 30.0
                tm = 1 if gteam == 'Blue' else -1
                fig.add_trace(go.Scatter(x=[time_sec], y=[85 * tm], mode='markers+text',
                    marker=dict(symbol='circle', size=8, color='white', line=dict(width=1, color='black')),
                    text="⚽", textposition="top center" if tm > 0 else "bottom center",
                    showlegend=False))
    fig.update_layout(
        title=dict(text="Pressure Index", font=dict(size=14, color='white')),
        yaxis=dict(range=[-105, 105], showgrid=False, zeroline=True, zerolinecolor='rgba(255,255,255,0.15)',
                   showticklabels=False),
        xaxis=dict(title=dict(text="Match Time (s)", font=dict(size=10)), showgrid=False, color='#888'),
        plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font=dict(color='white', size=10),
        margin=dict(l=10, r=10, t=30, b=25)
    )
    return fig

# --- 11. UI COMPONENTS ---
def render_scoreboard(df, shot_df=None, is_overtime=False):
    st.markdown("### 🏆 Final Scoreboard")
    blue_goals = df[df['Team']=='Blue']['Goals'].sum()
    orange_goals = df[df['Team']=='Orange']['Goals'].sum()
    ot_badge = " <span style='font-size: 0.4em; color: #ffcc00;'>(OT)</span>" if is_overtime else ""
    c1, c2, c3 = st.columns([1, 0.5, 1])
    with c1: st.markdown(f"<h1 style='text-align: center; color: #007bff; margin: 0;'>{blue_goals}</h1>", unsafe_allow_html=True)
    with c2: st.markdown(f"<h1 style='text-align: center; color: white; margin: 0;'>-{ot_badge}</h1>", unsafe_allow_html=True)
    with c3: st.markdown(f"<h1 style='text-align: center; color: #ff9900; margin: 0;'>{orange_goals}</h1>", unsafe_allow_html=True)
    # Luck % display
    if shot_df is not None and not shot_df.empty:
        blue_luck = calculate_luck_percentage(shot_df, "Blue", int(blue_goals))
        orange_luck = calculate_luck_percentage(shot_df, "Orange", int(orange_goals))
        lc1, lc2 = st.columns(2)
        with lc1:
            luck_color = "#00cc96" if blue_luck > 50 else "#ff6b6b" if blue_luck < 30 else "#ffcc00"
            st.markdown(f"<div style='text-align:center;'><span style='color:{luck_color}; font-size:1.1em;'>Luck: {blue_luck}%</span></div>", unsafe_allow_html=True)
        with lc2:
            luck_color = "#00cc96" if orange_luck > 50 else "#ff6b6b" if orange_luck < 30 else "#ffcc00"
            st.markdown(f"<div style='text-align:center;'><span style='color:{luck_color}; font-size:1.1em;'>Luck: {orange_luck}%</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    cols = ['Name', 'Rating', 'Score', 'Goals', 'Assists', 'Saves', 'Shots']
    col_blue, col_orange = st.columns(2)
    with col_blue:
        st.markdown("#### 🔵 Blue Team")
        st.dataframe(df[df['Team']=='Blue'][cols].sort_values(by='Score', ascending=False), use_container_width=True, hide_index=True)
    with col_orange:
        st.markdown("#### 🟠 Orange Team")
        st.dataframe(df[df['Team']=='Orange'][cols].sort_values(by='Score', ascending=False), use_container_width=True, hide_index=True)
    st.divider()

def render_dashboard(df, shot_df, pass_df):
    st.markdown("### 📊 Match Performance")
    blue_poss = df[df['Team']=='Blue']['Possession'].sum()
    orange_poss = df[df['Team']=='Orange']['Possession'].sum()
    total = blue_poss + orange_poss
    if total > 0:
        blue_pct = int((blue_poss / total) * 100)
        orange_pct = 100 - blue_pct
        st.write(f"**Possession**")
        st.markdown(f"""
            <div style="display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background-color: #007bff; width: {blue_pct}%;"></div>
                <div style="background-color: #ff9900; width: {orange_pct}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span style="color: #007bff; font-weight: bold;">{blue_pct}%</span>
                <span style="color: #ff9900; font-weight: bold;">{orange_pct}%</span>
            </div>
        """, unsafe_allow_html=True)
    st.divider()
    blue_df = df[df['Team']=='Blue']
    orange_df = df[df['Team']=='Orange']
    col1, col2 = st.columns(2)
    for col, team_df, team_name in [(col1, blue_df, "Blue"), (col2, orange_df, "Orange")]:
        with col:
            st.subheader(f"{'🔵' if team_name == 'Blue' else '🟠'} {team_name} Team")
            for _, p in team_df.iterrows():
                with st.expander(f"**{p['Name']}** (Rating: {p['Rating']})"):
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.caption("Attack")
                        st.write(f"⚽ Goals: {p['Goals']}")
                        st.write(f"🎯 xG: {p['xG']}")
                        st.write(f"🔥 Big Chances: {p['Big Chances']}")
                    with sc2:
                        st.caption("Playmaking")
                        st.write(f"👟 Assists: {p['Assists']}")
                        st.write(f"🧠 xA: {p['xA']}")
                        st.write(f"🔑 Key Passes: {p['Key Passes']}")
                    with sc3:
                        st.caption("Defense")
                        st.write(f"🛡️ Saves: {p['Saves']}")
                        st.write(f"⏱️ Poss: {p['Possession']}%")

# --- 12. MAIN APP FLOW ---
st.sidebar.header("Settings")
app_mode = st.sidebar.radio("Mode:", ["🔍 Single Match Analysis", "📈 Season Batch Processor"])
filter_ghosts = st.sidebar.checkbox("Hide Bots", value=True)
st.sidebar.divider()
pass_threshold = st.sidebar.slider("Pass Window (Seconds)", 1.0, 5.0, 2.0, 0.5)

if not SPROCKET_AVAILABLE:
    st.error(f"⚠️ Library Error: {IMPORT_ERROR}")
    st.stop()

# 
# MODE 1: SINGLE MATCH
# 
if app_mode == "🔍 Single Match Analysis":
    uploaded_file = st.file_uploader("Upload Replay", type=["replay"])
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        with st.spinner("Parsing Replay..."):
            manager, game_df, proto, parse_error = get_parsed_replay_data(file_bytes, file_name)

        if parse_error:
            st.error(f"Failed to parse replay: {parse_error}")
        elif manager:
            with st.spinner("Calculating Advanced Physics Stats..."):
                temp_map = {str(p.id.id): p.name for p in proto.players}
                shot_df = calculate_shot_data(manager, temp_map)
                momentum_series = calculate_contextual_momentum(manager, game_df, proto)
                pass_df = calculate_advanced_passing(manager, temp_map, shot_df, pass_threshold)
                kickoff_df = calculate_kickoff_stats(manager, temp_map)
                df = calculate_final_stats(manager, shot_df, pass_df)
                is_overtime = detect_overtime(manager)  # NEW
                if not df.empty:
                    df['Overtime'] = is_overtime
                    # Calculate per-team luck
                    for team in ["Blue", "Orange"]:
                        team_goals = int(df[df['Team']==team]['Goals'].sum())
                        luck_val = calculate_luck_percentage(shot_df, team, team_goals)
                        df.loc[df['Team']==team, 'Luck'] = luck_val
                if not df.empty and 'IsBot' in df.columns and filter_ghosts:
                    df = df[~df['IsBot']]

            all_players = sorted(list(temp_map.values()))
            default_focus = [p for p in ["Fueg", "Zelli197"] if p in all_players]
            focus_players = st.sidebar.multiselect("🎯 Focus Analysis On:", all_players, default=default_focus)

            render_scoreboard(df, shot_df, is_overtime)
            render_dashboard(df, shot_df, pass_df)
            
            t1, t2, t3, t3b, t4, t5, t6, t7 = st.tabs(["🚀 Kickoffs", "🌊 Match Narrative", "🎯 Shot Map", "🎬 Shot Viewer", "🕸️ Pass Map", "🔥 Heatmaps", "⚡ Speed", "📸 Export"])
            
            with t1:
                st.subheader("Kickoff Analysis")
                if not kickoff_df.empty:
                    disp_kickoff = kickoff_df.copy()
                    if focus_players: disp_kickoff = disp_kickoff[disp_kickoff['Player'].isin(focus_players)]
                    if not disp_kickoff.empty:
                        col_k1, col_k2 = st.columns(2)
                        with col_k1:
                            wins = len(disp_kickoff[disp_kickoff['Result'] == 'Win'])
                            total = len(disp_kickoff)
                            win_rate = int((wins/total)*100) if total > 0 else 0
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number", value = win_rate,
                                title = {'text': "Kickoff Win Rate (Selected)"},
                                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#00cc96"}}
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        with col_k2:

                            color_map = {"Win": "#00cc96", "Loss": "#EF553B", "Neutral": "#AB63FA"}
                            fig = go.Figure()
                            for outcome, color in color_map.items():
                                subset = disp_kickoff[disp_kickoff['Result'] == outcome]
                                if not subset.empty:
                                    fig.add_trace(go.Scatter(
                                        x=subset['End_X'], y=subset['End_Y'], mode='markers',
                                        marker=dict(size=12, color=color, opacity=0.85, line=dict(width=1, color='white')),
                                        name=outcome, text=subset['Player'],
                                        hovertemplate="%{text}<br>Result: " + outcome + "<extra></extra>"
                                    ))
                            fig.update_layout(get_field_layout("Kickoff Outcomes"))
                            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color='white')))
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("#### Kickoff Log")
                        disp_cols = ['Player', 'Spawn', 'Time to Hit', 'Boost', 'Result', 'Goal (5s)']
                        _style_fn = lambda x: 'color: green' if x == 'Win' or x == True else ('color: red' if x == 'Loss' else 'color: gray')
                        _styler = disp_kickoff[disp_cols].style
                        if hasattr(_styler, 'map'):
                            _styler = _styler.map(_style_fn, subset=['Result', 'Goal (5s)'])
                        else:
                            _styler = _styler.applymap(_style_fn, subset=['Result', 'Goal (5s)'])
                        st.dataframe(_styler, use_container_width=True)

                            result_colors = {"Win": "#00cc96", "Loss": "#EF553B", "Neutral": "#636efa"}
                            fig = go.Figure()
                            for result, color in result_colors.items():
                                subset = disp_kickoff[disp_kickoff['Result'] == result]
                                if not subset.empty:
                                    fig.add_trace(go.Scatter(
                                        x=subset['End_X'], y=subset['End_Y'],
                                        mode='markers',
                                        marker=dict(size=14, color=color, line=dict(width=1.5, color='white'), opacity=0.9),
                                        name=result,
                                        hovertemplate="Player: %{text}<br>Result: " + result + "<extra></extra>",
                                        text=subset['Player']
                                    ))
                            fig.update_layout(get_field_layout("Kickoff Outcomes"))
                            fig.update_layout(legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)'))
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("#### Kickoff Log")
                        disp_cols = ['Player', 'Spawn', 'Time to Hit', 'Boost', 'Result', 'Goal (5s)']
                        st.dataframe(disp_kickoff[disp_cols].style.map(lambda x: 'color: green' if x == 'Win' or x is True else ('color: red' if x == 'Loss' else 'color: gray'), subset=['Result', 'Goal (5s)']), use_container_width=True)
 
                    else: st.info("No kickoffs found for selected players.")
                else: st.info("No kickoff data found.")

            with t2:
                st.subheader("Match Narrative")
                # --- A. WIN PROBABILITY CHART ---
                try:
                    win_prob_df = calculate_win_probability(manager)
                    if not win_prob_df.empty:
                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Scatter(x=win_prob_df['Time'], y=win_prob_df['WinProb'], fill='tozeroy', mode='lines', line=dict(width=0), fillcolor='rgba(0, 123, 255, 0.2)', name='Blue Win %', showlegend=False))
                        fig_prob.add_trace(go.Scatter(x=win_prob_df['Time'], y=[100]*len(win_prob_df), fill='tonexty', mode='none', fillcolor='rgba(255, 153, 0, 0.2)', name='Orange Win %', showlegend=False))
                        fig_prob.add_trace(go.Scatter(x=win_prob_df['Time'], y=win_prob_df['WinProb'], mode='lines', line=dict(color='white', width=2), name='Win Probability'))
                        fig_prob.add_shape(type="line", x0=win_prob_df['Time'].min(), y0=50, x1=win_prob_df['Time'].max(), y1=50, line=dict(color="gray", width=1, dash="dot"))
                        if is_overtime:
                            fig_prob.add_vline(x=300, line_dash="dash", line_color="rgba(255,204,0,0.7)", annotation_text="OT Start")
                        fig_prob.update_layout(title="🏆 Win Probability" + (" (Overtime)" if is_overtime else ""), yaxis=dict(title="Blue Win %", range=[0, 100], showgrid=False), xaxis=dict(title="Time (Seconds)", showgrid=False), plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_prob, use_container_width=True)
                except Exception as e: st.error(f"Could not calculate Win Probability: {e}")
                st.divider()

                # --- A2. CUMULATIVE xG TIMELINE ---
                st.markdown("#### 📈 Cumulative xG Timeline")
                if not shot_df.empty:
                    sorted_shots = shot_df.sort_values('Frame').copy()
                    sorted_shots['Time'] = sorted_shots['Frame'] / 30.0
                    fig_xg = go.Figure()
                    # Build goal list from proto metadata (authoritative source for all goals)
                    meta_goals = {"Blue": [], "Orange": []}
                    # Build player ID -> team lookup
                    pid_team_map = {}
                    for p in proto.players:
                        pid_team_map[str(p.id.id)] = "Orange" if p.is_orange else "Blue"
                    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
                        for g in proto.game_metadata.goals:
                            gf = getattr(g, 'frame_number', getattr(g, 'frame', 0))
                            scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
                            gteam = pid_team_map.get(scorer_pid, "Blue")
                            meta_goals[gteam].append(gf / 30.0)
                    for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
                        team_shots = sorted_shots[sorted_shots['Team'] == team]
                        if not team_shots.empty:
                            times = [0] + team_shots['Time'].tolist()
                            cum_xg = [0] + team_shots['xG'].cumsum().tolist()
                            # Extend to end of match
                            match_end = game_df.index.max() / 30.0
                            times.append(match_end)
                            cum_xg.append(cum_xg[-1])
                            fig_xg.add_trace(go.Scatter(x=times, y=cum_xg, mode='lines', name=f"{team} xG", line=dict(color=color, width=3, shape='hv')))
                        # Overlay ALL actual goals from proto metadata
                        if meta_goals[team]:
                            goal_times = sorted(meta_goals[team])
                            goal_cum = []
                            for gt in goal_times:
                                if not team_shots.empty:
                                    prior = team_shots[team_shots['Time'] <= gt]['xG'].sum()
                                else:
                                    prior = 0
                                goal_cum.append(prior)
                            fig_xg.add_trace(go.Scatter(x=goal_times, y=goal_cum, mode='markers', name=f"{team} Goal", marker=dict(size=14, color=color, symbol='star', line=dict(width=2, color='white'))))
                    if is_overtime:
                        fig_xg.add_vline(x=300, line_dash="dash", line_color="rgba(255,204,0,0.7)", annotation_text="OT")
                    fig_xg.update_layout(title="Cumulative xG Over Time", xaxis=dict(title="Time (s)", showgrid=False), yaxis=dict(title="Cumulative xG", showgrid=True, gridcolor='rgba(255,255,255,0.1)'), plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=280, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_xg, use_container_width=True)
                st.divider()

                # --- B. MOMENTUM CHART ---
                st.markdown("#### 🌊 Pressure Index")
                if not momentum_series.empty:
                    fig = go.Figure()
                    x_time = momentum_series.index
                    y_values = momentum_series.values
                    fig.add_trace(go.Scatter(x=x_time, y=y_values.clip(min=0), fill='tozeroy', mode='none', name='Blue Pressure', fillcolor='rgba(0, 123, 255, 0.6)'))
                    fig.add_trace(go.Scatter(x=x_time, y=y_values.clip(max=0), fill='tozeroy', mode='none', name='Orange Pressure', fillcolor='rgba(255, 153, 0, 0.6)'))

                    # Use proto metadata for authoritative goal list
                    _pi_pid_team = {str(p.id.id): ("Orange" if p.is_orange else "Blue") for p in proto.players}
                    _pi_pid_name = {str(p.id.id): p.name for p in proto.players}
                    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
                        for g in proto.game_metadata.goals:
                            gf = getattr(g, 'frame_number', getattr(g, 'frame', 0))
                            scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
                            gteam = _pi_pid_team.get(scorer_pid, "Blue")
                            scorer_name = _pi_pid_name.get(scorer_pid, "Unknown")
                            time_sec = gf / 30.0
                            tm_multiplier = 1 if gteam == 'Blue' else -1
                            fig.add_trace(go.Scatter(x=[time_sec], y=[85 * tm_multiplier], mode='markers+text', marker=dict(symbol='circle', size=10, color='white', line=dict(width=1, color='black')), text="⚽", textposition="top center" if tm_multiplier > 0 else "bottom center", name=scorer_name, hoverinfo="text+name", showlegend=False))

                    if not shot_df.empty:
                        goals = shot_df[shot_df['Result'] == 'Goal']
                        for _, g in goals.iterrows():
                            time_sec = g['Frame'] / float(REPLAY_FPS)
                            tm_multiplier = 1 if g['Team'] == 'Blue' else -1
                            fig.add_trace(go.Scatter(x=[time_sec], y=[85 * tm_multiplier], mode='markers+text', marker=dict(symbol='circle', size=10, color='white', line=dict(width=1, color='black')), text="⚽", textposition="top center" if tm_multiplier > 0 else "bottom center", name=f"{g['Team']} Goal", hoverinfo="text+name", showlegend=False))
 
                    fig.update_layout(yaxis=dict(title="Pressure", range=[-105, 105], showgrid=False, zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'), xaxis=dict(title="Match Time (Seconds)", showgrid=False), plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with t3:
                if not shot_df.empty:
                    fig = go.Figure()
                    fig.update_layout(get_field_layout("Shot Map"))

                    # Team-colored shots and goals
                    for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
                        t_shots = shot_df[(shot_df['Team'] == team) & (shot_df['Result'] == 'Shot')]
                        t_goals = shot_df[(shot_df['Team'] == team) & (shot_df['Result'] == 'Goal')]
                        if not t_shots.empty:
                            fig.add_trace(go.Scatter(x=t_shots['X'], y=t_shots['Y'], mode='markers',
                                marker=dict(size=10, color=color, opacity=0.5),
                                name=f'{team} Shot', text=t_shots['Player'],
                                customdata=t_shots['xG'], hovertemplate="%{text}<br>xG: %{customdata:.2f}<extra></extra>"))
                        if not t_goals.empty:
                            fig.add_trace(go.Scatter(x=t_goals['X'], y=t_goals['Y'], mode='markers',
                                marker=dict(size=15, color=color, line=dict(width=2, color='white'), symbol='circle'),
                                name=f'{team} Goal', text=t_goals['Player'],
                                customdata=t_goals['xG'], hovertemplate="%{text}<br>xG: %{customdata:.2f}<extra></extra>"))
                    big_chances = shot_df[shot_df['BigChance'] == True]
                    if not big_chances.empty:
                        fig.add_trace(go.Scatter(x=big_chances['X'], y=big_chances['Y'], mode='markers',
                            marker=dict(size=25, color='rgba(0,0,0,0)', line=dict(width=2, color='yellow')),
                            name='Big Chance', hoverinfo='skip'))

                    team_colors = {"Blue": "#3399ff", "Orange": "#ff9900", "Unknown": "gray"}
                    goals = shot_df[shot_df['Result'] == 'Goal']
                    misses = shot_df[shot_df['Result'] == 'Shot']
                    big_chances = shot_df[shot_df['BigChance'] == True]
                    # Shots (misses) - colored by team, semi-transparent
                    for team_name, color in team_colors.items():
                        t_misses = misses[misses['Team'] == team_name]
                        if not t_misses.empty:
                            fig.add_trace(go.Scatter(
                                x=t_misses['X'], y=t_misses['Y'], mode='markers',
                                marker=dict(size=10, color=color, opacity=0.45),
                                name=f'{team_name} Shot',
                                customdata=np.stack([t_misses['Player'], t_misses['xG']], axis=-1),
                                hovertemplate="<b>%{customdata[0]}</b><br>xG: %{customdata[1]}<extra>Shot</extra>",
                            ))
                    # Goals - colored by team, larger with white border
                    for team_name, color in team_colors.items():
                        t_goals = goals[goals['Team'] == team_name]
                        if not t_goals.empty:
                            fig.add_trace(go.Scatter(
                                x=t_goals['X'], y=t_goals['Y'], mode='markers',
                                marker=dict(size=16, color=color, line=dict(width=2.5, color='white')),
                                name=f'{team_name} Goal',
                                customdata=np.stack([t_goals['Player'], t_goals['xG']], axis=-1),
                                hovertemplate="<b>%{customdata[0]}</b><br>xG: %{customdata[1]}<extra>Goal</extra>",
                            ))
                    # Big chance ring overlay
                    if not big_chances.empty:
                        fig.add_trace(go.Scatter(
                            x=big_chances['X'], y=big_chances['Y'], mode='markers',
                            marker=dict(size=26, color='rgba(0,0,0,0)', line=dict(width=2, color='yellow')),
                            name='Big Chance', hoverinfo='skip',
                        ))
 
                    st.plotly_chart(fig, use_container_width=True)

            with t3b:
                st.subheader("Frozen Frame Shot Viewer")
                if not shot_df.empty:
                    sorted_shots_ff = shot_df.sort_values('Frame').reset_index(drop=True)
                    shot_labels = [f"#{i+1}: {row['Player']} ({row['Result']}) - xG {row['xG']:.2f}" for i, row in sorted_shots_ff.iterrows()]
                    selected_shot_idx = st.selectbox("Select Shot:", range(len(shot_labels)), format_func=lambda i: shot_labels[i])
                    shot_row = sorted_shots_ff.iloc[selected_shot_idx]
                    frame = int(shot_row['Frame'])
                    # Build field with all player positions at this frame
                    fig_ff = go.Figure()
                    fig_ff.update_layout(get_field_layout(f"Frame {frame} | {shot_row['Player']} ({shot_row['Result']})"))
                    # Ball position
                    if 'ball' in game_df and frame in game_df.index:
                        ball_data = game_df['ball'].loc[frame]
                        fig_ff.add_trace(go.Scatter(x=[ball_data['pos_x']], y=[ball_data['pos_y']], mode='markers', marker=dict(size=16, color='white', symbol='circle', line=dict(width=2, color='black')), name='Ball'))
                        # Shot direction arrow
                        if 'vel_x' in game_df['ball'].columns:
                            arrow_scale = 0.3
                            fig_ff.add_annotation(x=ball_data['pos_x'] + ball_data['vel_x']*arrow_scale, y=ball_data['pos_y'] + ball_data['vel_y']*arrow_scale, ax=ball_data['pos_x'], ay=ball_data['pos_y'], xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=2, arrowcolor='#ffcc00')
                    # Player positions
                    for p in proto.players:
                        if p.name in game_df and frame in game_df.index:
                            try:
                                p_data = game_df[p.name].loc[frame]
                                color = '#007bff' if not p.is_orange else '#ff9900'
                                marker_sym = 'diamond' if p.name == shot_row['Player'] else 'circle'
                                marker_size = 16 if p.name == shot_row['Player'] else 12
                                fig_ff.add_trace(go.Scatter(x=[p_data['pos_x']], y=[p_data['pos_y']], mode='markers+text', marker=dict(size=marker_size, color=color, symbol=marker_sym, line=dict(width=1, color='white')), text=[p.name], textposition='top center', textfont=dict(size=9, color='white'), name=p.name, showlegend=False))
                            except:
                                pass
                    st.plotly_chart(fig_ff, use_container_width=True)
                    # Metadata panel
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Shooter", shot_row['Player'])
                    mc2.metric("xG", f"{shot_row['xG']:.2f}")
                    mc3.metric("Result", shot_row['Result'])
                    mc4.metric("Speed", f"{shot_row.get('Speed', 'N/A')} uu/s")
                    if shot_row.get('BigChance', False):
                        st.warning("Big Chance!")
                    # Navigation
                    nav1, nav2, nav3 = st.columns([1, 2, 1])
                    with nav1:
                        if selected_shot_idx > 0:
                            st.caption(f"Previous: {shot_labels[selected_shot_idx - 1]}")
                    with nav3:
                        if selected_shot_idx < len(shot_labels) - 1:
                            st.caption(f"Next: {shot_labels[selected_shot_idx + 1]}")
                else:
                    st.info("No shots detected in this match.")

            with t4:
                if not pass_df.empty:
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.write("#### Playmaker Leaderboard")
                        st.dataframe(pass_df.groupby('Sender')['xA'].sum().sort_values(ascending=False), use_container_width=True)
                    with col_b:
                        fig = go.Figure()
                        fig.update_layout(get_field_layout("Pass Map"))
                        pass_colors = {"Blue": "rgba(50,150,255,0.4)", "Orange": "rgba(255,160,50,0.4)"}
                        reg = pass_df[pass_df['KeyPass']==False]
                        # Draw regular passes colored by team
                        for team_name, color in pass_colors.items():
                            t_passes = reg[reg['Team'] == team_name]
                            if not t_passes.empty:
                                # Use a single trace with None separators for performance
                                xs, ys = [], []
                                for _, row in t_passes.iterrows():
                                    xs.extend([row['x1'], row['x2'], None])
                                    ys.extend([row['y1'], row['y2'], None])
                                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, width=1.5), name=f'{team_name} Pass'))
                        # Draw key passes
                        key = pass_df[pass_df['KeyPass']==True]
                        if not key.empty:
                            xs, ys = [], []
                            for _, row in key.iterrows():
                                xs.extend([row['x1'], row['x2'], None])
                                ys.extend([row['y1'], row['y2'], None])
                            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='gold', width=3), name='Key Pass'))
                        fig.update_layout(legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)'))
                        st.plotly_chart(fig, use_container_width=True)

            with t5:
                if isinstance(game_df.columns, pd.MultiIndex): all_cols = game_df.columns.levels[0].tolist()
                else: all_cols = game_df.columns.tolist()
                valid_players = [p for p in all_cols if p in df['Name'].values]
                target = st.selectbox("Select Player:", valid_players)
                if target:
                    p_frames = game_df[target]
                    if 'pos_z' in p_frames.columns:

                        valid_pos = p_frames[p_frames['pos_z'] > 0]
                        # Downsample for performance (every 3rd frame is plenty for KDE)
                        sampled = valid_pos.iloc[::3]
                        # Sofascore-style smooth KDE heatmap with contour fill
                        fig = go.Figure()
                        fig.add_trace(go.Histogram2dContour(
                            x=sampled['pos_x'], y=sampled['pos_y'],
                            colorscale=[[0, 'rgba(0,0,0,0)'], [0.15, 'rgba(0,80,0,0.25)'],
                                        [0.3, 'rgba(0,160,0,0.4)'], [0.5, 'rgba(80,200,0,0.5)'],
                                        [0.7, 'rgba(200,220,0,0.6)'], [0.85, 'rgba(255,200,0,0.65)'],
                                        [1.0, 'rgba(255,255,50,0.75)']],
                            contours=dict(coloring='fill', showlines=False),
                            ncontours=20, showscale=False,
                            hoverinfo='skip'

                        valid_pos = p_frames[p_frames['pos_z'] > 0].dropna(subset=['pos_x', 'pos_y'])
                        sofascore_scale = [
                            [0.0, 'rgba(0,0,0,0)'],
                            [0.15, 'rgba(20,60,0,0.35)'],
                            [0.3, 'rgba(60,100,0,0.5)'],
                            [0.45, 'rgba(120,140,0,0.6)'],
                            [0.6, 'rgba(180,180,0,0.65)'],
                            [0.75, 'rgba(220,210,0,0.75)'],
                            [0.9, 'rgba(255,240,50,0.85)'],
                            [1.0, 'rgba(255,255,120,0.95)'],
                        ]
                        fig = go.Figure()
                        fig.add_trace(go.Histogram2dContour(
                            x=valid_pos['pos_x'], y=valid_pos['pos_y'],
                            colorscale=sofascore_scale,
                            ncontours=20,
                            contours=dict(coloring='fill', showlines=False),
                            showscale=False,
                            hoverinfo='skip',
 
                        ))
                        fig.update_layout(get_field_layout(f"{target} Heatmap"))
                        st.plotly_chart(fig, use_container_width=True)

            with t6:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(df, x='Name', y='Time Supersonic', color='Team', title="Time Supersonic (s)", color_discrete_map={"Blue": "#007bff", "Orange": "#ff9000"})
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.bar(df, x='Name', y='Boost Used', color='Team', title="Total Boost Used", color_discrete_map={"Blue": "#007bff", "Orange": "#ff9000"})
                    st.plotly_chart(fig, use_container_width=True)

            with t7:
                st.subheader("Composite Dashboard Export")
                can_export = KALEIDO_AVAILABLE and PIL_AVAILABLE
                if not KALEIDO_AVAILABLE:
                    st.warning("Install `kaleido` for image export: `pip install kaleido`")
                if not PIL_AVAILABLE:
                    st.warning("Install `Pillow` for image export: `pip install Pillow`")
                if can_export:
                    # Let user pick heatmap player
                    heatmap_player_opts = sorted(list(temp_map.values()))
                    default_hp = focus_players[0] if focus_players else (heatmap_player_opts[0] if heatmap_player_opts else None)
                    hp_idx = heatmap_player_opts.index(default_hp) if default_hp in heatmap_player_opts else 0
                    heatmap_player = st.selectbox("Heatmap Player:", heatmap_player_opts, index=hp_idx, key="export_hp")
                    if st.button("Generate Dashboard Image"):
                        with st.spinner("Rendering 7 panels... this may take a moment"):
                            try:
                                # All dimensions in LOGICAL pixels. scale=2 gives us 2x resolution.
                                # We render each panel at scale=2, then resize back to logical px for stitching.
                                S = 2  # scale factor for quality
                                PAD = 8
                                TITLE_H = 50
                                ROW1_H = 480
                                ROW2_H = 300
                                ROW3_H = 220
                                PITCH_W = 380
                                CANVAS_W = 1800
                                SCORE_W = CANVAS_W - 2 * PITCH_W - 2 * PAD
                                COL3_W = (CANVAS_W - 2 * PAD) // 3
                                CANVAS_H = TITLE_H + ROW1_H + ROW2_H + ROW3_H + 3 * PAD

                                # --- Build all 7 panels ---
                                fig_shotmap = build_export_shot_map(shot_df, proto)
                                fig_heatmap = build_export_heatmap(game_df, heatmap_player)
                                fig_scoreboard = build_export_scoreboard(df, shot_df, is_overtime)
                                fig_xg = build_export_xg_timeline(shot_df, game_df, proto, is_overtime)
                                fig_winprob = build_export_win_prob(manager, is_overtime)
                                fig_zones = build_export_zones(df, focus_players)
                                fig_pressure = build_export_pressure(momentum_series, proto)

                                # --- Render each to PIL Image at scale, then resize to logical ---
                                def _render(fig, w, h):
                                    img = render_panel_to_image(fig, w, h, scale=S)
                                    return img.resize((w, h), PILImage.LANCZOS)

                                img_shotmap = _render(fig_shotmap, PITCH_W, ROW1_H)
                                img_heatmap = _render(fig_heatmap, PITCH_W, ROW1_H)
                                img_scoreboard = _render(fig_scoreboard, SCORE_W, ROW1_H)
                                img_xg = _render(fig_xg, COL3_W, ROW2_H)
                                img_winprob = _render(fig_winprob, COL3_W, ROW2_H)
                                img_zones = _render(fig_zones, COL3_W, ROW2_H)
                                img_pressure = _render(fig_pressure, CANVAS_W, ROW3_H)

                                # --- Composite onto dark canvas ---
                                canvas = PILImage.new('RGB', (CANVAS_W, CANVAS_H), color=(30, 30, 30))

                                # Title bar
                                blue_goals_exp = int(df[df['Team']=='Blue']['Goals'].sum())
                                orange_goals_exp = int(df[df['Team']=='Orange']['Goals'].sum())
                                ot_label = " (OT)" if is_overtime else ""
                                draw = ImageDraw.Draw(canvas)
                                try:
                                    title_font = ImageFont.truetype("arial.ttf", 24)
                                    sub_font = ImageFont.truetype("arial.ttf", 14)
                                except:
                                    title_font = ImageFont.load_default()
                                    sub_font = title_font
                                title_text = f"Match Dashboard  |  Blue {blue_goals_exp} - {orange_goals_exp} Orange{ot_label}"
                                bbox = draw.textbbox((0, 0), title_text, font=title_font)
                                tw = bbox[2] - bbox[0]
                                draw.text(((CANVAS_W - tw) // 2, 6), title_text, fill=(255, 255, 255), font=title_font)
                                subtitle = "RL Pro Analytics"
                                bbox2 = draw.textbbox((0, 0), subtitle, font=sub_font)
                                sw = bbox2[2] - bbox2[0]
                                draw.text(((CANVAS_W - sw) // 2, 34), subtitle, fill=(136, 136, 136), font=sub_font)

                                # Row 1: Shot Map | Scoreboard | Heatmap
                                y1 = TITLE_H
                                canvas.paste(img_shotmap, (0, y1))
                                canvas.paste(img_scoreboard, (PITCH_W + PAD, y1))
                                canvas.paste(img_heatmap, (PITCH_W + SCORE_W + 2 * PAD, y1))

                                # Row 2: xG Timeline | Win Prob | Zones
                                y2 = y1 + ROW1_H + PAD
                                canvas.paste(img_xg, (0, y2))
                                canvas.paste(img_winprob, (COL3_W + PAD, y2))
                                canvas.paste(img_zones, (2 * COL3_W + 2 * PAD, y2))

                                # Row 3: Pressure Index (full width)
                                y3 = y2 + ROW2_H + PAD
                                canvas.paste(img_pressure, (0, y3))

                                # --- Upscale final canvas for quality, then export ---
                                final_canvas = canvas.resize((CANVAS_W * 2, CANVAS_H * 2), PILImage.LANCZOS)
                                buf = io.BytesIO()
                                final_canvas.save(buf, format='PNG', optimize=True)
                                final_bytes = buf.getvalue()
                                st.image(final_bytes, caption="Match Dashboard", use_container_width=True)
                                st.download_button("Download Dashboard PNG", data=final_bytes, file_name="match_dashboard.png", mime="image/png")
                            except Exception as e:
                                st.error(f"Export failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())

# 
# MODE 2: SEASON BATCH
# 
elif app_mode == "📈 Season Batch Processor":
    existing_stats, existing_kickoffs = load_data()
    session_gap = st.sidebar.slider("Session Gap (minutes)", 10, 120, 30, 5)
    uploaded_files = st.file_uploader("Upload Batch (Auto-Saves to Database)", type=["replay"], accept_multiple_files=True)
    if uploaded_files:
        new_stats_list = []
        new_kickoffs_list = []
        bar = st.progress(0)
        for i, f in enumerate(uploaded_files):
            bytes_data = f.read()
            manager, game_df, proto, parse_error = get_parsed_replay_data(bytes_data, f.name)
            if parse_error:
                logger.warning("Skipping %s: %s", f.name, parse_error)
            elif manager:
                game_id = getattr(manager.game, 'id', None) or f.name
                existing_ids = existing_stats['MatchID'].astype(str).values if not existing_stats.empty else []
                if str(game_id) in existing_ids:
                    bar.progress((i+1)/len(uploaded_files))
                    continue
                try:
                    temp_map = {str(p.id.id): p.name for p in proto.players}
                    shot_df = calculate_shot_data(manager, temp_map)
                    pass_df = calculate_advanced_passing(manager, temp_map, shot_df, pass_threshold)
                    stats = calculate_final_stats(manager, shot_df, pass_df)
                    kickoff_df = calculate_kickoff_stats(manager, temp_map, game_id)
                    if not stats.empty and 'IsBot' in stats.columns and filter_ghosts: stats = stats[~stats['IsBot']]
                    if not stats.empty:
                        stats['MatchID'] = str(game_id)
                        blue_g = stats[stats['Team']=='Blue']['Goals'].sum()
                        orange_g = stats[stats['Team']=='Orange']['Goals'].sum()

                        winner = "Blue" if blue_g > orange_g else "Orange"
                        stats['Won'] = stats['Team'] == winner
                        # Overtime detection
                        is_ot = detect_overtime(manager)  # NEW
                        stats['Overtime'] = is_ot
                        # Luck calculation
                        for team in ["Blue", "Orange"]:
                            team_goals = int(stats[stats['Team']==team]['Goals'].sum())
                            luck_val = calculate_luck_percentage(shot_df, team, team_goals)
                            stats.loc[stats['Team']==team, 'Luck'] = luck_val
                        # Timestamp
                        ts = get_match_timestamp(proto)
                        stats['Timestamp'] = ts

                        if blue_g > orange_g:
                            stats['Won'] = stats['Team'] == "Blue"
                        elif orange_g > blue_g:
                            stats['Won'] = stats['Team'] == "Orange"
                        else:
                            stats['Won'] = False
 
                        new_stats_list.append(stats)
                    if not kickoff_df.empty:
                        kickoff_df['MatchID'] = str(game_id)
                        new_kickoffs_list.append(kickoff_df)
                except Exception as e:
                    logger.warning("Failed to process replay %s: %s", f.name, e)
            bar.progress((i+1)/len(uploaded_files))
        bar.empty()
        if new_stats_list:
            new_stats_df = pd.concat(new_stats_list, ignore_index=True)
            new_kickoffs_df = pd.concat(new_kickoffs_list, ignore_index=True) if new_kickoffs_list else pd.DataFrame()
            save_data(new_stats_df, new_kickoffs_df)
            st.success(f"Added {len(new_stats_df['MatchID'].unique())} new matches to database!")
            existing_stats, existing_kickoffs = load_data()
        else:
            if uploaded_files: st.info("No new matches found (duplicates skipped).")

    if not existing_stats.empty:
        season = existing_stats
        season_kickoffs = existing_kickoffs
        # Ensure backward compatibility for new columns
        for col, default in [('Overtime', False), ('Luck', 0.0), ('Timestamp', ''), ('Wall_Time', 0.0), ('Corner_Time', 0.0), ('On_Wall_Time', 0.0), ('Carry_Time', 0.0)]:
            if col not in season.columns:
                season[col] = default
        st.divider()
        st.write(f"📚 **Career Database:** {len(season['MatchID'].unique())} Matches Loaded")
        players = sorted(season['Name'].unique())
        ix = 0
        if "Fueg" in players: ix = players.index("Fueg")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1: hero = st.selectbox("Select Yourself:", players, index=ix)
        with col_sel2:
            mate_opts = ["None"] + [p for p in players if p != hero]
            mate_ix = 0
            if "Zelli197" in mate_opts: mate_ix = mate_opts.index("Zelli197")
            teammate = st.selectbox("Compare With (Optional):", mate_opts, index=mate_ix)

        hero_df = season[season['Name'] == hero].reset_index(drop=True)
        hero_df['GameNum'] = hero_df.index + 1
        csv = season.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📥 Download Career Data", csv, "career_stats.csv", "text/csv")

        # Session detection
        def detect_sessions(df, gap_minutes=30):
            """Auto-detect play sessions using timestamp gap threshold."""
            if 'Timestamp' not in df.columns or df['Timestamp'].isna().all() or (df['Timestamp'] == '').all():
                # Fallback: assign sequential session IDs by MatchID order
                match_ids = df['MatchID'].unique()
                session_map = {}
                sid = 1
                for mid in match_ids:
                    session_map[mid] = sid
                    # Increment session every 5 matches as rough grouping when no timestamps
                    if len([k for k, v in session_map.items() if v == sid]) >= 5:
                        sid += 1
                df['SessionID'] = df['MatchID'].map(session_map)
                return df
            # Try to parse timestamps
            try:
                match_ts = df.drop_duplicates('MatchID')[['MatchID', 'Timestamp']].copy()
                match_ts['ParsedTime'] = pd.to_datetime(match_ts['Timestamp'], errors='coerce')
                match_ts = match_ts.sort_values('ParsedTime')
                sid = 1
                session_map = {}
                prev_time = None
                for _, row in match_ts.iterrows():
                    if prev_time is not None and pd.notna(row['ParsedTime']):
                        diff = (row['ParsedTime'] - prev_time).total_seconds() / 60.0
                        if diff > gap_minutes:
                            sid += 1
                    session_map[row['MatchID']] = sid
                    if pd.notna(row['ParsedTime']):
                        prev_time = row['ParsedTime']
                df['SessionID'] = df['MatchID'].map(session_map)
            except:
                df['SessionID'] = 1
            return df

        hero_df = detect_sessions(hero_df, session_gap)

        # Summary metrics
        ot_count = int(hero_df['Overtime'].sum()) if 'Overtime' in hero_df.columns else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Games", len(hero_df))
        c2.metric("Win Rate", f"{(hero_df['Won'].sum()/len(hero_df)*100):.1f}%")
        if 'Rating' in hero_df: c3.metric("Avg Rating", f"{hero_df['Rating'].mean():.2f}")
        if 'xG' in hero_df: c4.metric("Total xG", f"{hero_df['xG'].sum():.2f}")
        c5.metric("OT Games", ot_count)

        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📈 Performance", "🚀 Season Kickoffs", "🧠 Playstyle", "🕸️ Radar", "📊 Log", "🗓️ Sessions", "📸 Export"])
        with t1:
            st.subheader("Performance Trends")
            metric = st.selectbox("Metric:", ['Rating', 'Score', 'Goals', 'Assists', 'Saves', 'xG', 'Avg Speed', 'Luck', 'Carry_Time'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hero_df['GameNum'], y=hero_df[metric], name=hero, line=dict(color='#007bff', width=3)))
            if teammate != "None":
                mate_df = season[season['Name'] == teammate].reset_index(drop=True)
                mate_df['GameNum'] = mate_df.index + 1
                if metric in mate_df.columns:
                    fig.add_trace(go.Scatter(x=mate_df['GameNum'], y=mate_df[metric], name=teammate, line=dict(color='#ff9900', width=3, dash='dot')))
            # Mark OT games
            if 'Overtime' in hero_df.columns:
                ot_games = hero_df[hero_df['Overtime'] == True]
                if not ot_games.empty:
                    fig.add_trace(go.Scatter(x=ot_games['GameNum'], y=ot_games[metric], mode='markers', marker=dict(size=10, color='#ffcc00', symbol='diamond'), name='OT Game'))
            fig.update_layout(title=f"{metric} over Time", plot_bgcolor='#1e1e1e')
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            st.subheader("Season Kickoff Meta")
            if not season_kickoffs.empty:
                hero_k = season_kickoffs[season_kickoffs['Player'] == hero]
                c_a, c_b = st.columns(2)
                with c_a:
                    wins = len(hero_k[hero_k['Result'] == 'Win'])
                    win_rate = int((wins/len(hero_k))*100) if len(hero_k) > 0 else 0
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number", value = win_rate, title = {'text': f"{hero} Win Rate"},
                        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#00cc96"}}
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                with c_b:
                    spawn_grp = hero_k.groupby('Spawn')['Result'].apply(lambda x: (x=='Win').mean()*100).reset_index(name='WinRate')
                    fig = px.bar(spawn_grp, x='Spawn', y='WinRate', title="Win Rate by Spawn Location", color_discrete_sequence=['#636efa'])
                    st.plotly_chart(fig, use_container_width=True)

                st.write("#### Season Kickoff Outcomes")
                color_map_k = {"Win": "#00cc96", "Loss": "#EF553B", "Neutral": "#AB63FA"}
                fig = go.Figure()
                for outcome, color in color_map_k.items():
                    subset = hero_k[hero_k['Result'] == outcome]
                    if not subset.empty:
                        fig.add_trace(go.Scatter(
                            x=subset['End_X'], y=subset['End_Y'], mode='markers',
                            marker=dict(size=10, color=color, opacity=0.7, line=dict(width=1, color='white')),
                            name=outcome
                        ))
                fig.update_layout(get_field_layout(f"Where {hero}'s Kickoffs Go (Season View)"))
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color='white')))

                st.write("#### Season Kickoff Outcome Map")
                result_colors = {"Win": "#00cc96", "Loss": "#EF553B", "Neutral": "#636efa"}
                fig = go.Figure()
                for result, color in result_colors.items():
                    subset = hero_k[hero_k['Result'] == result]
                    if not subset.empty:
                        fig.add_trace(go.Scatter(
                            x=subset['End_X'], y=subset['End_Y'],
                            mode='markers',
                            marker=dict(size=10, color=color, line=dict(width=1, color='white'), opacity=0.7),
                            name=f"{result} ({len(subset)})",
                            hovertemplate="Result: " + result + "<extra></extra>",
                        ))
                fig.update_layout(get_field_layout(f"Where {hero}'s Kickoffs Go (Season View)"))
                fig.update_layout(legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)'))
 
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No kickoff data collected.")
        with t3:
            st.subheader("Positional Tendencies")
            if 'Pos_Def' in hero_df:
                col_pie, col_bar = st.columns(2)
                with col_pie:
                    avg_def = hero_df['Pos_Def'].mean()
                    avg_mid = hero_df['Pos_Mid'].mean()
                    avg_off = hero_df['Pos_Off'].mean()
                    fig = px.pie(names=['Defense', 'Midfield', 'Offense'], values=[avg_def, avg_mid, avg_off], title=f"{hero} Avg Positioning", color_discrete_sequence=['#EF553B', '#FFA15A', '#00CC96'])
                    st.plotly_chart(fig, use_container_width=True)
                with col_bar:
                    # Granular zones bar chart
                    zone_data = {
                        'Zone': ['Wall Play', 'Corner Time', 'On Wall', 'Ball Carry'],
                        'Time %': [
                            hero_df['Wall_Time'].mean() if 'Wall_Time' in hero_df else 0,
                            hero_df['Corner_Time'].mean() if 'Corner_Time' in hero_df else 0,
                            hero_df['On_Wall_Time'].mean() if 'On_Wall_Time' in hero_df else 0,
                            hero_df['Carry_Time'].mean() if 'Carry_Time' in hero_df else 0
                        ]
                    }
                    fig_zones = px.bar(zone_data, x='Zone', y='Time %', title=f"{hero} Granular Zones (Avg %)", color='Zone', color_discrete_sequence=['#636efa', '#EF553B', '#AB63FA', '#00CC96'])
                    fig_zones.update_layout(showlegend=False, plot_bgcolor='#1e1e1e')
                    st.plotly_chart(fig_zones, use_container_width=True)
                # Comparison with teammate
                if teammate != "None":
                    mate_df_ps = season[season['Name'] == teammate]
                    st.markdown("#### Comparison")
                    comp_zones = pd.DataFrame({
                        'Zone': ['Wall', 'Corner', 'On Wall', 'Carry'] * 2,
                        'Player': [hero]*4 + [teammate]*4,
                        'Time %': [
                            hero_df['Wall_Time'].mean(), hero_df['Corner_Time'].mean(),
                            hero_df['On_Wall_Time'].mean(), hero_df['Carry_Time'].mean(),
                            mate_df_ps['Wall_Time'].mean() if 'Wall_Time' in mate_df_ps else 0,
                            mate_df_ps['Corner_Time'].mean() if 'Corner_Time' in mate_df_ps else 0,
                            mate_df_ps['On_Wall_Time'].mean() if 'On_Wall_Time' in mate_df_ps else 0,
                            mate_df_ps['Carry_Time'].mean() if 'Carry_Time' in mate_df_ps else 0
                        ]
                    })
                    fig_comp = px.bar(comp_zones, x='Zone', y='Time %', color='Player', barmode='group', color_discrete_map={hero: '#007bff', teammate: '#ff9900'})
                    fig_comp.update_layout(plot_bgcolor='#1e1e1e')
                    st.plotly_chart(fig_comp, use_container_width=True)
        with t4:
            st.subheader("Player Comparison Radar")
            categories = ['Rating', 'Goals', 'Assists', 'Saves', 'Shots', 'xG', 'xA', 'Big Chances']
            # Normalize each category to 0-100 scale across all players for fair comparison
            all_avgs = season.groupby('Name')[categories].mean()
            cat_max = all_avgs.max()
            cat_max = cat_max.replace(0, 1)  # avoid division by zero
            hero_avg = hero_df[categories].mean()
            hero_norm = (hero_avg / cat_max * 100).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=hero_norm.values, theta=categories, fill='toself', name=hero, line=dict(color='#007bff')))
            if teammate != "None":
                mate_df = season[season['Name'] == teammate]
                mate_avg = mate_df[categories].mean()
                mate_norm = (mate_avg / cat_max * 100).fillna(0)
                fig.add_trace(go.Scatterpolar(r=mate_norm.values, theta=categories, fill='toself', name=teammate, line=dict(color='#ff9900')))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=500)
            st.plotly_chart(fig, use_container_width=True)
        with t5:

            st.subheader("Match Log")
            # Enhanced log with OT and Luck columns
            log_cols = ['GameNum', 'MatchID', 'Won', 'Goals', 'Assists', 'Saves', 'Rating', 'xG', 'Luck']
            if 'Overtime' in hero_df.columns:
                log_cols.insert(3, 'Overtime')
            available_cols = [c for c in log_cols if c in hero_df.columns]
            def style_log(row):
                styles = [''] * len(row)
                for i, col in enumerate(row.index):
                    if col == 'Won':
                        styles[i] = 'color: green' if row[col] else 'color: red'
                    elif col == 'Overtime' and row[col]:
                        styles[i] = 'color: #ffcc00; font-weight: bold'
                return styles
            st.dataframe(hero_df[available_cols].style.apply(style_log, axis=1), use_container_width=True)
        with t6:
            st.subheader("Session Analytics")
            if 'SessionID' in hero_df.columns:
                session_ids = sorted(hero_df['SessionID'].unique())
                st.write(f"**{len(session_ids)} sessions detected** (gap threshold: {session_gap} min)")
                session_summary = []
                for sid in session_ids:
                    s_df = hero_df[hero_df['SessionID'] == sid]
                    wins = int(s_df['Won'].sum())
                    games = len(s_df)
                    wr = round((wins/games)*100, 1) if games > 0 else 0
                    avg_r = round(s_df['Rating'].mean(), 2) if 'Rating' in s_df else 0
                    avg_luck = round(s_df['Luck'].mean(), 1) if 'Luck' in s_df else 0
                    ot_count_s = int(s_df['Overtime'].sum()) if 'Overtime' in s_df else 0
                    session_summary.append({
                        'Session': sid, 'Games': games, 'Wins': wins,
                        'Win Rate %': wr, 'Avg Rating': avg_r,
                        'Avg Luck %': avg_luck, 'OT Games': ot_count_s
                    })
                summary_df = pd.DataFrame(session_summary)
                st.dataframe(summary_df.style.background_gradient(subset=['Win Rate %'], cmap='RdYlGn', vmin=0, vmax=100), use_container_width=True, hide_index=True)
                # Session performance chart
                fig_sess = go.Figure()
                fig_sess.add_trace(go.Bar(x=summary_df['Session'], y=summary_df['Win Rate %'], name='Win Rate %', marker_color='#00cc96'))
                fig_sess.add_trace(go.Scatter(x=summary_df['Session'], y=summary_df['Avg Rating'], name='Avg Rating', yaxis='y2', line=dict(color='#ff9900', width=3), mode='lines+markers'))
                fig_sess.update_layout(
                    title="Session Performance Overview",
                    xaxis=dict(title="Session #", dtick=1),
                    yaxis=dict(title="Win Rate %", range=[0, 100]),
                    yaxis2=dict(title="Avg Rating", overlaying='y', side='right', range=[0, 10]),
                    plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    legend=dict(x=0.01, y=0.99)
                )
                st.plotly_chart(fig_sess, use_container_width=True)
            else:
                st.info("No session data available. Upload replays to generate sessions.")
        with t7:
            st.subheader("Season Dashboard Export")
            if KALEIDO_AVAILABLE:
                if st.button("Generate Season Dashboard Image"):
                    with st.spinner("Rendering..."):
                        comp_fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=["Rating Over Time", "Positioning", "Win Rate by Session", "Radar"],
                            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "scatterpolar"}]],
                            vertical_spacing=0.15, horizontal_spacing=0.1
                        )
                        # Rating trend
                        comp_fig.add_trace(go.Scatter(x=hero_df['GameNum'], y=hero_df['Rating'], line=dict(color='#007bff', width=2), name=hero, showlegend=False), row=1, col=1)
                        # Positioning bars
                        pos_vals = [hero_df['Pos_Def'].mean(), hero_df['Pos_Mid'].mean(), hero_df['Pos_Off'].mean()]
                        comp_fig.add_trace(go.Bar(x=['Def', 'Mid', 'Off'], y=pos_vals, marker_color=['#EF553B', '#FFA15A', '#00CC96'], showlegend=False), row=1, col=2)
                        # Session win rates
                        if 'SessionID' in hero_df.columns and len(session_ids) > 1:
                            comp_fig.add_trace(go.Bar(x=summary_df['Session'], y=summary_df['Win Rate %'], marker_color='#00cc96', showlegend=False), row=2, col=1)
                        # Radar
                        categories_exp = ['Rating', 'Goals', 'Assists', 'Saves', 'Shots', 'xG']
                        hero_avg_exp = hero_df[categories_exp].mean()
                        comp_fig.add_trace(go.Scatterpolar(r=hero_avg_exp.values.tolist() + [hero_avg_exp.values[0]], theta=categories_exp + [categories_exp[0]], fill='toself', line=dict(color='#007bff'), name=hero, showlegend=False), row=2, col=2)
                        comp_fig.update_layout(height=800, width=1200, plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font=dict(color='white', size=11), title_text=f"{hero} Season Dashboard")
                        img_bytes = comp_fig.to_image(format="png", width=1200, height=800, scale=2)
                        st.image(img_bytes, caption="Season Dashboard", use_container_width=True)
                        st.download_button("Download Season Dashboard PNG", data=img_bytes, file_name="season_dashboard.png", mime="image/png")
            else:
                st.warning("Install `kaleido` for image export: `pip install kaleido`")
                st.code("pip install kaleido", language="bash")

            st.dataframe(hero_df.style.map(lambda x: 'color: green' if x else 'color: red', subset=['Won']), use_container_width=True)
 
