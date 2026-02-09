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

SKLEARN_AVAILABLE = False
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

import json

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
WIN_PROB_MODEL_FILE = "win_prob_model.json"

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
    match_duration_s = max_frame / float(REPLAY_FPS)
    is_ot = match_duration_s > 305
    frames = np.arange(0, max_frame, REPLAY_FPS)
    seconds = frames / float(REPLAY_FPS)

    blue_goals = []
    orange_goals = []
    # Build player ID -> team lookup for reliable team detection
    _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            frame = getattr(g, 'frame_number', getattr(g, 'frame', 0))
            scorer_pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
            team = _pid_team.get(scorer_pid, "Orange" if getattr(g.player_id, 'is_orange', False) else "Blue")
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

# --- 7b. WIN PROBABILITY MODEL (TRAINED) ---
def extract_win_prob_training_data(manager, game_df, proto):
    """Extract per-second training samples from a replay for the win probability model."""
    max_frame = game_df.index.max()
    match_duration = max_frame / float(REPLAY_FPS)
    if match_duration < 60:
        return pd.DataFrame()
    blue_goals_total = sum(p.goals for p in proto.players if not p.is_orange)
    orange_goals_total = sum(p.goals for p in proto.players if p.is_orange)
    if blue_goals_total == orange_goals_total:
        return pd.DataFrame()
    blue_won = 1 if blue_goals_total > orange_goals_total else 0

    _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    blue_gf, orange_gf = [], []
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            frame = getattr(g, 'frame_number', getattr(g, 'frame', 0))
            pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
            team = _pid_team.get(pid, "Blue")
            (blue_gf if team == "Blue" else orange_gf).append(frame)

    ball_df = game_df['ball'] if 'ball' in game_df else None
    ball_idx = ball_df.index.values if ball_df is not None else np.array([])
    ball_y_arr = ball_df['pos_y'].values if ball_df is not None and 'pos_y' in ball_df.columns else np.array([])

    samples = []
    reg_end = min(int(max_frame), 300 * REPLAY_FPS)
    for f in range(0, reg_end, REPLAY_FPS):
        b_score = sum(1 for gf in blue_gf if gf <= f)
        o_score = sum(1 for gf in orange_gf if gf <= f)
        time_remaining = max(300 - f / float(REPLAY_FPS), 0)
        ball_y_norm = 0.0
        if len(ball_idx) > 0:
            bi = min(np.searchsorted(ball_idx, f), len(ball_y_arr) - 1)
            ball_y_norm = ball_y_arr[bi] / 5120.0
        boost_diff = 0.0
        bl_b, or_b = [], []
        for p in proto.players:
            if p.name in game_df and 'boost' in game_df[p.name].columns:
                pdf = game_df[p.name]
                pi = np.searchsorted(pdf.index.values, f)
                pi = min(pi, len(pdf) - 1)
                if pi >= 0:
                    try:
                        bv = pdf.iloc[pi]['boost']
                        if not np.isnan(bv):
                            (or_b if p.is_orange else bl_b).append(bv)
                    except:
                        pass
        if bl_b and or_b:
            boost_diff = np.mean(bl_b) - np.mean(or_b)
        samples.append({'score_diff': b_score - o_score, 'time_remaining': time_remaining,
                        'boost_diff': boost_diff, 'ball_y_normalized': ball_y_norm, 'blue_won': blue_won})
    return pd.DataFrame(samples)

def train_win_probability_model(training_data, min_samples=500):
    """Train a logistic regression win probability model. Returns (model, scaler) or (None, None)."""
    if not SKLEARN_AVAILABLE or training_data is None or len(training_data) < min_samples:
        return None, None
    features = ['score_diff', 'time_remaining', 'boost_diff', 'ball_y_normalized']
    X = training_data[features].values
    y = training_data['blue_won'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler

def save_win_prob_model(model, scaler, filepath=WIN_PROB_MODEL_FILE):
    """Save trained model coefficients as JSON."""
    if model is None or scaler is None:
        return
    data = {
        'coef': model.coef_.tolist(),
        'intercept': model.intercept_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'n_training_samples': int(model.n_features_in_) if hasattr(model, 'n_features_in_') else 0
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_win_prob_model(filepath=WIN_PROB_MODEL_FILE):
    """Load a trained win probability model from JSON. Returns (model, scaler) or (None, None)."""
    if not SKLEARN_AVAILABLE or not os.path.exists(filepath):
        return None, None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        model = LogisticRegression()
        model.coef_ = np.array(data['coef'])
        model.intercept_ = np.array(data['intercept'])
        model.classes_ = np.array([0, 1])
        scaler = StandardScaler()
        scaler.mean_ = np.array(data['scaler_mean'])
        scaler.scale_ = np.array(data['scaler_scale'])
        scaler.n_features_in_ = len(scaler.mean_)
        return model, scaler
    except Exception:
        return None, None

def calculate_win_probability_trained(manager, model, scaler):
    """Calculate win probability using a trained model. Falls back to hand-tuned if model is None."""
    if model is None or scaler is None:
        return calculate_win_probability(manager), False
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    max_frame = game_df.index.max()
    match_duration_s = max_frame / float(REPLAY_FPS)
    is_ot = match_duration_s > 305
    frames = np.arange(0, max_frame, REPLAY_FPS)
    seconds = frames / float(REPLAY_FPS)

    _pid_team = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    blue_gf, orange_gf = [], []
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            frame = getattr(g, 'frame_number', getattr(g, 'frame', 0))
            pid = str(g.player_id.id) if hasattr(g.player_id, 'id') else ""
            team = _pid_team.get(pid, "Blue")
            (blue_gf if team == "Blue" else orange_gf).append(min(frame, max_frame))

    ball_df = game_df['ball'] if 'ball' in game_df else None
    ball_idx = ball_df.index.values if ball_df is not None else np.array([])
    ball_y_arr = ball_df['pos_y'].values if ball_df is not None and 'pos_y' in ball_df.columns else np.array([])

    probs = []
    for f, t in zip(frames, seconds):
        b_score = sum(1 for gf in blue_gf if gf <= f)
        o_score = sum(1 for gf in orange_gf if gf <= f)
        diff = b_score - o_score
        if t >= 300 and diff == 0:
            probs.append(50.0)
            continue
        if t >= 300 and diff != 0:
            probs.append(95.0 if diff > 0 else 5.0)
            continue
        time_remaining = max(300 - t, 1.0)
        ball_y_norm = 0.0
        if len(ball_idx) > 0:
            bi = min(np.searchsorted(ball_idx, f), len(ball_y_arr) - 1)
            ball_y_norm = ball_y_arr[bi] / 5120.0
        boost_diff = 0.0
        bl_b, or_b = [], []
        for p in proto.players:
            if p.name in game_df and 'boost' in game_df[p.name].columns:
                pdf = game_df[p.name]
                pi = min(np.searchsorted(pdf.index.values, f), len(pdf) - 1)
                if pi >= 0:
                    try:
                        bv = pdf.iloc[pi]['boost']
                        if not np.isnan(bv):
                            (or_b if p.is_orange else bl_b).append(bv)
                    except:
                        pass
        if bl_b and or_b:
            boost_diff = np.mean(bl_b) - np.mean(or_b)
        X = np.array([[diff, time_remaining, boost_diff, ball_y_norm]])
        X_scaled = scaler.transform(X)
        p = model.predict_proba(X_scaled)[0][1] * 100
        probs.append(p)

    return pd.DataFrame({'Time': seconds, 'WinProb': probs, 'IsOT': is_ot}), True

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

# --- 9b. MATH: AERIAL STATS ---
def calculate_aerial_stats(manager, player_map):
    """Compute per-player aerial stats: aerial hits, avg height, time airborne."""
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    hits = proto.game_stats.hits
    team_map = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    AERIAL_HEIGHT = 500  # unreal units — roughly above crossbar height

    aerial_data = {}  # pid -> {hits, heights[], team, name}
    for hit in hits:
        if not hit.player_id:
            continue
        pid = str(hit.player_id.id)
        frame = hit.frame_number
        if 'ball' in game_df and frame in game_df.index:
            try:
                ball_z = game_df['ball'].loc[frame]['pos_z']
                if ball_z >= AERIAL_HEIGHT:
                    if pid not in aerial_data:
                        aerial_data[pid] = {'hits': 0, 'heights': [], 'team': team_map.get(pid, 'Unknown'),
                                            'name': player_map.get(pid, 'Unknown')}
                    aerial_data[pid]['hits'] += 1
                    aerial_data[pid]['heights'].append(float(ball_z))
            except (KeyError, IndexError):
                pass

    # Time airborne per player (pos_z > 300 = off ground, lower than aerial threshold)
    airborne_data = {}
    for p in proto.players:
        name = p.name
        if name in game_df:
            try:
                z = game_df[name]['pos_z']
                airborne_frames = int((z > 300).sum())
                airborne_data[name] = round(airborne_frames / float(REPLAY_FPS), 1)
            except (KeyError, AttributeError):
                airborne_data[name] = 0.0

    # Total hits per player for aerial %
    total_hits_per_player = {}
    for hit in hits:
        if hit.player_id:
            pid = str(hit.player_id.id)
            total_hits_per_player[pid] = total_hits_per_player.get(pid, 0) + 1

    results = []
    for p in proto.players:
        pid = str(p.id.id)
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        a = aerial_data.get(pid, {'hits': 0, 'heights': []})
        total_h = total_hits_per_player.get(pid, 0)
        results.append({
            'Name': name, 'Team': team,
            'Aerial Hits': a['hits'],
            'Aerial %': round((a['hits'] / total_h * 100), 1) if total_h > 0 else 0,
            'Avg Aerial Height': int(np.mean(a['heights'])) if a['heights'] else 0,
            'Max Aerial Height': int(max(a['heights'])) if a['heights'] else 0,
            'Time Airborne (s)': airborne_data.get(name, 0.0),
        })
    return pd.DataFrame(results)

# --- 9c. MATH: RECOVERY TIME ---
def calculate_recovery_time(manager, player_map):
    """After each hit, measure how many seconds until the player reaches supersonic (2200 uu/s)."""
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    hits = proto.game_stats.hits
    team_map = {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    SUPERSONIC = 2200
    MAX_RECOVERY_FRAMES = 5 * REPLAY_FPS  # cap at 5 seconds

    recovery_events = []  # per-hit recovery times
    player_recoveries = {}  # pid -> list of recovery times

    for hit in hits:
        if not hit.player_id:
            continue
        pid = str(hit.player_id.id)
        name = player_map.get(pid, 'Unknown')
        frame = hit.frame_number
        if name not in game_df or 'vel_x' not in game_df[name].columns:
            continue
        try:
            pdf = game_df[name]
            end_frame = min(frame + MAX_RECOVERY_FRAMES, game_df.index.max())
            window = pdf.loc[frame:end_frame]
            if window.empty:
                continue
            vels = window[['vel_x', 'vel_y', 'vel_z']].to_numpy()
            speeds = np.linalg.norm(vels, axis=1)
            supersonic_idx = np.where(speeds >= SUPERSONIC)[0]
            if len(supersonic_idx) > 0:
                recovery_frames = supersonic_idx[0]
                recovery_sec = recovery_frames / float(REPLAY_FPS)
            else:
                recovery_sec = MAX_RECOVERY_FRAMES / float(REPLAY_FPS)  # didn't reach supersonic
            if pid not in player_recoveries:
                player_recoveries[pid] = []
            player_recoveries[pid].append(recovery_sec)
        except (KeyError, IndexError):
            pass

    results = []
    for p in proto.players:
        pid = str(p.id.id)
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        times = player_recoveries.get(pid, [])
        results.append({
            'Name': name, 'Team': team,
            'Avg Recovery (s)': round(np.mean(times), 2) if times else 0,
            'Fast Recoveries': sum(1 for t in times if t < 1.0),
            'Total Hits': len(times),
            'Recovery < 1s %': round(sum(1 for t in times if t < 1.0) / len(times) * 100, 1) if times else 0,
        })
    return pd.DataFrame(results)

# --- 9d. MATH: DEFENSIVE PRESSURE / SHADOW DEFENSE ---
def calculate_defensive_pressure(manager, game_df, proto):
    """Track time each player spends in shadow defense position:
    between the ball and their own goal, moving in same direction as ball."""
    team_map = {p.name: "Orange" if p.is_orange else "Blue" for p in proto.players}
    if 'ball' not in game_df:
        return pd.DataFrame()

    results = []
    ball_df = game_df['ball']
    sample_step = 3  # every 3rd frame for performance

    for p in proto.players:
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        own_goal_y = -5120 if team == "Blue" else 5120
        direction = -1 if team == "Blue" else 1  # toward own goal

        if name not in game_df:
            results.append({'Name': name, 'Team': team, 'Shadow %': 0, 'Pressure Time (s)': 0})
            continue

        try:
            pdf = game_df[name]
            common_idx = pdf.index.intersection(ball_df.index)[::sample_step]
            if len(common_idx) < 30:
                results.append({'Name': name, 'Team': team, 'Shadow %': 0, 'Pressure Time (s)': 0})
                continue

            p_y = pdf.loc[common_idx, 'pos_y'].to_numpy()
            p_x = pdf.loc[common_idx, 'pos_x'].to_numpy()
            b_y = ball_df.loc[common_idx, 'pos_y'].to_numpy()
            b_x = ball_df.loc[common_idx, 'pos_x'].to_numpy()

            # Player is between ball and own goal (closer to own goal than ball)
            if team == "Blue":
                between = p_y < b_y  # player is closer to blue goal (negative y)
            else:
                between = p_y > b_y  # player is closer to orange goal (positive y)

            # Player is in defensive half
            if team == "Blue":
                in_def_half = p_y < 0
            else:
                in_def_half = p_y > 0

            # Player is within reasonable x-range of ball (not on far side of field)
            x_close = np.abs(p_x - b_x) < 2500

            # Retreating: player velocity toward own goal
            if 'vel_y' in pdf.columns:
                p_vy = pdf.loc[common_idx, 'vel_y'].to_numpy()
                if team == "Blue":
                    retreating = p_vy < -200  # moving toward blue goal
                else:
                    retreating = p_vy > 200  # moving toward orange goal
            else:
                retreating = np.ones(len(common_idx), dtype=bool)

            shadow_frames = np.sum(between & in_def_half & x_close & retreating)
            total_frames = len(common_idx)
            shadow_pct = round((shadow_frames / total_frames) * 100, 1)
            pressure_seconds = round(shadow_frames * sample_step / float(REPLAY_FPS), 1)

            results.append({
                'Name': name, 'Team': team,
                'Shadow %': shadow_pct,
                'Pressure Time (s)': pressure_seconds,
            })
        except (KeyError, AttributeError):
            results.append({'Name': name, 'Team': team, 'Shadow %': 0, 'Pressure Time (s)': 0})

    return pd.DataFrame(results)

# --- 9e. MATH: SHOT QUALITY CONCEDED (xG-Against) ---
def calculate_xg_against(manager, player_map, shot_df):
    """For each shot, find the closest defender and assign xG-against to them."""
    if shot_df.empty:
        return pd.DataFrame()

    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    team_map = {p.name: "Orange" if p.is_orange else "Blue" for p in proto.players}
    orange_players = [p.name for p in proto.players if p.is_orange]
    blue_players = [p.name for p in proto.players if not p.is_orange]

    xga_per_player = {}  # name -> list of xG values conceded
    for p in proto.players:
        xga_per_player[p.name] = []

    for _, shot in shot_df.iterrows():
        frame = int(shot['Frame'])
        shooter_team = shot['Team']
        xg = shot['xG']
        # Defenders are the opposing team
        defenders = blue_players if shooter_team == "Orange" else orange_players

        if frame not in game_df.index:
            continue

        # Find closest defender to ball at shot frame
        closest_defender = None
        min_dist = 99999
        try:
            ball_x, ball_y = shot['X'], shot['Y']
            for d_name in defenders:
                if d_name in game_df:
                    d_data = game_df[d_name].loc[frame]
                    dist = np.sqrt((ball_x - d_data['pos_x'])**2 + (ball_y - d_data['pos_y'])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_defender = d_name
        except (KeyError, IndexError):
            continue

        if closest_defender:
            xga_per_player[closest_defender].append({
                'xG': xg, 'result': shot['Result'], 'dist': round(min_dist),
                'frame': frame
            })

    results = []
    for p in proto.players:
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        events = xga_per_player.get(name, [])
        xg_vals = [e['xG'] for e in events]
        goals_conceded = sum(1 for e in events if e['result'] == 'Goal')
        results.append({
            'Name': name, 'Team': team,
            'Shots Faced': len(events),
            'xGA': round(sum(xg_vals), 2),
            'Goals Conceded (nearest)': goals_conceded,
            'Avg Dist to Shot': int(np.mean([e['dist'] for e in events])) if events else 0,
            'High xG Faced': sum(1 for e in events if e['xG'] > 0.3),
        })
    return pd.DataFrame(results)

# --- 9f. MATH: VAEP (Valuing Actions by Estimating Probabilities) ---
def estimate_scoring_threat(ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, team, nearest_own_dist, nearest_opp_dist):
    """Estimate scoring threat for a given game state. Returns [0, 0.99]."""
    target_y = 5120.0 if team == "Blue" else -5120.0
    dist_to_goal = np.sqrt(ball_x**2 + (target_y - ball_y)**2)
    positional_threat = np.exp(-0.0004 * dist_to_goal)
    vel_towards = ball_vy if team == "Blue" else -ball_vy
    vel_bonus = max(0.0, vel_towards / 4000.0) * 0.3
    dist_diff = nearest_opp_dist - nearest_own_dist
    possession_factor = 1.0 / (1.0 + np.exp(-dist_diff / 500.0))
    proximity_bonus = 0.2 * max(0.0, 1.0 - dist_to_goal / 2000.0) if dist_to_goal < 2000 else 0.0
    height_factor = 1.0 if ball_z < 500 else 0.85
    threat = (positional_threat * 0.5 + possession_factor * 0.3 + vel_bonus + proximity_bonus) * height_factor
    return max(0.0, min(threat, 0.99))

def calculate_vaep(manager, player_map, shot_df):
    """Calculate VAEP for each touch. Returns (vaep_df, vaep_summary)."""
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    max_frame = game_df.index.max()
    player_teams = {str(p.id.id): ("Orange" if p.is_orange else "Blue") for p in proto.players}
    player_names = {str(p.id.id): p.name for p in proto.players}

    ball_df = game_df['ball'] if 'ball' in game_df else None
    if ball_df is None:
        return pd.DataFrame(), pd.DataFrame()

    # Pre-compute arrays for fast lookup
    ball_frames = ball_df.index.values
    ball_x = ball_df['pos_x'].values
    ball_y = ball_df['pos_y'].values
    ball_z = ball_df['pos_z'].values if 'pos_z' in ball_df.columns else np.zeros(len(ball_df))
    ball_vx = ball_df['vel_x'].values if 'vel_x' in ball_df.columns else np.zeros(len(ball_df))
    ball_vy = ball_df['vel_y'].values if 'vel_y' in ball_df.columns else np.zeros(len(ball_df))
    ball_vz = ball_df['vel_z'].values if 'vel_z' in ball_df.columns else np.zeros(len(ball_df))

    player_pos = {}
    for p in proto.players:
        if p.name in game_df:
            pdf = game_df[p.name]
            if 'pos_x' in pdf.columns:
                player_pos[p.name] = {
                    'team': "Orange" if p.is_orange else "Blue",
                    'x': pdf['pos_x'].values, 'y': pdf['pos_y'].values,
                    'frames': pdf.index.values
                }

    def _ball_at(frame):
        bi = min(np.searchsorted(ball_frames, frame), len(ball_x) - 1)
        if bi < 0:
            return None
        return ball_x[bi], ball_y[bi], ball_z[bi], ball_vx[bi], ball_vy[bi], ball_vz[bi]

    def _nearest_dists(frame, team):
        bs = _ball_at(frame)
        if bs is None:
            return 5000.0, 5000.0
        bx, by = bs[0], bs[1]
        own_min, opp_min = 10000.0, 10000.0
        for pname, pd_info in player_pos.items():
            pi = min(np.searchsorted(pd_info['frames'], frame), len(pd_info['x']) - 1)
            if pi < 0:
                continue
            dist = np.sqrt((pd_info['x'][pi] - bx)**2 + (pd_info['y'][pi] - by)**2)
            if pd_info['team'] == team:
                own_min = min(own_min, dist)
            else:
                opp_min = min(opp_min, dist)
        return own_min, opp_min

    touches = []
    for hit in proto.game_stats.hits:
        if not hit.player_id:
            continue
        pid = str(hit.player_id.id)
        team = player_teams.get(pid)
        name = player_names.get(pid)
        if not team or not name:
            continue
        frame = hit.frame_number
        f_before = max(0, frame - 1)
        f_after = min(max_frame, frame + 5)
        sb = _ball_at(f_before)
        sa = _ball_at(f_after)
        if sb is None or sa is None:
            continue
        own_b, opp_b = _nearest_dists(f_before, team)
        own_a, opp_a = _nearest_dists(f_after, team)
        threat_before = estimate_scoring_threat(sb[0], sb[1], sb[2], sb[3], sb[4], sb[5], team, own_b, opp_b)
        threat_after = estimate_scoring_threat(sa[0], sa[1], sa[2], sa[3], sa[4], sa[5], team, own_a, opp_a)
        vaep = round(threat_after - threat_before, 4)
        touches.append({'Player': name, 'Team': team, 'Frame': frame,
                        'Time': round(frame / REPLAY_FPS, 1), 'VAEP': vaep,
                        'BallX': sa[0], 'BallY': sa[1]})

    vaep_df = pd.DataFrame(touches)
    summary = []
    for p in proto.players:
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        pt = vaep_df[vaep_df['Player'] == name] if not vaep_df.empty else pd.DataFrame()
        summary.append({
            'Name': name, 'Team': team,
            'Total_VAEP': round(pt['VAEP'].sum(), 3) if not pt.empty else 0,
            'Avg_VAEP': round(pt['VAEP'].mean(), 4) if not pt.empty else 0,
            'Positive_Actions': int((pt['VAEP'] > 0).sum()) if not pt.empty else 0,
            'Negative_Actions': int((pt['VAEP'] < 0).sum()) if not pt.empty else 0
        })
    return vaep_df, pd.DataFrame(summary)

# --- 9g. MATH: ROTATION ANALYSIS ---
def calculate_rotation_analysis(manager, game_df, proto, sample_step=5):
    """Analyze rotation patterns: 1st/2nd/3rd man roles and double commits.
    Returns (rotation_timeline, rotation_summary, double_commits_df)."""
    blue_players = [p.name for p in proto.players if not p.is_orange]
    orange_players = [p.name for p in proto.players if p.is_orange]
    ball_df = game_df['ball'] if 'ball' in game_df else None
    if ball_df is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    max_frame = game_df.index.max()
    ball_frames = ball_df.index.values
    ball_x_arr = ball_df['pos_x'].values
    ball_y_arr = ball_df['pos_y'].values

    player_data = {}
    for p in proto.players:
        if p.name in game_df:
            pdf = game_df[p.name]
            if 'pos_x' in pdf.columns:
                player_data[p.name] = {
                    'x': pdf['pos_x'].values, 'y': pdf['pos_y'].values,
                    'frames': pdf.index.values,
                    'team': "Orange" if p.is_orange else "Blue"
                }

    timeline = []
    double_commits_raw = []
    role_counts = {p.name: {'1st': 0, '2nd': 0, '3rd': 0, 'total': 0} for p in proto.players}
    rotation_break_counts = {p.name: 0 for p in proto.players}

    for frame in range(0, max_frame, sample_step):
        bi = min(np.searchsorted(ball_frames, frame), len(ball_x_arr) - 1)
        if bi < 0:
            continue
        bx, by = ball_x_arr[bi], ball_y_arr[bi]
        time_s = round(frame / REPLAY_FPS, 1)

        for team_players, team_name in [(blue_players, "Blue"), (orange_players, "Orange")]:
            distances = []
            for name in team_players:
                if name not in player_data:
                    continue
                pd_info = player_data[name]
                pi = min(np.searchsorted(pd_info['frames'], frame), len(pd_info['x']) - 1)
                if pi < 0:
                    continue
                px, py = pd_info['x'][pi], pd_info['y'][pi]
                dist = np.sqrt((px - bx)**2 + (py - by)**2)
                in_attack = py > 0 if team_name == "Blue" else py < 0
                distances.append((name, dist, px, py, in_attack))

            distances.sort(key=lambda x: x[1])
            for rank, (name, dist, px, py, in_attack) in enumerate(distances):
                role = ['1st', '2nd', '3rd'][min(rank, 2)]
                timeline.append({'Frame': frame, 'Time': time_s, 'Player': name, 'Role': role, 'Team': team_name})
                role_counts[name]['total'] += 1
                role_counts[name][role] += 1

            # Double commit: 2+ players within 800 units of ball in attacking half
            attacking_close = [(n, d) for n, d, px, py, ia in distances if d < 800 and ia]
            if len(attacking_close) >= 2:
                double_commits_raw.append({
                    'Frame': frame, 'Time': time_s,
                    'Player1': attacking_close[0][0], 'Player2': attacking_close[1][0],
                    'Team': team_name, 'BallX': bx, 'BallY': by
                })

            # Rotation break: all players in attacking half (nobody back)
            if len(distances) >= 3:
                all_attacking = all(ia for _, _, _, _, ia in distances)
                if all_attacking:
                    for name, _, _, _, _ in distances:
                        rotation_break_counts[name] += 1

    rotation_timeline = pd.DataFrame(timeline)

    # Cluster double commits within 30 frames
    dc_clustered = []
    if double_commits_raw:
        dc_raw_df = pd.DataFrame(double_commits_raw)
        for team in dc_raw_df['Team'].unique():
            team_dc = dc_raw_df[dc_raw_df['Team'] == team].sort_values('Frame')
            prev_frame = -100
            for _, row in team_dc.iterrows():
                if row['Frame'] - prev_frame > 30:
                    dc_clustered.append(row.to_dict())
                prev_frame = row['Frame']
    double_commits_df = pd.DataFrame(dc_clustered) if dc_clustered else pd.DataFrame()

    # Build summary
    summary = []
    for p in proto.players:
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        rc = role_counts.get(name, {'1st': 0, '2nd': 0, '3rd': 0, 'total': 0})
        total = max(rc['total'], 1)
        dc_count = 0
        if not double_commits_df.empty:
            dc_count = len(double_commits_df[(double_commits_df['Player1'] == name) | (double_commits_df['Player2'] == name)])
        # Scale rotation breaks back from sample_step
        rot_breaks = rotation_break_counts.get(name, 0)
        summary.append({
            'Name': name, 'Team': team,
            'Time_1st%': round((rc['1st'] / total) * 100, 1),
            'Time_2nd%': round((rc['2nd'] / total) * 100, 1),
            'Time_3rd%': round((rc['3rd'] / total) * 100, 1),
            'DoubleCommits': dc_count,
            'RotationBreaks': rot_breaks
        })
    rotation_summary = pd.DataFrame(summary)
    return rotation_timeline, rotation_summary, double_commits_df

# --- 10. MATH: AGGREGATE ---
def calculate_final_stats(manager, shot_df, pass_df, aerial_df=None, recovery_df=None, defense_df=None, xga_df=None, vaep_summary=None, rotation_summary=None):
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
                "Aerial Hits": 0, "Aerial %": 0, "Avg Aerial Height": 0, "Time Airborne (s)": 0,
                "Avg Recovery (s)": 0, "Fast Recoveries": 0, "Recovery < 1s %": 0,
                "Shadow %": 0, "Pressure Time (s)": 0,
                "xGA": 0, "Shots Faced": 0, "Goals Conceded (nearest)": 0,
                "Total_VAEP": 0.0, "Avg_VAEP": 0.0, "Positive_Actions": 0, "Negative_Actions": 0,
                "Time_1st%": 0.0, "Time_2nd%": 0.0, "Time_3rd%": 0.0, "DoubleCommits": 0, "RotationBreaks": 0,
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
                    p_data['Time Supersonic'] = round(np.sum(speeds >= 2200) / float(REPLAY_FPS), 2)
                if 'pos_y' in pdf.columns:
                    x_pos = pdf['pos_x'].to_numpy() if 'pos_x' in pdf.columns else np.zeros(len(pdf))
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
            # Merge in pre-computed advanced stats by player name
            for extra_df, cols in [
                (aerial_df, ['Aerial Hits', 'Aerial %', 'Avg Aerial Height', 'Time Airborne (s)']),
                (recovery_df, ['Avg Recovery (s)', 'Fast Recoveries', 'Recovery < 1s %']),
                (defense_df, ['Shadow %', 'Pressure Time (s)']),
                (xga_df, ['xGA', 'Shots Faced', 'Goals Conceded (nearest)']),
                (vaep_summary, ['Total_VAEP', 'Avg_VAEP', 'Positive_Actions', 'Negative_Actions']),
                (rotation_summary, ['Time_1st%', 'Time_2nd%', 'Time_3rd%', 'DoubleCommits', 'RotationBreaks']),
            ]:
                if extra_df is not None and not extra_df.empty:
                    row = extra_df[extra_df['Name'] == name]
                    if not row.empty:
                        for c in cols:
                            if c in row.columns:
                                p_data[c] = row.iloc[0][c]
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
                    sc1, sc2, sc3, sc4 = st.columns(4)
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
                        st.write(f"🛡️ xGA: {p.get('xGA', 0)}")
                        st.write(f"👤 Shadow: {p.get('Shadow %', 0)}%")
                    with sc4:
                        st.caption("Mechanics")
                        st.write(f"✈️ Aerials: {p.get('Aerial Hits', 0)}")
                        st.write(f"⬆️ Airborne: {p.get('Time Airborne (s)', 0)}s")
                        st.write(f"⚡ Recovery: {p.get('Avg Recovery (s)', 0)}s")

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
                aerial_df = calculate_aerial_stats(manager, temp_map)
                recovery_df = calculate_recovery_time(manager, temp_map)
                defense_df = calculate_defensive_pressure(manager, game_df, proto)
                xga_df = calculate_xg_against(manager, temp_map, shot_df)
                vaep_df, vaep_summary = calculate_vaep(manager, temp_map, shot_df)
                rotation_timeline, rotation_summary, double_commits_df = calculate_rotation_analysis(manager, game_df, proto)
                wp_model, wp_scaler = load_win_prob_model()
                wp_result = calculate_win_probability_trained(manager, wp_model, wp_scaler)
                if isinstance(wp_result, tuple):
                    win_prob_df, wp_model_used = wp_result
                else:
                    win_prob_df, wp_model_used = wp_result, False
                df = calculate_final_stats(manager, shot_df, pass_df, aerial_df, recovery_df, defense_df, xga_df, vaep_summary, rotation_summary)
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
            
            t1, t2, t3, t3b, t4, t5, t6, t8, t9, t7 = st.tabs(["🚀 Kickoffs", "🌊 Match Narrative", "🎯 Shot Map", "🎬 Shot Viewer", "🕸️ Pass Map", "🔥 Heatmaps", "⚡ Speed", "🛡️ Advanced", "🔄 Rotation", "📸 Export"])
            
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
                    else: st.info("No kickoffs found for selected players.")
                else: st.info("No kickoff data found.")

            with t2:
                st.subheader("Match Narrative")
                # --- A. WIN PROBABILITY CHART ---
                try:
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
                        st.caption("Model: " + ("Trained (logistic regression on career data)" if wp_model_used else "Hand-tuned heuristic") + ". Process 15+ replays in Season mode to train a data-driven model.")
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
                        valid_pos = p_frames[p_frames['pos_z'] > 0].dropna(subset=['pos_x', 'pos_y'])
                        sampled = valid_pos.iloc[::3]
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
                            x=sampled['pos_x'], y=sampled['pos_y'],
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

            with t8:
                st.subheader("Advanced Analytics")

                # --- SECTION 1: Aerial Stats ---
                st.markdown("#### Aerial Stats")
                if not aerial_df.empty:
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        fig_aer = px.bar(aerial_df, x='Name', y='Aerial Hits', color='Team',
                            title="Aerial Hits", color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_aer.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_aer, use_container_width=True)
                    with ac2:
                        fig_air = go.Figure()
                        for _, row in aerial_df.iterrows():
                            color = '#007bff' if row['Team'] == 'Blue' else '#ff9900'
                            fig_air.add_trace(go.Bar(x=[row['Name']], y=[row['Time Airborne (s)']],
                                name=row['Name'], marker_color=color, showlegend=False))
                        fig_air.update_layout(title="Time Airborne (s)",
                            plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_air, use_container_width=True)
                    aer_cols = ['Name', 'Team', 'Aerial Hits', 'Aerial %', 'Avg Aerial Height', 'Max Aerial Height', 'Time Airborne (s)']
                    st.dataframe(aerial_df[aer_cols].sort_values('Aerial Hits', ascending=False), use_container_width=True, hide_index=True)
                st.divider()

                # --- SECTION 2: Recovery Time ---
                st.markdown("#### Recovery Time")
                if not recovery_df.empty:
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        fig_rec = px.bar(recovery_df, x='Name', y='Avg Recovery (s)', color='Team',
                            title="Avg Time to Supersonic After Hit",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_rec.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_rec, use_container_width=True)
                    with rc2:
                        fig_fast = px.bar(recovery_df, x='Name', y='Recovery < 1s %', color='Team',
                            title="Fast Recovery Rate (< 1s)",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_fast.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_fast, use_container_width=True)
                    rec_cols = ['Name', 'Team', 'Avg Recovery (s)', 'Fast Recoveries', 'Total Hits', 'Recovery < 1s %']
                    st.dataframe(recovery_df[rec_cols].sort_values('Avg Recovery (s)'), use_container_width=True, hide_index=True)
                st.divider()

                # --- SECTION 3: xA Sankey Flow ---
                st.markdown("#### Pass Chain Flow (xA Sankey)")
                if not pass_df.empty:
                    # Build Sankey from pass chains: Sender -> Receiver, weighted by count or xA
                    chain_df = pass_df.groupby(['Sender', 'Receiver', 'Team']).agg(
                        count=('xA', 'size'), total_xA=('xA', 'sum'),
                        key_passes=('KeyPass', 'sum')
                    ).reset_index()
                    chain_df = chain_df[chain_df['count'] >= 2]  # filter noise
                    if not chain_df.empty:
                        all_names = sorted(set(chain_df['Sender'].tolist() + chain_df['Receiver'].tolist()))
                        name_to_idx = {n: i for i, n in enumerate(all_names)}
                        node_colors = []
                        for n in all_names:
                            team = chain_df[chain_df['Sender'] == n]['Team'].values
                            if len(team) == 0:
                                team = chain_df[chain_df['Receiver'] == n]['Team'].values
                            node_colors.append('#007bff' if (len(team) > 0 and team[0] == 'Blue') else '#ff9900')
                        link_colors = []
                        for _, row in chain_df.iterrows():
                            if row['Team'] == 'Blue':
                                link_colors.append('rgba(0,123,255,0.35)')
                            else:
                                link_colors.append('rgba(255,153,0,0.35)')
                        fig_sankey = go.Figure(go.Sankey(
                            node=dict(pad=15, thickness=20, line=dict(color='white', width=0.5),
                                      label=all_names, color=node_colors),
                            link=dict(
                                source=[name_to_idx[r['Sender']] for _, r in chain_df.iterrows()],
                                target=[name_to_idx[r['Receiver']] for _, r in chain_df.iterrows()],
                                value=chain_df['count'].tolist(),
                                color=link_colors,
                                customdata=np.stack([chain_df['total_xA'].round(2), chain_df['key_passes'].astype(int)], axis=-1),
                                hovertemplate='%{source.label} → %{target.label}<br>Passes: %{value}<br>xA: %{customdata[0]}<br>Key Passes: %{customdata[1]}<extra></extra>',
                            )
                        ))
                        fig_sankey.update_layout(title="Pass Flow Network",
                            plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=12), height=400)
                        st.plotly_chart(fig_sankey, use_container_width=True)
                    else:
                        st.info("Not enough passing connections to build Sankey diagram.")
                else:
                    st.info("No pass data available.")
                st.divider()

                # --- SECTION 4: Defensive Pressure / Shadow Defense ---
                st.markdown("#### Defensive Pressure (Shadow Defense)")
                if not defense_df.empty:
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        fig_shadow = px.bar(defense_df, x='Name', y='Shadow %', color='Team',
                            title="Shadow Defense Time %",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_shadow.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_shadow, use_container_width=True)
                    with dc2:
                        fig_pres = px.bar(defense_df, x='Name', y='Pressure Time (s)', color='Team',
                            title="Total Pressure Time (s)",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_pres.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_pres, use_container_width=True)
                    st.dataframe(defense_df[['Name', 'Team', 'Shadow %', 'Pressure Time (s)']].sort_values('Shadow %', ascending=False),
                        use_container_width=True, hide_index=True)
                    st.caption("Shadow defense: time spent between ball and own goal while retreating in defensive half.")
                st.divider()

                # --- SECTION 5: Shot Quality Conceded (xG-Against) ---
                st.markdown("#### Shot Quality Conceded (xG-Against)")
                if not xga_df.empty:
                    xc1, xc2 = st.columns(2)
                    with xc1:
                        fig_xga = px.bar(xga_df, x='Name', y='xGA', color='Team',
                            title="Expected Goals Against (as nearest defender)",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_xga.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_xga, use_container_width=True)
                    with xc2:
                        fig_dist = px.bar(xga_df, x='Name', y='Avg Dist to Shot', color='Team',
                            title="Avg Distance to Shot When Nearest Defender",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_dist.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_dist, use_container_width=True)
                    xga_cols = ['Name', 'Team', 'Shots Faced', 'xGA', 'Goals Conceded (nearest)', 'Avg Dist to Shot', 'High xG Faced']
                    st.dataframe(xga_df[xga_cols].sort_values('xGA', ascending=False), use_container_width=True, hide_index=True)
                    st.caption("xG-Against: cumulative xG of shots where this player was the nearest defender.")
                st.divider()

                # --- SECTION 6: Action Value (VAEP) ---
                st.markdown("#### Action Value (VAEP)")
                if not vaep_summary.empty:
                    vc1, vc2 = st.columns(2)
                    with vc1:
                        fig_vaep_bar = px.bar(vaep_summary.sort_values('Total_VAEP', ascending=False),
                            x='Name', y='Total_VAEP', color='Team',
                            title="Total VAEP per Player",
                            color_discrete_map={"Blue": "#007bff", "Orange": "#ff9900"})
                        fig_vaep_bar.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_vaep_bar, use_container_width=True)
                    with vc2:
                        if not vaep_df.empty:
                            fig_vaep_scatter = go.Figure()
                            for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
                                t_data = vaep_df[vaep_df['Team'] == team]
                                if not t_data.empty:
                                    colors_arr = ['#00cc96' if v > 0 else '#EF553B' for v in t_data['VAEP']]
                                    fig_vaep_scatter.add_trace(go.Scatter(
                                        x=t_data['Time'], y=t_data['VAEP'], mode='markers',
                                        marker=dict(size=5, color=colors_arr, opacity=0.6),
                                        name=team, text=t_data['Player'],
                                        hovertemplate="<b>%{text}</b><br>Time: %{x}s<br>VAEP: %{y:.3f}<extra></extra>"))
                            fig_vaep_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
                            fig_vaep_scatter.update_layout(title="Touch VAEP Timeline (green=positive, red=negative)",
                                xaxis_title="Time (s)", yaxis_title="VAEP",
                                plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=350)
                            st.plotly_chart(fig_vaep_scatter, use_container_width=True)
                    vaep_show_cols = ['Name', 'Team', 'Total_VAEP', 'Avg_VAEP', 'Positive_Actions', 'Negative_Actions']
                    st.dataframe(vaep_summary[vaep_show_cols].sort_values('Total_VAEP', ascending=False),
                        use_container_width=True, hide_index=True)
                    st.caption("VAEP: each touch scored by change in scoring threat. Positive = moved team closer to scoring.")
                else:
                    st.info("No VAEP data available.")

            with t9:
                st.subheader("Rotation Analysis")
                if not rotation_summary.empty:
                    # Role time distribution stacked bar
                    st.markdown("#### Role Distribution")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        fig_roles = go.Figure()
                        for role, color in [('1st', '#EF553B'), ('2nd', '#FFA15A'), ('3rd', '#00CC96')]:
                            col_name = f'Time_{role}%'
                            fig_roles.add_trace(go.Bar(
                                x=rotation_summary['Name'], y=rotation_summary[col_name],
                                name=f'{role} Man', marker_color=color))
                        fig_roles.update_layout(barmode='stack', title="Time Spent as 1st/2nd/3rd Man",
                            yaxis_title="% of Match", plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'), legend=dict(orientation='h', y=1.12))
                        st.plotly_chart(fig_roles, use_container_width=True)
                    with rc2:
                        # Team comparison
                        for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
                            team_data = rotation_summary[rotation_summary['Team'] == team]
                            if not team_data.empty:
                                st.markdown(f"**{team} Team**")
                                st.dataframe(team_data[['Name', 'Time_1st%', 'Time_2nd%', 'Time_3rd%', 'DoubleCommits']].reset_index(drop=True),
                                    use_container_width=True, hide_index=True)

                    st.divider()

                    # Role timeline heatmap
                    if not rotation_timeline.empty:
                        st.markdown("#### Role Timeline")
                        for team in ["Blue", "Orange"]:
                            team_tl = rotation_timeline[rotation_timeline['Team'] == team]
                            if team_tl.empty:
                                continue
                            role_map = {'1st': 1, '2nd': 2, '3rd': 3}
                            team_tl = team_tl.copy()
                            team_tl['RoleNum'] = team_tl['Role'].map(role_map)
                            players = sorted(team_tl['Player'].unique())
                            # Sample to avoid too many points
                            sampled = team_tl.iloc[::3] if len(team_tl) > 5000 else team_tl
                            fig_tl = go.Figure()
                            fig_tl.add_trace(go.Heatmap(
                                x=sampled['Time'], y=sampled['Player'], z=sampled['RoleNum'],
                                colorscale=[[0, '#EF553B'], [0.5, '#FFA15A'], [1.0, '#00CC96']],
                                zmin=1, zmax=3, showscale=True,
                                colorbar=dict(title="Role", tickvals=[1, 2, 3], ticktext=['1st', '2nd', '3rd'])))
                            fig_tl.update_layout(title=f"{team} Rotation Timeline",
                                xaxis_title="Time (s)", yaxis_title="Player",
                                plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'), height=250)
                            st.plotly_chart(fig_tl, use_container_width=True)

                    st.divider()

                    # Double commit map
                    if not double_commits_df.empty:
                        st.markdown(f"#### Double Commits ({len(double_commits_df)} detected)")
                        fig_dc = go.Figure()
                        fig_dc.update_layout(get_field_layout("Double Commit Locations"))
                        for team, color in [("Blue", "#007bff"), ("Orange", "#ff9900")]:
                            t_dc = double_commits_df[double_commits_df['Team'] == team] if 'Team' in double_commits_df.columns else pd.DataFrame()
                            if not t_dc.empty:
                                fig_dc.add_trace(go.Scatter(
                                    x=t_dc['BallX'], y=t_dc['BallY'], mode='markers',
                                    marker=dict(size=14, color=color, symbol='x', line=dict(width=2, color='white')),
                                    name=f'{team} ({len(t_dc)})',
                                    text=[f"{r['Player1']} + {r['Player2']} @ {r['Time']}s" for _, r in t_dc.iterrows()],
                                    hovertemplate="<b>%{text}</b><extra></extra>"))
                        st.plotly_chart(fig_dc, use_container_width=True)
                    else:
                        st.success("No double commits detected!")

                    st.caption("1st man = closest to ball, 2nd = support, 3rd = last back. Double commits = 2 players within 800u of ball in attacking half.")
                else:
                    st.info("No rotation data available.")

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
        wp_training_list = []
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
                    aerial_df_b = calculate_aerial_stats(manager, temp_map)
                    recovery_df_b = calculate_recovery_time(manager, temp_map)
                    defense_df_b = calculate_defensive_pressure(manager, game_df, proto)
                    xga_df_b = calculate_xg_against(manager, temp_map, shot_df)
                    vaep_df_b, vaep_summary_b = calculate_vaep(manager, temp_map, shot_df)
                    _, rotation_summary_b, _ = calculate_rotation_analysis(manager, game_df, proto)
                    stats = calculate_final_stats(manager, shot_df, pass_df, aerial_df_b, recovery_df_b, defense_df_b, xga_df_b, vaep_summary_b, rotation_summary_b)
                    # Collect win prob training data
                    wp_train = extract_win_prob_training_data(manager, game_df, proto)
                    if not wp_train.empty:
                        wp_training_list.append(wp_train)
                    kickoff_df = calculate_kickoff_stats(manager, temp_map, game_id)
                    if not stats.empty and 'IsBot' in stats.columns and filter_ghosts: stats = stats[~stats['IsBot']]
                    if not stats.empty:
                        stats['MatchID'] = str(game_id)
                        blue_g = stats[stats['Team']=='Blue']['Goals'].sum()
                        orange_g = stats[stats['Team']=='Orange']['Goals'].sum()

                        if blue_g > orange_g:
                            stats['Won'] = stats['Team'] == "Blue"
                        elif orange_g > blue_g:
                            stats['Won'] = stats['Team'] == "Orange"
                        else:
                            stats['Won'] = False
                        # Overtime detection
                        is_ot = detect_overtime(manager)
                        stats['Overtime'] = is_ot
                        # Luck calculation
                        for team in ["Blue", "Orange"]:
                            team_goals = int(stats[stats['Team']==team]['Goals'].sum())
                            luck_val = calculate_luck_percentage(shot_df, team, team_goals)
                            stats.loc[stats['Team']==team, 'Luck'] = luck_val
                        # Timestamp
                        ts = get_match_timestamp(proto)
                        stats['Timestamp'] = ts
 
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
            # Train win probability model if enough data
            if wp_training_list:
                all_wp_data = pd.concat(wp_training_list, ignore_index=True)
                wp_model, wp_scaler = train_win_probability_model(all_wp_data)
                if wp_model is not None:
                    save_win_prob_model(wp_model, wp_scaler)
                    st.info(f"Win probability model trained on {len(all_wp_data)} samples from {len(wp_training_list)} matches and saved.")
            st.success(f"Added {len(new_stats_df['MatchID'].unique())} new matches to database!")
            existing_stats, existing_kickoffs = load_data()
        else:
            if uploaded_files: st.info("No new matches found (duplicates skipped).")

    if not existing_stats.empty:
        season = existing_stats
        season_kickoffs = existing_kickoffs
        # Ensure backward compatibility for new columns
        for col, default in [('Overtime', False), ('Luck', 0.0), ('Timestamp', ''), ('Wall_Time', 0.0), ('Corner_Time', 0.0), ('On_Wall_Time', 0.0), ('Carry_Time', 0.0),
                              ('Aerial Hits', 0), ('Aerial %', 0.0), ('Avg Aerial Height', 0), ('Time Airborne (s)', 0.0),
                              ('Avg Recovery (s)', 0.0), ('Fast Recoveries', 0), ('Recovery < 1s %', 0.0),
                              ('Shadow %', 0.0), ('Pressure Time (s)', 0.0),
                              ('xGA', 0.0), ('Shots Faced', 0), ('Goals Conceded (nearest)', 0),
                              ('Total_VAEP', 0.0), ('Avg_VAEP', 0.0), ('Positive_Actions', 0), ('Negative_Actions', 0),
                              ('Time_1st%', 0.0), ('Time_2nd%', 0.0), ('Time_3rd%', 0.0), ('DoubleCommits', 0), ('RotationBreaks', 0)]:
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
            metric = st.selectbox("Metric:", ['Rating', 'Score', 'Goals', 'Assists', 'Saves', 'xG', 'Avg Speed', 'Luck', 'Carry_Time',
                'Aerial Hits', 'Aerial %', 'Time Airborne (s)', 'Avg Recovery (s)', 'Recovery < 1s %', 'Shadow %', 'xGA',
                'Total_VAEP', 'Avg_VAEP', 'Time_1st%', 'Time_2nd%', 'Time_3rd%', 'DoubleCommits'])
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
 
