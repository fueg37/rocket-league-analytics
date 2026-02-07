import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import logging
import numpy as np

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
    """Returns a Plotly layout dict that enforces correct RL Field Aspect Ratio."""
    return dict(
        title=title,
        xaxis=dict(range=[-4200, 4200], visible=False, fixedrange=True),
        yaxis=dict(range=[-5200, 5200], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        plot_bgcolor='#1a241a', # Dark grass green
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        shapes=[
            dict(type="rect", x0=-4096, y0=-5120, x1=4096, y1=5120, line=dict(color="rgba(255,255,255,0.8)", width=2)),
            dict(type="rect", x0=-893, y0=5120, x1=893, y1=6000, line=dict(color="#ff9900", width=2)),
            dict(type="rect", x0=-893, y0=-6000, x1=893, y1=-5120, line=dict(color="#007bff", width=2)),
            dict(type="line", x0=-4096, y0=0, x1=4096, y1=0, line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dot")),
            dict(type="circle", x0=-1000, y0=-1000, x1=1000, y1=1000, line=dict(color="rgba(255,255,255,0.5)", width=2))
        ]
    )

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
    """Calculates win probability for Blue Team over time."""
    proto = manager.get_protobuf_data()
    game_df = manager.get_data_frame()
    max_frame = game_df.index.max()
    frames = np.arange(0, max_frame, REPLAY_FPS)
    seconds = frames / float(REPLAY_FPS)
    
    blue_goals = []
    orange_goals = []
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            frame = getattr(g, 'frame_number', getattr(g, 'frame', 0))
            team = "Orange" if getattr(g.player_id, 'is_orange', False) else "Blue"
            frame = min(frame, max_frame)
            if team == "Blue": blue_goals.append(frame)
            else: orange_goals.append(frame)
            
    blue_goals.sort()
    orange_goals.sort()
    probs = []
    
    for f, t in zip(frames, seconds):
        b_score = sum(1 for gf in blue_goals if gf <= f)
        o_score = sum(1 for gf in orange_goals if gf <= f)
        diff = b_score - o_score
        time_remaining = max(300 - t, 0.1)
        
        if t >= 300 and diff == 0:
            p = 0.5
        else:
            time_in_minutes = time_remaining / 60.0
            x = diff * (2.0 / max(time_in_minutes, 0.1))
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
    
    metadata_goal_frames = set()
    if hasattr(proto, 'game_metadata') and hasattr(proto.game_metadata, 'goals'):
        for g in proto.game_metadata.goals:
            f = getattr(g, 'frame_number', getattr(g, 'frame', None))
            if f: 
                for i in range(f - 100, f + 20): metadata_goal_frames.add(i)

    for hit in hits:
        frame = hit.frame_number
        if not hit.player_id: continue
        is_lib_shot = getattr(hit, 'is_shot', False)
        is_lib_goal = getattr(hit, 'is_goal', False)
        is_meta_goal = frame in metadata_goal_frames
        is_physics_shot = False
        ball_pos, ball_vel = None, None
        
        if 'ball' in game_df and frame in game_df.index:
            try:
                ball_data = game_df['ball'].loc[frame]
                ball_pos = (ball_data['pos_x'], ball_data['pos_y'], ball_data['pos_z'])
                ball_vel = (ball_data['vel_x'], ball_data['vel_y'], ball_data['vel_z'])
                pid = str(hit.player_id.id)
                shooter_team = "Unknown"
                for p in proto.players:
                    if str(p.id.id) == pid:
                        shooter_team = "Orange" if p.is_orange else "Blue"
                        break
                direction_sign = 1 if shooter_team == "Blue" else -1
                if (ball_vel[1] * direction_sign > 0) and (abs(ball_vel[1]) > 800): is_physics_shot = True
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
                result = "Goal" if (is_lib_goal or is_meta_goal) else "Shot"
                shot_list.append({
                    "Player": player_name, "Team": shooter_team, "Frame": frame,
                    "xG": round(xg, 2), "Result": result, "BigChance": is_big_chance,
                    "X": ball_pos[0], "Y": ball_pos[1],
                    "Speed": int(np.sqrt(ball_vel[0]**2 + ball_vel[1]**2 + ball_vel[2]**2))
                })

    if shot_list:
        raw_df = pd.DataFrame(shot_list)
        raw_df['TimeGroup'] = (raw_df['Frame'] // 15)
        final_df = raw_df.sort_values('xG', ascending=False).drop_duplicates(subset=['Player', 'TimeGroup', 'Result'])
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
                "Boost Used": 0, "Wasted Boost": 0, "Avg Speed": 0, "Time Supersonic": 0, "Pos_Def": 0, "Pos_Mid": 0, "Pos_Off": 0
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
                    y_pos = pdf['pos_y'].to_numpy()
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
            stats.append(p_data)
    return pd.DataFrame(stats)

# --- 11. UI COMPONENTS ---
def render_scoreboard(df):
    st.markdown("### 🏆 Final Scoreboard")
    blue_goals = df[df['Team']=='Blue']['Goals'].sum()
    orange_goals = df[df['Team']=='Orange']['Goals'].sum()
    c1, c2, c3 = st.columns([1, 0.5, 1])
    with c1: st.markdown(f"<h1 style='text-align: center; color: #007bff; margin: 0;'>{blue_goals}</h1>", unsafe_allow_html=True)
    with c2: st.markdown(f"<h1 style='text-align: center; color: white; margin: 0;'>-</h1>", unsafe_allow_html=True)
    with c3: st.markdown(f"<h1 style='text-align: center; color: #ff9900; margin: 0;'>{orange_goals}</h1>", unsafe_allow_html=True)
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

# ==========================================
# MODE 1: SINGLE MATCH
# ==========================================
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
                if not df.empty and 'IsBot' in df.columns and filter_ghosts: 
                    df = df[~df['IsBot']]
            
            all_players = sorted(list(temp_map.values()))
            default_focus = [p for p in ["Fueg", "Zelli197"] if p in all_players]
            focus_players = st.sidebar.multiselect("🎯 Focus Analysis On:", all_players, default=default_focus)
            
            render_scoreboard(df)
            render_dashboard(df, shot_df, pass_df)
            
            t1, t2, t3, t4, t5, t6 = st.tabs(["🚀 Kickoffs", "🌊 Match Narrative", "🎯 Shot Map", "🕸️ Pass Map", "🔥 Heatmaps", "⚡ Speed"])
            
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
                            fig = px.density_heatmap(disp_kickoff, x='End_X', y='End_Y', nbinsx=60, nbinsy=80, title="Kickoff Outcome Heatmap")
                            fig.update_layout(get_field_layout())
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
                        fig_prob.update_layout(title="🏆 Win Probability", yaxis=dict(title="Blue Win %", range=[0, 100], showgrid=False), xaxis=dict(title="Time (Seconds)", showgrid=False), plot_bgcolor='#1e1e1e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_prob, use_container_width=True)
                except Exception as e: st.error(f"Could not calculate Win Probability: {e}")
                st.divider()

                # --- B. MOMENTUM CHART ---
                st.markdown("#### 🌊 Pressure Index")
                if not momentum_series.empty:
                    fig = go.Figure()
                    x_time = momentum_series.index
                    y_values = momentum_series.values
                    fig.add_trace(go.Scatter(x=x_time, y=y_values.clip(min=0), fill='tozeroy', mode='none', name='Blue Pressure', fillcolor='rgba(0, 123, 255, 0.6)'))
                    fig.add_trace(go.Scatter(x=x_time, y=y_values.clip(max=0), fill='tozeroy', mode='none', name='Orange Pressure', fillcolor='rgba(255, 153, 0, 0.6)'))
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
                    goals = shot_df[shot_df['Result'] == 'Goal']
                    misses = shot_df[shot_df['Result'] == 'Shot']
                    big_chances = shot_df[shot_df['BigChance'] == True]
                    fig.add_trace(go.Scatter(x=misses['X'], y=misses['Y'], mode='markers', marker=dict(size=10, color='gray', opacity=0.5), name='Shot', text=misses['Player'], customdata=misses['xG'], hovertemplate="xG: %{customdata:.2f}"))
                    fig.add_trace(go.Scatter(x=goals['X'], y=goals['Y'], mode='markers', marker=dict(size=15, color='#00cc96', line=dict(width=2, color='white')), name='Goal', text=goals['Player'], customdata=goals['xG'], hovertemplate="xG: %{customdata:.2f}"))
                    fig.add_trace(go.Scatter(x=big_chances['X'], y=big_chances['Y'], mode='markers', marker=dict(size=25, color='rgba(0,0,0,0)', line=dict(width=2, color='yellow')), name='Big Chance', hoverinfo='skip'))
                    st.plotly_chart(fig, use_container_width=True)

            with t4:
                if not pass_df.empty:
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.write("#### Playmaker Leaderboard")
                        st.dataframe(pass_df.groupby('Sender')['xA'].sum().sort_values(ascending=False), use_container_width=True)
                    with col_b:
                        fig = go.Figure()
                        fig.update_layout(get_field_layout("Pass Map"))
                        reg = pass_df[pass_df['KeyPass']==False]
                        for _, row in reg.iterrows():
                            fig.add_trace(go.Scatter(x=[row['x1'], row['x2']], y=[row['y1'], row['y2']], mode='lines', line=dict(color='rgba(100,100,100,0.3)', width=1), showlegend=False))
                        key = pass_df[pass_df['KeyPass']==True]
                        key_legend_shown = False
                        for _, row in key.iterrows():
                            fig.add_trace(go.Scatter(x=[row['x1'], row['x2']], y=[row['y1'], row['y2']], mode='lines', line=dict(color='gold', width=3), name='Key Pass' if not key_legend_shown else None, showlegend=not key_legend_shown))
                            key_legend_shown = True
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
                        soccer_scale = ['#00008b', '#00ffff', '#00ff00', '#ffff00', '#ff0000']
                        fig = px.density_heatmap(valid_pos, x='pos_x', y='pos_y', nbinsx=120, nbinsy=160, color_continuous_scale=soccer_scale)
                        fig.update_traces(opacity=0.7)
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

# ==========================================
# MODE 2: SEASON BATCH
# ==========================================
elif app_mode == "📈 Season Batch Processor":
    existing_stats, existing_kickoffs = load_data()
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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games", len(hero_df))
        c2.metric("Win Rate", f"{(hero_df['Won'].sum()/len(hero_df)*100):.1f}%")
        if 'Rating' in hero_df: c3.metric("Avg Rating", f"{hero_df['Rating'].mean():.2f}")
        if 'xG' in hero_df: c4.metric("Total xG", f"{hero_df['xG'].sum():.2f}")
        
        t1, t2, t3, t4, t5 = st.tabs(["📈 Performance", "🚀 Season Kickoffs", "🧠 Playstyle", "🕸️ Radar", "📊 Log"])
        with t1:
            st.subheader("Performance Trends")
            metric = st.selectbox("Metric:", ['Rating', 'Score', 'Goals', 'Assists', 'Saves', 'xG', 'Avg Speed'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hero_df['GameNum'], y=hero_df[metric], name=hero, line=dict(color='#007bff', width=3)))
            if teammate != "None":
                mate_df = season[season['Name'] == teammate].reset_index(drop=True)
                mate_df['GameNum'] = mate_df.index + 1 
                fig.add_trace(go.Scatter(x=mate_df['GameNum'], y=mate_df[metric], name=teammate, line=dict(color='#ff9900', width=3, dash='dot')))
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
                st.write("#### Season Outcome Heatmap")
                fig = px.density_heatmap(hero_k, x='End_X', y='End_Y', nbinsx=60, nbinsy=80, title=f"Where {hero}'s Kickoffs Go (Season View)")
                fig.update_layout(get_field_layout())
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No kickoff data collected.")
        with t3:
            st.subheader("Positional Tendencies (Avg Time Spent)")
            if 'Pos_Def' in hero_df:
                avg_def = hero_df['Pos_Def'].mean()
                avg_mid = hero_df['Pos_Mid'].mean()
                avg_off = hero_df['Pos_Off'].mean()
                fig = px.pie(names=['Defense', 'Midfield', 'Offense'], values=[avg_def, avg_mid, avg_off], title=f"{hero} Average Positioning", color_discrete_sequence=['#EF553B', '#FFA15A', '#00CC96'])
                st.plotly_chart(fig, use_container_width=True)
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
            st.dataframe(hero_df.style.map(lambda x: 'color: green' if x else 'color: red', subset=['Won']), use_container_width=True)
