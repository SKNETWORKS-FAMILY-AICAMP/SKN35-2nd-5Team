"""
Feature Engineering module to transform raw logs into observation-window tabular features.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def extract_user_observation_features(
    user_rows: List[Dict[str, Any]],
    first_pay_ts: float,
    obs_window_days: int = 14,
) -> Dict[str, Any]:
    """
    Extract engagement, performance, and decay features from a single user's log.
    """
    ms_per_day = 86_400_000.0
    obs_end_ts = first_pay_ts + (obs_window_days * ms_per_day)
    w1_end_ts = first_pay_ts + (7 * ms_per_day)
    
    total_events = 0
    pre_pay_events = 0
    obs_events = 0
    obs_w1_events = 0
    obs_w2_events = 0
    obs_dates = set()
    obs_solve_count = 0
    obs_submit_count = 0
    obs_video_count = 0
    obs_audio_count = 0
    obs_quit_count = 0
    obs_last_event_ts = first_pay_ts
    
    for r in user_rows:
        total_events += 1
        ts = float(r["timestamp"])
        action = r.get("action_type", "")
        
        if ts < first_pay_ts:
            pre_pay_events += 1
        elif first_pay_ts <= ts <= obs_end_ts:
            obs_events += 1
            obs_dates.add(int(ts // ms_per_day))
            obs_last_event_ts = max(obs_last_event_ts, ts)
            
            if ts <= w1_end_ts:
                obs_w1_events += 1
            else:
                obs_w2_events += 1
                
            if action == "respond":
                obs_solve_count += 1
            elif action == "submit":
                obs_submit_count += 1
            elif action == "play_video":
                obs_video_count += 1
            elif action == "play_audio":
                obs_audio_count += 1
            elif action == "quit":
                obs_quit_count += 1
                
    obs_active_days = len(obs_dates)
    decay_ratio = obs_w2_events / (obs_w1_events + 1.0)
    days_since_first_pay_last_act = (obs_last_event_ts - first_pay_ts) / ms_per_day
    recency_days = max(0.0, obs_window_days - days_since_first_pay_last_act)
    
    return {
        "pre_pay_events": pre_pay_events,
        "obs_total_events": obs_events,
        "obs_w1_events": obs_w1_events,
        "obs_w2_events": obs_w2_events,
        "obs_decay_ratio": round(decay_ratio, 4),
        "obs_active_days": obs_active_days,
        "obs_solve_count": obs_solve_count,
        "obs_submit_count": obs_submit_count,
        "obs_play_video_count": obs_video_count,
        "obs_play_audio_count": obs_audio_count,
        "obs_quit_count": obs_quit_count,
        "obs_last_active_day": round(days_since_first_pay_last_act, 2),
        "obs_recency_days": round(recency_days, 2),
        "total_lifetime_events": total_events,
    }
