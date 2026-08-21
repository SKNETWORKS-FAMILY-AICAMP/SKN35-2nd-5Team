#!/usr/bin/env python3
"""
Detailed EDA script for churn_modeling_features.csv
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("data/churn_modeling_features.csv")
    
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total Users: {len(df):,}")
    print(f"Total Columns: {df.shape[1]}")
    print(f"Churn Users (is_churn=1): {int(df['is_churn'].sum()):,} ({df['is_churn'].mean()*100:.2f}%)")
    print(f"Retained Users (is_churn=0): {int((df['is_churn']==0).sum()):,} ({(1-df['is_churn'].mean())*100:.2f}%)")
    print(f"  - Refund Churn: {int(df['is_refund_churn'].sum()):,} ({df['is_refund_churn'].mean()*100:.2f}%)")
    print(f"  - Non-renewal Churn: {int(df['is_non_renewal_churn'].sum()):,} ({df['is_non_renewal_churn'].mean()*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("2. BEHAVIOR COMPARISON: RETAINED (0) vs CHURNED (1)")
    print("=" * 60)
    cols = [
        "pre_pay_events", "pre_pay_active_days",
        "obs_total_events", "obs_active_days", "obs_events_per_active_day",
        "obs_w1_events", "obs_w2_events", "obs_decay_ratio", "obs_activity_change_rate",
        "obs_recency_days", "obs_last_active_day",
        "obs_respond_count", "obs_submit_count", "obs_play_video_count", "obs_play_audio_count",
        "total_lifetime_events"
    ]
    
    stats = df.groupby("is_churn")[cols].agg(["mean", "median"])
    print(f"{'Feature':<28} | {'Retained Mean (Med)':<20} | {'Churned Mean (Med)':<20} | {'Diff (%)':<8}")
    print("-" * 80)
    for c in cols:
        mean_0 = stats.loc[0, (c, "mean")]
        mean_1 = stats.loc[1, (c, "mean")]
        med_0 = stats.loc[0, (c, "median")]
        med_1 = stats.loc[1, (c, "median")]
        diff_pct = ((mean_1 - mean_0) / (mean_0 + 1e-5)) * 100
        ret_str = f"{mean_0:7.1f} ({med_0:5.1f})"
        chn_str = f"{mean_1:7.1f} ({med_1:5.1f})"
        print(f"{c:<28} | {ret_str:<20} | {chn_str:<20} | {diff_pct:+7.1f}%")
        
    print("\n" + "=" * 60)
    print("3. CORRELATION WITH CHURN (is_churn)")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corrs = df[numeric_cols].corr()["is_churn"].sort_values()
    for feat, corr_val in corrs.items():
        if feat not in ["is_churn", "is_non_renewal_churn", "is_refund_churn"]:
            print(f"{feat:<32} : {corr_val:+7.4f}")
            
    print("\n" + "=" * 60)
    print("4. TOP PASS ITEMS CHURN RATE")
    print("=" * 60)
    item_stats = df.groupby("first_pay_item").agg(
        user_count=("user_id", "count"),
        churn_rate=("is_churn", "mean"),
        refund_rate=("is_refund_churn", "mean")
    ).sort_values("user_count", ascending=False).head(10)
    print(item_stats.to_string())

if __name__ == "__main__":
    main()
