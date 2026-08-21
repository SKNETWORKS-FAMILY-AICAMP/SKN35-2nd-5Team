#!/usr/bin/env python3
"""
Extract and analyze payment, refund, repurchase, and churn metadata from EdNet KT4 dataset.
"""

import os
import sys
import time
import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KT4_DIR = PROJECT_ROOT / "KT4"
DATA_DIR = PROJECT_ROOT / "data"

def process_batch(file_paths):
    records = []
    user_stats = {} # uid -> {first_ts, last_ts, total_events, pays: [], refunds: [], coupons: []}
    
    for path in file_paths:
        fname = os.path.basename(path)
        uid = fname[:-4]
        
        first_ts = None
        last_ts = None
        total_events = 0
        pays = []
        refunds = []
        coupons = []
        
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    total_events += 1
                    ts_str = row[0].strip()
                    try:
                        ts = float(ts_str)
                    except ValueError:
                        continue
                    
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
                        
                    action = row[1].strip()
                    if action in ("pay", "refund", "enroll_coupon"):
                        item_id = row[2].strip()
                        cursor_time = row[3].strip() if len(row) > 3 else ""
                        source = row[4].strip() if len(row) > 4 else ""
                        user_ans = row[5].strip() if len(row) > 5 else ""
                        platform = row[6].strip() if len(row) > 6 else ""
                        rec = {
                            "user_id": uid,
                            "timestamp": ts,
                            "action_type": action,
                            "item_id": item_id,
                            "cursor_time": cursor_time,
                            "source": source,
                            "user_answer": user_ans,
                            "platform": platform,
                        }
                        records.append(rec)
                        if action == "pay":
                            pays.append((ts, item_id))
                        elif action == "refund":
                            refunds.append((ts, item_id))
                        elif action == "enroll_coupon":
                            coupons.append((ts, item_id))
                            
            if pays or refunds or coupons:
                user_stats[uid] = {
                    "total_events": total_events,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "pays": pays,
                    "refunds": refunds,
                    "coupons": coupons,
                }
        except Exception as e:
            pass
            
    return records, user_stats

def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_files = [str(p) for p in KT4_DIR.glob("*.csv")]
    total_files = len(all_files)
    print(f"Total KT4 user files: {total_files}")
    
    batch_size = 5000
    batches = [all_files[i:i + batch_size] for i in range(0, total_files, batch_size)]
    
    all_records = []
    all_user_stats = {}
    
    num_workers = min(16, os.cpu_count() or 4)
    print(f"Running extraction with {num_workers} processes...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_records, batch_user_stats in executor.map(process_batch, batches):
            all_records.extend(batch_records)
            all_user_stats.update(batch_user_stats)
            
    print(f"Extraction completed in {time.time() - t0:.2f}s")
    print(f"Total transaction records: {len(all_records)}")
    print(f"Total users with transactions: {len(all_user_stats)}")
    
    # 1. Transactions CSV (All pay, refund, coupon events)
    tx_file = DATA_DIR / "kt4_payment_transactions.csv"
    with open(tx_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "timestamp", "datetime_utc", "action_type", "item_id", "cursor_time", "source", "user_answer", "platform"])
        for r in sorted(all_records, key=lambda x: (x["timestamp"], x["user_id"])):
            dt = ""
            try:
                # auto detect unit
                v = r["timestamp"]
                if v >= 1e17: s = v / 1e9
                elif v >= 1e14: s = v / 1e6
                elif v >= 1e11: s = v / 1e3
                else: s = v
                dt = datetime.fromtimestamp(s, tz=timezone.utc).isoformat()
            except Exception:
                pass
            writer.writerow([r["user_id"], r["timestamp"], dt, r["action_type"], r["item_id"], r["cursor_time"], r["source"], r["user_answer"], r["platform"]])
            
    tx_size = tx_file.stat().st_size
    print(f"Saved {tx_file} ({tx_size:,} bytes, {tx_size/1024:.2f} KB, {tx_size/(1024*1024):.2f} MB)")
    
    # 2. Build User Expiry & Repurchase Summary
    # User-level summary
    summary_file = DATA_DIR / "kt4_pass_expiry_repurchase_analysis.csv"
    summary_rows = []
    
    item_counter = Counter()
    pay_users_set = set()
    refund_users_set = set()
    repurchase_users_set = set()
    
    # Duration analysis
    cursor_time_non_empty = 0
    
    for r in all_records:
        if r["action_type"] == "pay":
            item_counter[r["item_id"]] += 1
            pay_users_set.add(r["user_id"])
            if r["cursor_time"]:
                cursor_time_non_empty += 1
        elif r["action_type"] == "refund":
            refund_users_set.add(r["user_id"])
            
    # Per user analysis
    user_summary_list = []
    for uid, stats in all_user_stats.items():
        pays = sorted(stats["pays"], key=lambda x: x[0])
        refunds = sorted(stats["refunds"], key=lambda x: x[0])
        coupons = sorted(stats["coupons"], key=lambda x: x[0])
        
        pay_count = len(pays)
        refund_count = len(refunds)
        coupon_count = len(coupons)
        
        has_pay = pay_count > 0
        has_refund = refund_count > 0
        has_repurchase = pay_count >= 2
        if has_repurchase:
            repurchase_users_set.add(uid)
            
        first_pay_ts = pays[0][0] if pays else None
        last_pay_ts = pays[-1][0] if pays else None
        first_pay_item = pays[0][1] if pays else ""
        last_pay_item = pays[-1][1] if pays else ""
        
        first_refund_ts = refunds[0][0] if refunds else None
        
        # Calculate intervals if multiple payments
        pay_intervals_days = []
        for i in range(1, len(pays)):
            interval_sec = (pays[i][0] - pays[i-1][0]) / 1000.0 # ms to sec approx
            pay_intervals_days.append(round(interval_sec / 86400.0, 2))
            
        # Churn classification:
        # 1. Refund churn: Has refund
        # 2. Non-renewal churn (1-time pay and no subsequent pay)
        # 3. Retained / Repurchased: pay_count >= 2
        is_refund_churn = has_refund
        is_non_renewal_churn = (has_pay and not has_repurchase and not has_refund)
        is_churn = is_refund_churn or is_non_renewal_churn
        
        user_summary_list.append({
            "user_id": uid,
            "has_pay": int(has_pay),
            "pay_count": pay_count,
            "first_pay_item": first_pay_item,
            "last_pay_item": last_pay_item,
            "has_refund": int(has_refund),
            "refund_count": refund_count,
            "has_repurchase": int(has_repurchase),
            "pay_intervals_days": ";".join(str(x) for x in pay_intervals_days),
            "first_pay_ts": first_pay_ts or "",
            "last_pay_ts": last_pay_ts or "",
            "first_event_ts": stats["first_ts"],
            "last_event_ts": stats["last_ts"],
            "total_events": stats["total_events"],
            "is_refund_churn": int(is_refund_churn),
            "is_non_renewal_churn": int(is_non_renewal_churn),
            "is_churn_overall": int(is_churn),
        })
        
    with open(summary_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "user_id", "has_pay", "pay_count", "first_pay_item", "last_pay_item",
            "has_refund", "refund_count", "has_repurchase", "pay_intervals_days",
            "first_pay_ts", "last_pay_ts", "first_event_ts", "last_event_ts",
            "total_events", "is_refund_churn", "is_non_renewal_churn", "is_churn_overall"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(user_summary_list, key=lambda x: x["user_id"]):
            writer.writerow(row)
            
    summary_size = summary_file.stat().st_size
    print(f"Saved {summary_file} ({summary_size:,} bytes, {summary_size/1024:.2f} KB, {summary_size/(1024*1024):.2f} MB)")
    
    # 3. Print Report
    report = {
        "total_kt4_users": total_files,
        "pass_item_types_count": len(item_counter),
        "pass_item_top15": item_counter.most_common(15),
        "pass_all_items": sorted(item_counter.keys()),
        "pay_users_count": len(pay_users_set),
        "refund_users_count": len(refund_users_set),
        "repurchase_users_count": len(repurchase_users_set),
        "one_time_pay_users_count": len(pay_users_set) - len(repurchase_users_set),
        "cursor_time_non_empty_in_pay": cursor_time_non_empty,
        "refund_churn_users": sum(1 for u in user_summary_list if u["is_refund_churn"]),
        "non_renewal_churn_users": sum(1 for u in user_summary_list if u["is_non_renewal_churn"]),
        "total_churn_users": sum(1 for u in user_summary_list if u["is_churn_overall"]),
        "transactions_file_bytes": tx_size,
        "analysis_summary_file_bytes": summary_size,
    }
    
    report_file = DATA_DIR / "kt4_payment_analysis_result.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
