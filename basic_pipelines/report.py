import argparse
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path


# load_entries: fetch raw rows from SQLite DB
def load_entries(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT id, created_at, last_seen_ts, missed_frames, total_hits FROM reid_entries"
        ).fetchall()
    entries = []
    for rid, created_at, last_seen_ts, missed_frames, hits in rows:
        entries.append(
            {
                "id": rid,
                "created_at": created_at,
                "last_seen_ts": last_seen_ts or created_at,
                "missed_frames": missed_frames or 0,
                "hits": hits or 0,
            }
        )
    return entries


# summarize: compute total IDs, currently active ones and dwell times
def summarize(entries, fps, active_gap_seconds):
    stats = {
        "total_ids": len(entries),
        "active_ids": 0,
        "dwell_seconds": [],
        "today_ids": 0,
    }
    frame_gap = int(fps * active_gap_seconds)
    local_now = datetime.now().astimezone()
    start_of_today = local_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    for entry in entries:
        if entry["missed_frames"] <= frame_gap:
            stats["active_ids"] += 1
        last_seen_ts = entry["last_seen_ts"] or entry["created_at"]
        duration = max(0, last_seen_ts - entry["created_at"])
        stats["dwell_seconds"].append(duration)
        if last_seen_ts >= start_of_today:
            stats["today_ids"] += 1
    return stats


# format_report: render summary stats as printable text
def format_report(stats):
    dwell = stats["dwell_seconds"]
    avg = statistics.mean(dwell) if dwell else 0.0
    median = statistics.median(dwell) if dwell else 0.0
    lines = [
        f"Total IDs ever seen : {stats['total_ids']}",
        f"IDs seen today      : {stats['today_ids']}",
        f"Currently active    : {stats['active_ids']}",
        f"Average dwell (s)   : {avg:.1f}",
        f"Median dwell (s)    : {median:.1f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize tracker SQLite statistics.")
    parser.add_argument(
        "--db-path",
        default="/home/pi/hailo-faceid-detection/hailo-rpi5-examples/data/faceid.sqlite",
        help="Path to SQLite database created by Tracker.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Approximate FPS of detection pipeline.",
    )
    parser.add_argument(
        "--active-gap-seconds",
        type=float,
        default=2.0,
        help="How long (seconds) a person can disappear and still be counted as active.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    entries = load_entries(str(db_path))
    stats = summarize(entries, fps=args.fps, active_gap_seconds=args.active_gap_seconds)
    print(format_report(stats))


if __name__ == "__main__":
    main()

