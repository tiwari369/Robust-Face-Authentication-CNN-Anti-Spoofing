"""View attendance CSV records. Run: python view_attendance.py [YYYY-MM-DD]"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ATTENDANCE_DIR = BASE_DIR / "attendance"


def view_date(date_str: str):
    csv_path = ATTENDANCE_DIR / f"attendance_{date_str}.csv"
    if not csv_path.exists():
        print(f"No attendance found for {date_str}")
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"Attendance file exists but is empty: {csv_path.name}")
        return
    headers = list(rows[0].keys())
    print("\n" + "=" * 90)
    print(f"Attendance - {date_str}")
    print("=" * 90)
    print(" | ".join(f"{h:<18}" for h in headers))
    print("-" * 90)
    for row in rows:
        print(" | ".join(f"{row.get(h, ''):<18}" for h in headers))
    print("=" * 90)
    print(f"Total present: {len(rows)}\n")


def view_all():
    if not ATTENDANCE_DIR.exists():
        print("No attendance folder found.")
        return
    files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"))
    if not files:
        print("No attendance records found.")
        return
    print("Available attendance files:")
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            count = len(list(csv.DictReader(fh)))
        print(f"  {f.name}: {count} record(s)")
    today = datetime.now().strftime("%Y-%m-%d")
    print()
    view_date(today)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_date(sys.argv[1])
    else:
        view_all()
