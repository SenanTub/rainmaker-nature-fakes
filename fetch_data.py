"""Download PAHG Level II radar files for the analysis window from NOAA's
public archive on AWS. No account needed.

Run:  python fetch_data.py
"""
from datetime import datetime, timedelta
import re

import config


BUCKETS = ["unidata-nexrad-level2", "noaa-nexrad-level2"]  # newest first
NAME_RE = re.compile(r"(?P<site>[A-Z]{4})(?P<date>\d{8})_(?P<time>\d{6})")


def _client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        raise SystemExit("boto3 is missing. Install it with:  pip install boto3")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_keys(s3, bucket, start, end):
    keys = []
    day = start.date()
    while day <= end.date():
        prefix = f"{day:%Y/%m/%d}/{config.SITE}/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_MDM"):
                    continue
                m = NAME_RE.search(key.split("/")[-1])
                if not m:
                    continue
                t = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
                if start <= t <= end:
                    keys.append(key)
        day += timedelta(days=1)
    return sorted(keys)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download NEXRAD Level II files")
    ap.add_argument("--site", default=config.SITE)
    ap.add_argument("--start", default=config.START_UTC, help="UTC, e.g. 2026-01-23T19:00")
    ap.add_argument("--end", default=config.END_UTC)
    ap.add_argument("--dest", default=str(config.RADAR_DIR))
    a = ap.parse_args()
    config.SITE = a.site
    config.RADAR_DIR = __import__("pathlib").Path(a.dest)
    start = datetime.fromisoformat(a.start) - timedelta(minutes=10)
    end = datetime.fromisoformat(a.end) + timedelta(minutes=10)
    config.RADAR_DIR.mkdir(parents=True, exist_ok=True)
    s3 = _client()

    for bucket in BUCKETS:
        try:
            keys = list_keys(s3, bucket, start, end)
        except Exception as exc:  # bucket renamed, network, etc.
            print(f"[{bucket}] could not list: {exc}")
            continue
        if not keys:
            print(f"[{bucket}] no files found for {config.SITE} in the window")
            continue
        print(f"[{bucket}] {len(keys)} files")
        for key in keys:
            dest = config.RADAR_DIR / key.split("/")[-1]
            if dest.exists():
                continue
            print("  downloading", dest.name)
            s3.download_file(bucket, key, str(dest))
        print("Done. Files are in", config.RADAR_DIR)
        return

    raise SystemExit(
        "No files found in any bucket. Search 'NEXRAD on AWS' for the current "
        "bucket name and add it to BUCKETS at the top of this file."
    )


if __name__ == "__main__":
    main()
