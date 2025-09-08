import argparse
import sys

def parse_script_args():
    # If called as: blenderproc run trash_proc.py -- --num_views 5
    # the script args are after the first '--'.
    raw = sys.argv
    if "--" in raw:
        raw = raw[raw.index("--") + 1 :]
    else:
        # When run via: python trash_proc.py --num_views 5
        raw = raw[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, default=3, help="number of camera views")
    parser.add_argument("--apply_dirt", action='store_true', help="whether to apply random dirt to objects")
    return parser.parse_args(raw)