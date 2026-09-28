"""Archive a finished set of LOSO arms: results, models, figures, config.

Keeps each experiment round self-contained so a later round cannot overwrite an
earlier one -- the reason this exists is that an earlier reorganisation left the
bench sources deleted and only bytecode behind.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, subprocess, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="arm tags to archive, e.g. N1_full_bg N2_full_idle")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--note", default="", help="one line describing the round")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for sub in ("results", "folds", "code", "logs"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    copied = {"results": 0, "folds": 0, "code": 0, "logs": 0}
    for tag in a.tags:
        src = "bench/results/loso_%s.json" % tag
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, "results")); copied["results"] += 1
        for f in glob.glob("bench/results/folds/%s_uid*.pth" % tag):
            shutil.copy2(f, os.path.join(a.out, "folds")); copied["folds"] += 1
        for f in glob.glob("bench/logs/*%s*.log" % tag.split("_")[0]):
            shutil.copy2(f, os.path.join(a.out, "logs")); copied["logs"] += 1

    # The code matters as much as the weights: results are not reproducible
    # without the exact data layer and loss that produced them.
    for f in glob.glob("bench/*.py") + glob.glob("bench/*.sh"):
        shutil.copy2(f, os.path.join(a.out, "code")); copied["code"] += 1

    for extra in ("bench/results/confusion_matrices.json",):
        if os.path.exists(extra):
            shutil.copy2(extra, os.path.join(a.out, "results"))

    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd="hrc_communication",
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        rev = None

    manifest = {
        "archived": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": a.note,
        "tags": a.tags,
        "files": copied,
        "hrc_communication_git": rev,
        "python": sys.version.split()[0],
    }
    for tag in a.tags:
        p = "bench/results/loso_%s.json" % tag
        if os.path.exists(p):
            d = json.load(open(p))
            manifest.setdefault("configs", {})[tag] = d.get("config")
            manifest.setdefault("summary", {})[tag] = d.get("summary")
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=2)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(a.out) for f in fs)
    print("[archive] %s" % a.out)
    for k, v in copied.items():
        print("    %-10s %d files" % (k, v))
    print("    total     %.1f MB" % (total / 1e6))


if __name__ == "__main__":
    main()
