"""Rebuild the label tensors of an existing corpus with a newer labels.py.

Features are copied unchanged; only `labels` (and the taxonomy fields in
`metadata`) are replaced. Used to move from hrc_communication/other/ -- built
by a labeller that removed Lift entirely, which left 41.6% of the real lifting
frames scored as "No Related Task" -- to the labels of LSTM_HRC commit f30a14d,
which trims each Lift span to the part before Place/Align starts and derives
idle from the frames no annotation covers.

Labels are generated from the ELAN exports (label__<uid-NN_take-NN>.json), with
the frame count of each .pt file, so they line up frame for frame. The source
files in this corpus have trim_start == 0 (checked), so no offset is needed.
Mirrored copies carry their source take's labels, as in augment_dataset.py.

Usage:
  python bench/relabel.py --labels-pkg <dir containing pkg/labels.py> \
      --src hrc_communication/other --dst data_f30a14d \
      --annotations ceiling_installation
"""
import argparse, glob, logging, os, re, sys
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-pkg", required=True,
                    help="folder holding pkg/labels.py and pkg/io_utils.py from the commit")
    ap.add_argument("--src", default="hrc_communication/other")
    ap.add_argument("--dst", default="data_f30a14d")
    ap.add_argument("--annotations", default="ceiling_installation")
    ap.add_argument("--source-tag", default="LSTM_HRC@f30a14d")
    a = ap.parse_args()

    sys.path.insert(0, a.labels_pkg)
    import pkg.labels as L
    log = logging.getLogger("relabel")
    ann = Path(a.annotations)

    jobs = [("original", sorted(glob.glob(os.path.join(a.src, "original", "*.pt"))))]
    # One mirror copy per recording is all training uses (_aug-02/03 are
    # byte-identical duplicates), so only _aug-01 is rebuilt.
    jobs.append(("augmented_mirror",
                 sorted(glob.glob(os.path.join(a.src, "augmented_mirror", "*_aug-01.pt")))))

    label_map = {tid: {name: list(steps)} for tid, name in L.TASK_NAMES.items()
                 for steps in [[s for s, t in L.STEP_TO_TASK.items() if t == tid]]}
    for sub, files in jobs:
        out_dir = os.path.join(a.dst, sub)
        os.makedirs(out_dir, exist_ok=True)
        for i, p in enumerate(files):
            take = re.search(r"(uid-\d+_take-[\d-]+?)(?:_aug|\.pt)", os.path.basename(p)).group(1)
            d = torch.load(p, map_location="cpu", weights_only=False)
            T = len(d["labels"]["task_id"])
            labels, _ = L.extract_labels(take, ann, T, log)
            assert set(labels) == set(d["labels"]), "label keys changed: %s" % p
            for k, v in labels.items():
                assert v.shape[0] == T, (p, k, v.shape)
            d["labels"] = labels
            md = d["metadata"]
            md["task_names"] = dict(L.TASK_NAMES)
            md["label_map"] = label_map
            md["label_source"] = a.source_tag
            torch.save(d, os.path.join(out_dir, os.path.basename(p)))
            if i % 20 == 0:
                print("  %s %3d/%d  %s" % (sub, i + 1, len(files), take), flush=True)
        print("[%s] %d files -> %s" % (sub, len(files), out_dir), flush=True)


if __name__ == "__main__":
    main()
