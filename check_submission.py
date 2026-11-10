#!/usr/bin/env python3
"""
DRISHTI 1.0 - check your submission file before handing it in.

    python check_submission.py  myteam_prediction_test1.csv
    python check_submission.py  myteam_prediction_test1.csv  test_set_1

Checks your CSV and tells you whether it is valid. A submission that does not
pass this check will not be accepted.

If you name the test image folder (or it can be found beside the CSV), it also
checks that EVERY test image has a prediction - a missing row counts as a wrong
answer, and nothing else can detect that.
"""

import glob
import os
import re
import sys

import pandas as pd

CLASSES = ["sp-01", "sp-02", "sp-03", "sp-04", "sp-05"]
IMG_EXTS = (".jpg", ".jpeg", ".png")

NAME_RE = re.compile(r"^(?P<team>.+?)_prediction_test(?P<n>[12])\.csv$", re.I)
PLACEHOLDERS = {"myteam", "teamname", "yourteam", "yourteamname", "team",
                "test", "predictions", "prediction", "submission"}


def check_filename(csv_path, problems, notes):
    base = os.path.basename(csv_path)
    m = NAME_RE.match(base)
    if not m:
        problems.append(
            f"the filename '{base}' is wrong. It must be "
            "<teamname>_prediction_test1.csv  or  <teamname>_prediction_test2.csv "
            "- for example  rosefinders_prediction_test2.csv")
        return None, None

    team, n = m.group("team"), m.group("n")
    if team.strip().lower() in PLACEHOLDERS:
        problems.append(f"'{team}' is not a real team name - put your own "
                        "registered team name at the front of the filename")
    if len(team.strip()) < 2:
        problems.append("the team name in the filename is too short")
    if " " in team:
        notes.append("the team name contains a space - no harm, but one word "
                     "is safer")
    return team, "test" + n


def find_test_dir(start="."):
    """The sub-folder with the most images - usually the test set."""
    best, n_best = None, 0
    for d in sorted(glob.glob(os.path.join(start, "*"))):
        if not os.path.isdir(d):
            continue
        n = sum(1 for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                if f.lower().endswith(IMG_EXTS))
        if n > n_best:
            best, n_best = d, n
    return best if n_best >= 10 else None


def check(csv_path, test_dir=None):
    problems, notes = [], []

    if not os.path.exists(csv_path):
        print(f"FAIL: file not found: {csv_path}")
        return False

    if test_dir is None:
        test_dir = find_test_dir(os.path.dirname(os.path.abspath(csv_path)))
        if test_dir:
            print(f"test folder found automatically: {test_dir}")
    if test_dir and not os.path.isdir(test_dir):
        print(f"FAIL: test folder not found: {test_dir}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"FAIL: cannot read the CSV ({e})")
        return False

    team, which = check_filename(csv_path, problems, notes)

    print(f"file    : {os.path.basename(csv_path)}")
    if team:
        print(f"team    : {team}")
        print(f"test set: {which}")
    print(f"columns : {list(df.columns)}")
    print(f"rows    : {len(df)}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    spare = [c for c in df.columns
             if c not in ("file", "predicted") and not c.startswith("unnamed")]
    if spare:
        notes.append(f"extra columns ignored: {', '.join(spare)} - the format is "
                     "just 'file' and 'predicted'")
    if "unnamed: 0" in df.columns:
        problems.append("there is an extra index column - save with "
                        "df.to_csv(path, index=False)")
    if "file" not in df.columns:
        problems.append("no column named 'file'")
    if "predicted" not in df.columns:
        problems.append("no column named 'predicted'")
    if problems:
        print()
        for p in problems:
            print("  PROBLEM:", p)
        print("\nNOT READY")
        return False

    df["file"] = df.file.astype(str).str.strip()
    df["predicted"] = df.predicted.astype(str).str.strip()

    n_path = int(df.file.str.contains(r"[\\/]").sum())
    if n_path:
        problems.append(f"{n_path} rows contain a folder path - use just the "
                        "filename, e.g. img_0007.jpg")

    n_dup = int(df.file.duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate filenames")

    blank = int((df.predicted == "").sum() + df.predicted.isna().sum())
    if blank:
        problems.append(f"{blank} rows have no prediction")

    bad = sorted(set(df.predicted) - set(CLASSES) - {""})
    if bad:
        if all(b.replace(".", "").isdigit() for b in bad):
            problems.append(f"predictions are numbers ({', '.join(bad[:5])}) - "
                            f"write the class names: {', '.join(CLASSES)}")
        else:
            problems.append(f"unrecognised labels: {', '.join(bad[:5])} - "
                            f"allowed: {', '.join(CLASSES)}")

    if test_dir:
        expected = {os.path.basename(f)
                    for f in glob.glob(os.path.join(test_dir, "**", "*"),
                                       recursive=True)
                    if f.lower().endswith(IMG_EXTS)}
        got = set(df.file)
        missing = sorted(expected - got)
        extra = sorted(got - expected)

        print(f"images in folder: {len(expected)}")
        if missing:
            problems.append(f"{len(missing)} test images have no prediction and "
                            f"would be counted wrong - first few: {missing[:5]}")
        if extra:
            notes.append(f"{len(extra)} rows are not test images and will be "
                         f"ignored - first few: {extra[:5]}")
        if not missing and not extra:
            print("every test image has exactly one prediction")
    else:
        notes.append("the test image folder was not given, so it could NOT be "
                     "checked that every image has a prediction. Re-run as: "
                     "python check_submission.py <your.csv> <test folder>")

    print()
    print("prediction counts:")
    print(df.predicted.value_counts().sort_index().to_string())
    if len(df) and df.predicted.value_counts().iloc[0] / len(df) > 0.60:
        notes.append("one class is over 60% of your predictions - worth checking "
                     "your model is loading and preprocessing correctly")

    if notes:
        print()
        for n in notes:
            print("  note:", n)

    print()
    if problems:
        print("NOT READY - fix these:")
        for p in problems:
            print("  -", p)
        return False
    print("READY - this file is valid.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if check(sys.argv[1],
                        sys.argv[2] if len(sys.argv) > 2 else None) else 1)
