from pathlib import Path
import os
import re
import subprocess
import sys


def test_submission_ready_script_checks_current_artifacts_without_running_pytest():
    script = Path("scripts/check_submission_ready.sh")

    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)

    env = os.environ.copy()
    env["ALLOW_PLACEHOLDERS"] = "1"
    env["PYTHON"] = sys.executable
    env["SKIP_TESTS"] = "1"
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)

    assert "summary.csv: 6 rows" in result.stdout
    assert "models=['dmsaformer', 'lstm', 'transformer']" in result.stdout
    match = re.search(r"report/report\.pdf: (?P<pages>\d+) pages", result.stdout)
    assert match is not None
    assert 12 <= int(match.group("pages")) <= 15
    assert "ignored: data/raw/household_power_consumption.txt" in result.stdout
    assert "dmsaformer calibration: source_model all dmsaformer" in result.stdout


def test_submission_ready_script_reports_placeholder_locations():
    script = Path("scripts/check_submission_ready.sh")
    draft = Path("report/report_draft.md")
    original = draft.read_text(encoding="utf-8")
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["SKIP_TESTS"] = "1"

    try:
        draft.write_text(original + "\n作者贡献与研究领域：待填写\n", encoding="utf-8")
        result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    finally:
        draft.write_text(original, encoding="utf-8")

    assert result.returncode == 1
    assert "Placeholder GitHub/author fields remain" in result.stderr
    assert "report/report_draft.md:" in result.stderr


def test_submission_ready_script_rejects_non_dmsaformer_calibration_source():
    script = Path("scripts/check_submission_ready.sh")
    choices = Path("results/metrics/dmsaformer_calibration_choices.csv")
    original = choices.read_text(encoding="utf-8")
    env = os.environ.copy()
    env["ALLOW_PLACEHOLDERS"] = "1"
    env["PYTHON"] = sys.executable
    env["SKIP_TESTS"] = "1"

    try:
        choices.write_text(original.replace(",dmsaformer,", ",lstm,", 1), encoding="utf-8")
        result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    finally:
        choices.write_text(original, encoding="utf-8")

    assert result.returncode == 1
    assert "DMSAFormer calibration must use only DMSAFormer predictions" in result.stderr
