from pathlib import Path
import os
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

    assert "summary.csv: 8 rows" in result.stdout
    assert "report/report.pdf: 10 pages" in result.stdout
    assert "ignored: data/raw/household_power_consumption.txt" in result.stdout


def test_submission_ready_script_reports_placeholder_locations():
    script = Path("scripts/check_submission_ready.sh")
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["SKIP_TESTS"] = "1"

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)

    assert result.returncode == 1
    assert "Placeholder GitHub/author fields remain" in result.stderr
    assert "report/report_draft.md:" in result.stderr
