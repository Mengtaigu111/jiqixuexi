from pathlib import Path
import subprocess


def test_gpu_watcher_script_waits_then_runs_cuda_experiments():
    script = Path("scripts/watch_gpu_and_run_full_experiments.sh")

    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")

    assert "CHECK_INTERVAL_SECONDS=\"${CHECK_INTERVAL_SECONDS:-300}\"" in text
    assert "nvidia-smi" in text
    assert "USED_MEM" in text
    assert "GPU_UTIL" in text
    assert "DEVICE=cuda" in text
    assert "scripts/run_all_experiments.sh" in text
