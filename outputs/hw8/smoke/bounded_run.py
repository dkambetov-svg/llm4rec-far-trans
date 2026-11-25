"""Run one bounded ranker smoke arm; no project code is modified."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/Users/davidkk/llm4rec-hw2/RecSys-LLMs')
PROJECT = ROOT / 'project/far-trans-llm'
SMOKE = ROOT / 'outputs/hw8/smoke'
PY = '/Users/davidkk/llm4rec-hw2/.venv/bin/python'


def memory_pressure():
    try:
        out = subprocess.check_output(['memory_pressure', '-Q'], stderr=subprocess.DEVNULL,
                                      text=True, timeout=5)
        for line in out.splitlines():
            if 'memory free percentage' in line:
                return int(line.rsplit(':', 1)[1].strip().rstrip('%'))
    except Exception:
        pass
    return None


def ps_rss_kb(pid):
    try:
        out = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)],
                                      stderr=subprocess.DEVNULL, text=True, timeout=5)
        return int(out.strip())
    except Exception:
        return 0


def main():
    arm = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    out_dir = SMOKE / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'run.log'
    result_path = out_dir / 'run_wrapper.json'
    cmd = [
        PY, 'ranker.py',
        '--data-dir', '/Users/davidkk/llm4rec-hw2/data/FAR-Trans',
        '--output-dir', str(out_dir),
        '--max-users', '3',
        '--candidate-count', '50',
        '--history-events', '15',
        '--model', 'Qwen/Qwen3-1.7B',
        '--revision', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        '--device', 'mps',
        '--on-invalid', 'fallback',
        '--retry-invalid', '0',
        '--history-mode', arm,
        '--short-term-days', '90',
    ]
    env = os.environ.copy()
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false'})
    before_free = memory_pressure()
    started = time.perf_counter()
    peak_rss_kb = 0
    timed_out = False
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                exit_code = proc.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                peak_rss_kb = max(peak_rss_kb, ps_rss_kb(proc.pid))
                if time.perf_counter() - started >= timeout:
                    timed_out = True
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait()
                    exit_code = -signal.SIGTERM
                    break
    wall = time.perf_counter() - started
    result = {
        'arm': arm, 'command': cmd, 'timeout_seconds': timeout,
        'wall_seconds': wall, 'exit_code': exit_code, 'timed_out': timed_out,
        'peak_child_rss_kb': peak_rss_kb,
        'peak_child_rss_mb': round(peak_rss_kb / 1024, 1),
        'system_memory_free_percent_before': before_free,
        'system_memory_free_percent_after': memory_pressure(),
        'log': str(log_path),
    }
    result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if exit_code == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
