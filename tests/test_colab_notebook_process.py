"""Notebook stdout transport checks using tiny CPU-only child processes."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from colab_notebook_process import stream


def test_both_child_streams_reach_notebook_python(capsys):
    args = [sys.executable, '-u', '-c', "import sys; print('CACHE_READY',flush=True); print('progress',file=sys.stderr,flush=True)"]
    assert stream(args, check=True) == 0
    output = capsys.readouterr().out
    assert 'CACHE_READY' in output and 'progress' in output


def test_failure_output_is_visible_before_check_raises(capsys):
    args = [sys.executable, '-u', '-c', "print('preflight failed',flush=True); raise SystemExit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        stream(args, check=True)
    assert error.value.returncode == 3
    assert 'preflight failed' in capsys.readouterr().out


def test_unchecked_failure_code_is_returned_for_archive_download(capsys):
    assert stream([sys.executable, '-c', 'raise SystemExit(1)']) == 1
