"""Relay child-process output through notebook Python, without changing arguments."""
from __future__ import annotations

import subprocess


def stream(args, *, cwd=None, env=None, check=False):
    process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if check and code:
        raise subprocess.CalledProcessError(code, args)
    return code
