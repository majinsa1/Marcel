#!/usr/bin/env python3
"""Repair a known failed Windows resume after history.tmp.npz rename was denied.

Validates the checkpoint and temporary history time, backs up the current history,
then promotes the complete temporary history into place. Never guesses if unsafe.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def checkpoint_time(path):
    with np.load(path, allow_pickle=False) as z:
        return float(z['t'])


def history_time(path):
    with np.load(path, allow_pickle=False) as z:
        rows = json.loads(str(z['rows_json']))
        if not rows:
            raise ValueError(f'empty history: {path}')
        # Last trajectory arrays should also represent the recorded time.
        if len(z['timeline_t']) == 0:
            raise ValueError('timeline data missing')
        return float(rows[-1]['t'])


def repair(folder):
    folder = Path(folder)
    cp, hist, tmp = (folder / x for x in ('checkpoint.npz', 'history.npz', 'history.tmp.npz'))
    if not cp.is_file():
        raise FileNotFoundError(f'checkpoint not found: {cp}')
    t_cp = checkpoint_time(cp)
    print(f'checkpoint time: {t_cp:g}')
    if hist.is_file():
        t_hist = history_time(hist)
        print(f'current history time: {t_hist:g}')
        if abs(t_cp - t_hist) < 1e-7:
            print('Already consistent; NO CHANGES. You may resume with patched run.py.')
            return
    else:
        print('No history.npz is present.')
    if not tmp.is_file():
        raise RuntimeError('history.tmp.npz missing; cannot safely fix this pair automatically. Do not delete anything.')
    t_tmp = history_time(tmp)
    print(f'temporary history time: {t_tmp:g}')
    if abs(t_cp - t_tmp) > 1e-7:
        raise RuntimeError('Temporary history does NOT match checkpoint. No files changed.')
    if hist.is_file():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = folder / f'history.backup_{stamp}.npz'
        if backup.exists():
            raise FileExistsError(f'Refuse overwrite: {backup}')
        shutil.copy2(hist, backup)
        print(f'Backup saved: {backup.name}')
    # Replacing the file can fail if a separate process (OneDrive, file viewer)
    # holds it open. In that case tmp and old history are both preserved.
    os.replace(tmp, hist)
    assert abs(checkpoint_time(cp) - history_time(hist)) < 1e-7
    print('SUCCESS: checkpoint.npz and history.npz agree. Safe to resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', help='Path to K_0_seed20261022 case directory')
    args = parser.parse_args()
    try:
        repair(args.folder)
    except Exception as exc:
        print(f'REPAIR NOT COMPLETED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
