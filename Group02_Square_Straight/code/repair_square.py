#!/usr/bin/env python3
"""Inspect and (with --apply) restore a matching checkpoint/history PAIR.
Never guess a state from just one file. The original damaged files are preserved.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import shutil


def inspect(cp,hist):
    if not cp.exists() or not hist.exists():return False,None,None,'file missing'
    try:
        with np.load(cp,allow_pickle=False) as z: t=float(z['t'])
        with np.load(hist,allow_pickle=False) as z:
            rows=json.loads(str(z['rows_json']))
            h=float(rows[-1]['t'])
        return abs(t-h)<=1e-8,t,h,'OK' if abs(t-h)<=1e-8 else 'TIME MISMATCH'
    except Exception as e:return False,None,None,f'{type(e).__name__}: {e}'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case_dir',type=Path,help='e.g. %%USERPROFILE%%\\N5000_square_runs\\mirror\\K_1.5_seed20261022')
    ap.add_argument('--apply',action='store_true',help='restore only if previous pair is valid and current pair is invalid')
    a=ap.parse_args();f=a.case_dir
    mainpair=(f/'checkpoint.npz',f/'history.npz')
    prev=(f/'checkpoint.npz.previous',f/'history.npz.previous')
    good,t1,t2,msg=inspect(*mainpair)
    print('CURRENT',msg,'times',t1,t2)
    if good:
        print('NO CHANGE: main files already match')
        return
    ok,p1,p2,msg=inspect(*prev)
    print('BACKUP ',msg,'times',p1,p2)
    if not ok:raise SystemExit('STOP: no known-consistent previous pair. Nothing modified.')
    if not a.apply:
        print('Previous pair can be restored with --apply. Current files stay untouched.')
        return
    for cur in mainpair:
        if cur.exists():shutil.copy2(cur,cur.with_name(cur.name+'.broken_backup'))
    for old,cur in zip(prev,mainpair):
        shutil.copy2(old,cur.with_name(cur.name+'.recovered.tmp'))
    for cur in mainpair:
        cur.with_name(cur.name+'.recovered.tmp').replace(cur)
    print('RESTORED:',inspect(*mainpair),'Originals kept as *.broken_backup')

if __name__=='__main__':main()
