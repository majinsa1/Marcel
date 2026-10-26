#!/usr/bin/env python3
"""Copy compatible previously computed periodic cases into Group02 without recomputing.
Never overwrites an existing target case. Verifies checkpoint/history time matching.
"""
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
from model import Params,Simulation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path.home()/'N5000_periodic_runs'/'periodic')
    ap.add_argument('--dest',type=Path,default=Path.home()/'N5000_square_runs'/'periodic')
    ap.add_argument('--N',type=int,default=5000)
    ap.add_argument('--L',type=float,default=50.)
    ap.add_argument('--seed',type=int,default=20261022)
    ap.add_argument('--K',type=float,nargs='+',default=[0,.5,1.5,3])
    args=ap.parse_args()
    for k in args.K:
        name=f'K_{k:g}_seed{args.seed}';old=args.source/name;new=args.dest/name
        if not old.exists():print('MISSING SOURCE',old);continue
        if new.exists():print('SKIP TARGET EXISTS',new);continue
        try:
            sim=Simulation.restore(old/'checkpoint.npz')
            with np.load(old/'history.npz',allow_pickle=False) as z:
                rows=json.loads(str(z['rows_json']))
            if abs(float(rows[-1]['t'])-sim.t)>1.e-8:
                raise ValueError('checkpoint/history time mismatch')
            expected=Params(N=args.N,side=args.L, K=k,shape='square',boundary='periodic',
                            omega_law='constant',omega_mean=0.,omega_width=0.,seed=args.seed)
            from dataclasses import asdict
            if json.dumps(asdict(sim.p),sort_keys=True)!=json.dumps(asdict(expected),sort_keys=True):
                raise ValueError('Physical parameters differ: refusing to import')
            new.mkdir(parents=True,exist_ok=False)
            for fn in ('checkpoint.npz','history.npz','config.json','observables.csv',
                       'summary.json','stationarity_checks.json','omega_samples.csv',
                       'particle_frequencies.csv'):
                if (old/fn).exists():shutil.copy2(old/fn,new/fn)
            print('IMPORTED',name,'t=',sim.t,'into',new,flush=True)
        except Exception as e:print('REFUSED',name,':',e,flush=True)

if __name__=='__main__':main()
