#!/usr/bin/env python3
"""Group 01: 5000 straight particles inside a circular boundary.

Three boundary variants, kept in separate subdirectories:
  mirror: specular reflection
  sliding: normal velocity removed at contact (heading unchanged)
  soft: finite outward potential layer; may leak outside R

Run only one boundary at a time to keep comparisons interpretable.
"""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from model import Params
from run import DEFAULT_RUN, run_case

NAME = {'mirror':'specular', 'sliding':'sliding', 'soft':'soft'}

def parse_args():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument('--boundary',choices=list(NAME),default='mirror')
    a.add_argument('--K',type=float,nargs='+',default=[0.,.5,1.5,3.])
    a.add_argument('--N',type=int,default=5000)
    a.add_argument('--side',type=float,default=50.,help='equal-area square side; R=side/sqrt(pi)')
    a.add_argument('--r-align',type=float,default=1.)
    a.add_argument('--v0',type=float,default=1.)
    a.add_argument('--end',type=float,default=120.)
    a.add_argument('--dt',type=float,default=.02)
    a.add_argument('--sample-dt',type=float,default=1.)
    a.add_argument('--seed',type=int,default=20261022)
    a.add_argument('--edge-width',type=float,default=2.)
    a.add_argument('--wall-width',type=float,default=2.,help='soft potential layer thickness')
    a.add_argument('--wall-k',type=float,default=4.,help='soft potential strength')
    a.add_argument('--root',type=Path,default=Path('results'))
    a.add_argument('--resume',action='store_true',help='continue an existing case to a later --end')
    a.add_argument('--plot',action='store_true',help='render static figures for selected cases')
    a.add_argument('--movies',action='store_true',help='also export MP4 animations')
    a.add_argument('--max-frames',type=int,default=120)
    return a.parse_args()

def main():
    a=parse_args()
    if a.N<1 or a.end<=0 or a.dt<=0 or a.sample_dt<=0 or any((not math.isfinite(k) or k<0) for k in a.K):
        raise SystemExit('N>=1, positive times and finite nonnegative K required')
    folders=[]
    for K in a.K:
        p=Params(N=a.N, side=a.side, v0=a.v0, K=K, align_radius=a.r_align,
                 include_self=True,repulsion_k=0., omega_mean=0., omega_width=0.,
                 omega_law='constant',shape='circle',boundary=NAME[a.boundary],
                 edge_width=a.edge_width, wall_width=a.wall_width,wall_k=a.wall_k,
                 dt=a.dt,seed=a.seed,init='random')
        p.validate()
        out=a.root/a.boundary/f'K_{K:g}_seed{a.seed}'
        folders.append(out)
        run=dict(DEFAULT_RUN, mode='fixed',end=a.end,sample_dt=a.sample_dt,
                 render=False,record_time=200.,early_time=min(a.end,60.))
        exists=(out/'checkpoint.npz').exists()
        if exists and a.resume:
            old=json.loads((out/'config.json').read_text(encoding='utf-8'))['run']
            # Preserve diagnostic sampling/window settings of older example data.
            for key in ('sample_dt','window','record_time','mean_tol','std_tol','tv_tol'):
                run[key]=old[key]
        if exists and not a.resume:
            raise SystemExit(f'{out} already exists: use --resume with later --end, or choose a new --root')
        print(f'RUN boundary={a.boundary} K={K:g} N={a.N} R={p.radius:.5f} t_end={a.end:g}',flush=True)
        run_case({'params':asdict(p),'run':run},out,resume=exists)
        print(f'DONE {out}',flush=True)
    if a.plot or a.movies:
        from plot_group import render_group
        render_group(a.root/a.boundary, movies=a.movies,max_frames=a.max_frames)
    meta={'topic':'Group01 straight particles, circle','boundary':a.boundary,
          'K':a.K,'N':a.N,'R':a.side/math.sqrt(math.pi),'Omega':0,
          'repulsion':0,'seed':a.seed,'end':a.end,'dt':a.dt,
          'message':'Fixed-time simulation; no stationary-state claim'}
    (a.root/a.boundary/'group_info.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

if __name__=='__main__':main()
