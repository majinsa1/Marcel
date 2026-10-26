#!/usr/bin/env python3
"""GROUP 02 - N straight particles in an equal-area square; 5 boundary controls.

Mirror, tangent sliding, mirror/sliding mixed, soft repulsive wall, and periodic.
All cases have omega_i=0, zero particle-particle mechanical repulsion.
K, system area, initial random seed, dt, v0, r_align are held fixed in comparisons.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from model import Params, Simulation
from run import DEFAULT_RUN, run_case

BOUNDARIES = {'mirror':'specular', 'sliding':'sliding', 'mixed':'mixed',
              'soft':'soft', 'periodic':'periodic'}


def case_dir(root, boundary, k, seed):
    return root / boundary / f'K_{k:g}_seed{seed}'


def run():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--boundaries',nargs='+',choices=tuple(BOUNDARIES),default=list(BOUNDARIES))
    ap.add_argument('--K',type=float,nargs='+',default=[0,0.5,1.5,3])
    ap.add_argument('--N',type=int,default=5000)
    ap.add_argument('--L',type=float,default=50.)
    ap.add_argument('--v0',type=float,default=1.)
    ap.add_argument('--r-align',type=float,default=1.)
    ap.add_argument('--dt',type=float,default=.02)
    ap.add_argument('--sample-dt',type=float,default=1.)
    ap.add_argument('--end',type=float,default=300.)
    ap.add_argument('--seed',type=int,default=20261022)
    ap.add_argument('--wall-width',type=float,default=2.)
    ap.add_argument('--wall-k',type=float,default=4.)
    ap.add_argument('--p-spec',type=float,default=.5,help='Probability of mirror response at each NEW contact in mixed case')
    ap.add_argument('--root',type=Path,default=Path.home()/'N5000_square_runs')
    ap.add_argument('--resume',action='store_true',help='resume matching incomplete cases; skip completed cases')
    ap.add_argument('--plot',action='store_true',help='make white-background PNGs and HTML index')
    ap.add_argument('--movies',action='store_true',help='also export PNG-based position movies (MP4 if ffmpeg available)')
    ap.add_argument('--heading-movies',action='store_true',help='also export heading-colored animations')
    ap.add_argument('--max-frames',type=int,default=120)
    args=ap.parse_args()
    if args.N<1 or min(args.end,args.dt,args.sample_dt,args.L,args.r_align,args.v0)<=0 or args.max_frames<2:
        ap.error('N>0; time, lengths, speeds >0 and --max-frames >=2 are required')
    if len(args.boundaries)!=len(set(args.boundaries)) or len(args.K)!=len(set(args.K)):
        ap.error('Repeated boundary or K value')
    if any(not math.isfinite(k) or k<0 for k in args.K):
        ap.error('K must be a finite nonnegative number')
    if not 0<=args.p_spec<=1: ap.error('p-spec must be between 0 and 1')
    root=args.root.resolve()
    print('RESULT ROOT:',root,flush=True)
    errors=[]
    for boundary in args.boundaries:
        for k in args.K:
            params=Params(N=args.N,side=args.L,v0=args.v0,K=k,align_radius=args.r_align,
                          omega_mean=0.,omega_width=0.,omega_law='constant',
                          repulsion_k=0.,shape='square',boundary=BOUNDARIES[boundary],
                          wall_width=args.wall_width,wall_k=args.wall_k,
                          wall_p_spec=(args.p_spec,)*4,
                          seed=args.seed,dt=args.dt,init='random')
            params.validate()
            folder=case_dir(root,boundary,k,args.seed)
            cp=folder/'checkpoint.npz'
            if cp.exists():
                if not args.resume:
                    raise SystemExit(f'EXISTS: {folder}  (add --resume or choose a different --root)')
                try:
                    sim=Simulation.restore(cp)
                    # Explicitly reject a parameter change before skipping a completed case.
                    if json.dumps(asdict(sim.p),sort_keys=True)!=json.dumps(asdict(params),sort_keys=True):
                        raise ValueError('Existing checkpoint parameters differ; choose a NEW root')
                    with __import__('numpy').load(folder/'history.npz',allow_pickle=False) as z:
                        last=json.loads(str(z['rows_json']))[-1]['t']
                    if abs(last-sim.t)>1.e-8:
                        raise ValueError('Checkpoint/history mismatch: recover pair before resuming')
                    if sim.t>=args.end-1.e-8:
                        print(f'SKIP {boundary} K={k:g}  already t={sim.t:g}',flush=True)
                        continue
                except Exception as e:
                    raise RuntimeError(f'Cannot resume {folder}: {e}') from e
            runopts=dict(DEFAULT_RUN,mode='fixed',end=args.end,sample_dt=args.sample_dt,
                         early_time=min(60.,args.end),render=False)
            if cp.exists():
                old=json.loads((folder/'config.json').read_text(encoding='utf-8'))
                for key in ('sample_dt','window','record_time','mean_tol','std_tol','tv_tol'):
                    runopts[key]=old['run'][key]
            print(f'RUN {boundary:8s}  K={k:g}  N={params.N}  L={params.side:g}  '
                  f'END={args.end:g}  RESUME={cp.exists()}',flush=True)
            # Every case is isolated; results of earlier cases remain intact on error.
            run_case({'params':asdict(params),'run':runopts},folder,resume=cp.exists())
            print('DONE',folder,flush=True)
    root.mkdir(parents=True,exist_ok=True)
    (root/'experiment_info.json').write_text(json.dumps(dict(group='GROUP02 square straight',
              N=args.N,L=args.L,rho=args.N/args.L**2,K=args.K,boundaries=args.boundaries,
              Omega=0,particle_force=0,seed=args.seed,target_time=args.end,
              interpretation='soft wall is a force, not a hard constraint; periodic has no real wall'),
              indent=2),encoding='utf-8')
    if args.plot or args.movies or args.heading_movies:
        from plot_square import render
        render(root,boundaries=args.boundaries,movies=args.movies,
               heading_movies=args.heading_movies,max_frames=args.max_frames)

if __name__=='__main__':
    run()
