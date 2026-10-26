#!/usr/bin/env python3
"""Unified visualizations; reads saved trajectories only. No integration.

Output: pure particles, heading-color movie, selected trajectories, snapshots,
        radial density, metrics, side-by-side parameter comparison, HTML index.
"""
import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import visual_style as S


def get_case(folder):
    folder=Path(folder)
    with np.load(folder/'history.npz',allow_pickle=False) as z:
        times=np.asarray(z['timeline_t'],float)
        xy=np.asarray(z['timeline_x'],dtype=np.float32)
        th=np.asarray(z['timeline_theta'],dtype=np.float32)
        track_times=np.asarray(z['track_t'],float)
        track_xy=np.asarray(z['track_x'],dtype=np.float32)
        tracks=np.asarray(z['tracks'],int)
    cfg=json.loads((folder/'config.json').read_text(encoding='utf8'))['params']
    assert cfg['shape']=='circle' and cfg['omega_mean']==cfg['omega_width']==cfg['repulsion_k']==0
    assert xy.shape[1]==cfg['N'] and times[0] == 0
    R=cfg['side']/np.sqrt(np.pi)
    if cfg['boundary'] != 'soft' and np.linalg.norm(xy,axis=2).max()>R+1e-4:
        raise ValueError('Hard-wall particle outside disk: '+str(folder))
    with (folder/'observables.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return dict(folder=folder, cfg=cfg, K=cfg['K'], n=cfg['N'],R=R,
                t=times, xy=xy, th=th, tt=track_times, tx=track_xy,tracks=tracks,
                rows=rows)


def decorate(ax,R):
    ax.set_facecolor('#ffffff')
    ax.set_aspect('equal')
    ax.set_xlim(-1.065*R,1.065*R)
    ax.set_ylim(-1.065*R,1.065*R)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.add_patch(Circle((0,0),R,fill=False,lw=S.EDGE_WIDTH,ec=S.BOUNDARY,zorder=8))


def points(ax,c,k,mode='positions'):
    decorate(ax,c['R'])
    xy=c['xy'][k]
    if mode=='heading':
        return ax.scatter(xy[:,0],xy[:,1],s=S.POINT_SIZE,c=np.mod(c['th'][k],2*np.pi),
                 cmap=S.HEADING,norm=colors.Normalize(0,2*np.pi),linewidths=0,
                 alpha=S.POINT_ALPHA,rasterized=True)
    return ax.scatter(xy[:,0],xy[:,1],s=S.POINT_SIZE,c=S.DOT,
                      alpha=S.POINT_ALPHA,linewidths=0,rasterized=True)


def save_image(fig,path):
    fig.savefig(path,dpi=260,facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)


def single_figures(c,out):
    end=float(c['t'][-1]);indices=[int(np.argmin(abs(c['t']-t))) for t in [0,end/3,2*end/3,end]]
    fig,axs=plt.subplots(1,4,figsize=(15,4),facecolor='#ffffff')
    for ax,idx in zip(axs,indices):
        points(ax,c,idx)
        ax.set_title(f't={c["t"][idx]:.1f}',fontsize=12,color=S.TEXT,pad=10)
    fig.suptitle(f'K={c["K"]:g}  |  {c["cfg"]["boundary"]}  |  N={c["n"]}',color=S.TEXT,fontsize=13)
    save_image(fig,out/'snapshots.png')
    for mode in ('positions','heading'):
        fig,ax=plt.subplots(figsize=(5.4,5.6),facecolor='#ffffff')
        points(ax,c,-1,mode)
        ax.set_title(f'{mode}  |  K={c["K"]:g}  |  t={end:g}',color=S.TEXT,pad=10)
        save_image(fig,out/(mode+'_final.png'))
    # Separate representative tracks, without the 5000-point cloud.
    fig,ax=plt.subplots(figsize=(5.7,5.8),facecolor='#ffffff')
    decorate(ax,c['R'])
    n=min(10,c['tx'].shape[1]);chosen=np.round(np.linspace(0,c['tx'].shape[1]-1,n)).astype(int)
    palette=plt.get_cmap('turbo')(np.linspace(.1,.9,n))
    for j,col in zip(chosen,palette):
        data=c['tx'][:,j,:]
        # Break plot segments across long jumps or sharp reflections in sparse samples
        ax.plot(data[:,0],data[:,1],color=col,lw=1,alpha=.90)
        ax.scatter(data[-1,0],data[-1,1],s=13,color=col,zorder=9)
    ax.set_title(f'{n} tracked particles  |  K={c["K"]:g}',color=S.TEXT,pad=10)
    save_image(fig,out/'selected_trajectories.png')
    # Area-normalized radial density: uniform distribution corresponds to 1.
    rr=np.linalg.norm(c['xy'][-1],axis=1)/c['R']
    bins=np.linspace(0,1,26)
    counts,_=np.histogram(rr,bins)
    shell=np.diff(bins**2)
    rel=counts/(c['n']*shell)
    fig,ax=plt.subplots(figsize=(6.4,3.6),facecolor='white')
    centers=(bins[1:]+bins[:-1])/2
    ax.plot(centers,rel,color='#287d9e',lw=2,marker='.',markersize=5)
    ax.axhline(1,color='#8494a0',ls='--',lw=1)
    ax.set(xlabel='Normalized radius   r / R',ylabel='Relative density  rho(r)/rho0',title=f'Radial density | K={c["K"]:g}')
    ax.grid(alpha=.12)
    fig.tight_layout();save_image(fig,out/'radial_density.png')
    # Small set of diagnostic curves. No theoretical stationary inference.
    metrics=[('P_heading','Global polarization  P'),
             ('local_alignment','Local alignment  Q'),
             ('enrichment','Boundary enrichment  E'),
             ('C_heading','Signed circulation  C')]
    fig,axs=plt.subplots(2,2,figsize=(10,6.7),facecolor='white')
    for ax,(key,title) in zip(axs.flat,metrics):
        t=np.array([float(r['t']) for r in c['rows']]);v=np.array([float(r[key]) for r in c['rows']]);
        ax.plot(t,v,c='#347f9e',lw=1.7)
        if key=='enrichment':ax.axhline(1,ls='--',c='#adb7bd',lw=1)
        if key=='C_heading':ax.axhline(0,ls='--',c='#adb7bd',lw=1)
        ax.set(xlabel='Simulation time t',title=title,xlim=(0,float(t[-1])))
        ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Observables  |  K={c["K"]:g}',fontsize=13)
    fig.tight_layout();save_image(fig,out/'observables.png')


def encode(anim,path,fps):
    if shutil.which('ffmpeg'):
        anim.save(str(path),writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=-1,
                extra_args=['-pix_fmt','yuv420p','-crf','18','-preset','medium','-movflags','+faststart']),dpi=180)
        return path
    else:
        path=path.with_suffix('.gif')
        anim.save(str(path),writer=PillowWriter(fps=fps),dpi=80)
        return path


def is_complete_movie(path, minimum_duration):
    if not Path(path).exists():return False
    if not shutil.which('ffprobe'):return False
    res=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(path)],capture_output=True,text=True)
    try:return res.returncode==0 and float(res.stdout.strip())>=minimum_duration-.12
    except ValueError:return False


def movie(c,out,heading=False,max_frames=120,fps=15,skip_existing=False):
    ids=np.unique(np.round(np.linspace(0,len(c['t'])-1,min(max_frames,len(c['t'])))).astype(int))
    existing=out/('heading.mp4' if heading else 'particles.mp4')
    if skip_existing and is_complete_movie(existing,len(ids)/fps):
        print('SKIP ALREADY VALID:',existing,flush=True);return
    fig,ax=plt.subplots(figsize=(5.5,5.7),facecolor='#ffffff')
    fig.subplots_adjust(left=.03,right=.97,top=.89,bottom=.015)
    pts=points(ax,c,ids[0],'heading' if heading else 'positions')
    clock=ax.text(.98,.98,'',transform=ax.transAxes,ha='right',va='top',color=S.TEXT,fontsize=10)
    ax.set_title(f'N={c["n"]}  |  K={c["K"]:g}  |  circle {c["cfg"]["boundary"]}',color=S.TEXT,pad=10)
    def update(i):
        k=ids[i]
        pts.set_offsets(c['xy'][k])
        if heading:pts.set_array(np.mod(c['th'][k],2*np.pi))
        clock.set_text(f't={c["t"][k]:.1f}')
        return pts,clock
    anim=FuncAnimation(fig,update,frames=len(ids),interval=1000/fps,blit=False)
    dest=encode(anim,out/('heading.mp4' if heading else 'particles.mp4'),fps)
    plt.close(fig)
    print('MOVIE:',dest,flush=True)


def compare(cases,out,movies=False,max_frames=120,fps=15,skip_existing=False):
    if not cases:return
    cases=sorted(cases,key=lambda c:c['K'])
    limit=min(c['t'][-1] for c in cases)
    fig,axs=plt.subplots(1,len(cases),figsize=(4.25*len(cases),4.7),facecolor='#ffffff',squeeze=False)
    for ax,c in zip(axs[0],cases):
        k=int(np.argmin(abs(c['t']-limit)))
        points(ax,c,k)
        ax.set_title(f'K={c["K"]:g}  |  t={c["t"][k]:g}',color=S.TEXT,pad=8)
    fig.tight_layout();save_image(fig,out/'K_comparison_final.png')
    metric=[('P_heading','P'),('local_alignment','Q'),('enrichment','E'),('C_heading','C')]
    fig,axs=plt.subplots(2,2,figsize=(10.5,7),facecolor='white')
    for ax,(key,title) in zip(axs.flat,metric):
        for index,c in enumerate(cases):
            tt=np.asarray([float(r['t']) for r in c['rows']]);yy=np.asarray([float(r[key]) for r in c['rows']]);
            ax.plot(tt,yy,color=S.METRIC_COLORS[index%len(S.METRIC_COLORS)],lw=1.5,label=f'K={c["K"]:g}')
        if key=='enrichment':ax.axhline(1,ls='--',c='#b1bac1',lw=1)
        if key=='C_heading':ax.axhline(0,ls='--',c='#b1bac1',lw=1)
        ax.set(xlim=(0,limit),xlabel='Simulation time t',title=title)
        ax.grid(alpha=.10);ax.legend(fontsize=8,frameon=False)
    fig.suptitle(f'Group 01 | N={cases[0]["n"]} | circle {cases[0]["cfg"]["boundary"]}',fontsize=14)
    fig.tight_layout();save_image(fig,out/'K_comparison_metrics.png')
    if movies and not (skip_existing and is_complete_movie(out/'K_comparison.mp4',max_frames/fps)):
        sample_t=np.linspace(0,limit,max_frames)
        fig,axs=plt.subplots(1,len(cases),figsize=(4.2*len(cases),4.8),facecolor='#ffffff',squeeze=False)
        fig.subplots_adjust(left=.015,right=.985,top=.89,bottom=.04,wspace=.02)
        scat=[]
        for ax,c in zip(axs[0],cases):
            scat.append(points(ax,c,0))
            ax.set_title(f'K={c["K"]:g}',color=S.TEXT,pad=10)
        stamp=fig.text(.5,.955,'',ha='center',color=S.TEXT,fontsize=11)
        def update(i):
            for c,sc in zip(cases,scat):
                idx=int(np.argmin(abs(c['t']-sample_t[i])))
                sc.set_offsets(c['xy'][idx])
            stamp.set_text(f'Straight particles  |  N={cases[0]["n"]}  |  t={sample_t[i]:.1f}')
            return scat+[stamp]
        anim=FuncAnimation(fig,update,frames=max_frames,interval=1000/fps,blit=False)
        target=encode(anim,out/'K_comparison.mp4',fps)
        plt.close(fig)
        print('COMPARISON MOVIE:',target,flush=True)
    # HTML overview including all cases.
    page=['<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Group 01 | Straight particles | Circular wall</title>',
          '<style>body{background:#ffffff;color:#20262b;font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1150px;margin:30px auto;padding:0 16px}h1{font-size:25px}h2{margin-top:34px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.panel{padding:14px;background:#f6f8fa;border:1px solid #dce2e7;border-radius:8px}.panel video,.panel img,.wide{width:100%}a{color:#1769aa}p{color:#56616b}</style></head><body>',
          '<h1>Group 01 · Straight particles in circular boundary</h1>',
          f'<p>N={cases[0]["n"]}, Omega=0, no mechanical force; wall={cases[0]["cfg"]["boundary"]}. Simulation t=0..{limit:g}. Not validated as steady state.</p>']
    if (out/'K_comparison.mp4').exists():page.append('<video class="wide" controls muted playsinline src="K_comparison.mp4"></video>')
    for name in ['K_comparison_final.png','K_comparison_metrics.png']:
        page += [f'<h2>{name}</h2><img class="wide" src="{name}">']
    page+=['<h2>Cases</h2><div class="grid">']
    for c in cases:
        p=out/f'K_{c["K"]:g}_seed{c["cfg"]["seed"]}'
        relative=p.relative_to(out)
        page.append(f'<div class="panel"><h3>K={c["K"]:g}</h3>')
        for filename in ['particles.mp4','heading.mp4']:
            if (p/filename).exists():page.append(f'<video controls muted loop playsinline src="{relative}/{filename}"></video>')
        for filename in ['snapshots.png','selected_trajectories.png','radial_density.png','observables.png']:
            if (p/filename).exists():page.append(f'<h4>{filename}</h4><img src="{relative}/{filename}">')
        page.append('</div>')
    page+=['</div></body></html>']
    (out/'index.html').write_text('\n'.join(page),encoding='utf-8')


def render_group(root,movies=False,max_frames=120,fps=15,skip_existing=False):
    root=Path(root)
    folders=[p for p in root.glob('K_*_seed*') if (p/'history.npz').exists()]
    if not folders:raise FileNotFoundError('No result case history.npz found under '+str(root))
    all_cases=[get_case(p) for p in folders]
    # Compare only cases from same seed, wall geometry and parameters except K.
    seeds=sorted(set(c['cfg']['seed'] for c in all_cases))
    for seed in seeds:
        cases=[c for c in all_cases if c['cfg']['seed']==seed]
        cases.sort(key=lambda c:c['K'])
        for c in cases:
            dest=root/f'K_{c["K"]:g}_seed{seed}'
            dest.mkdir(exist_ok=True)
            single_figures(c,dest)
            if movies:
                movie(c,dest,False,max_frames,fps,skip_existing)
                movie(c,dest,True,max_frames,fps,skip_existing)
        dest=root if len(seeds)==1 else root/f'seed{seed}_comparison'
        dest.mkdir(exist_ok=True)
        compare(cases,dest,movies,max_frames,fps,skip_existing)
        print('OPEN:',(dest/'index.html').resolve(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('../results/mirror'))
    ap.add_argument('--movies',action='store_true')
    ap.add_argument('--max-frames',type=int,default=120)
    ap.add_argument('--fps',type=int,default=15)
    ap.add_argument('--skip-existing-movies',action='store_true')
    a=ap.parse_args()
    render_group(a.root,a.movies,a.max_frames,a.fps,a.skip_existing_movies)
