#!/usr/bin/env python3
"""Uniform white-background visualizations for GROUP02 square straight particles.

Reads previously saved histories; rendering NEVER advances or modifies the simulation.
Distinguishes physical walls from periodic basic-cell edges and potential onset.
"""
import argparse
import csv
import html
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import visual_style as S

NAMES={'mirror':'Mirror reflection','sliding':'Tangential sliding',
       'mixed':'Mixed mirror / sliding','soft':'Soft wall potential',
       'periodic':'Periodic (no walls)'}
DPI=195


def load_case(folder):
    folder=Path(folder)
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))['params']
    with np.load(folder/'history.npz',allow_pickle=False) as z:
        d={k:z[k].copy() for k in ('timeline_t','timeline_x','timeline_theta','track_t','track_x')}
    with open(folder/'observables.csv',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return dict(folder=folder,cfg=cfg,t=d['timeline_t'],x=d['timeline_x'],
                theta=d['timeline_theta'],tt=d['track_t'],tx=d['track_x'],
                rows=rows,K=float(cfg['K']),boundary=('mirror' if cfg['boundary']=='specular' else cfg['boundary']))


def draw_box(ax,c):
    L=float(c['cfg']['side']);h=L/2
    ax.set_facecolor('white');ax.set_aspect('equal')
    # Same axis size in every comparison, no silent cropping of soft-wall crossings.
    margin=2.5
    ax.set_xlim(-h-margin,h+margin);ax.set_ylim(-h-margin,h+margin)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    if c['boundary']=='periodic':
        style=dict(color=S.BOUNDARY,lw=1.0,ls='--',alpha=.8)
    else:
        style=dict(color=S.BOUNDARY,lw=1.2,ls='-',alpha=.95)
    ax.plot([-h,h,h,-h,-h],[-h,-h,h,h,-h],**style)
    if c['boundary']=='soft':
        inner=h-float(c['cfg']['wall_width'])
        ax.plot([-inner,inner,inner,-inner,-inner],[-inner,-inner,inner,inner,-inner],
                color='#a9a9a9',lw=.7,ls=':',alpha=.8)


def dots(ax,c,i,heading=False,marker_area=4.2):
    draw_box(ax,c)
    kw=dict(s=marker_area,linewidths=0,alpha=.86,rasterized=True)
    if heading:
        kw.update(c=np.mod(c['theta'][i],2*np.pi),cmap=S.HEADING,vmin=0,vmax=2*np.pi)
    else:
        kw.update(color=S.DOT)
    return ax.scatter(c['x'][i,:,0],c['x'][i,:,1],**kw)


def save(fig,out,dpi=DPI):
    fig.savefig(out,dpi=dpi,facecolor='white',bbox_inches='tight',pad_inches=.13)
    plt.close(fig)


def field(c,key):
    return np.asarray([float(row['t']) for row in c['rows']]),np.asarray([float(row[key]) for row in c['rows']])


def images(c):
    out=c['folder'];end=float(c['t'][-1]);n=int(c['cfg']['N']);b=c['boundary']
    idx=[int(np.abs(c['t']-s).argmin()) for s in (0,end/3,2*end/3,end)]
    fig,axs=plt.subplots(1,4,figsize=(17.2,4.65))
    for ax,i in zip(axs,idx):
        dots(ax,c,i);ax.set_title('t = %.1f'%c['t'][i],fontsize=12)
    fig.suptitle(f'{NAMES[b]} | N={n:,} | K={c["K"]:g}',fontsize=14)
    save(fig,out/'snapshots.png',dpi=185)
    for mode in ('particles','heading'):
        fig,ax=plt.subplots(figsize=(7.3,7.6))
        dots(ax,c,-1,heading=(mode=='heading'),marker_area=5.)
        ax.set_title(f'{NAMES[b]}  |  K={c["K"]:g}  |  t={end:g}',fontsize=13)
        save(fig,out/(mode+'_final.png'),dpi=228)
    # Selected track segments only; break periodic boundary wraps, no spurious long line.
    fig,ax=plt.subplots(figsize=(7.2,7.5));draw_box(ax,c)
    picks=np.linspace(0,c['tx'].shape[1]-1,min(8,c['tx'].shape[1])).round().astype(int)
    cols=plt.get_cmap('tab10')(np.arange(len(picks)))
    L=c['cfg']['side']
    for tr,col in zip(picks,cols):
        x=c['tx'][:,tr].astype(float)
        if b=='periodic':
            cuts=np.flatnonzero(np.any(np.abs(np.diff(x,axis=0))>L/2,axis=1))+1
            chunks=np.split(x,cuts)
        else:chunks=[x]
        for chunk in chunks:
            if len(chunk)>1:ax.plot(chunk[:,0],chunk[:,1],c=col,lw=1.0,alpha=.8)
        ax.scatter(x[-1,0],x[-1,1],c=[col],s=12,zorder=5)
    ax.set_title('Eight sampled trajectories (not all particle paths)',fontsize=12)
    save(fig,out/'sampled_tracks.png',dpi=185)
    # Density grid includes soft-wall excursions beyond nominal square.
    h=c['cfg']['side']/2;extent=h+2.5
    hist,_,_=np.histogram2d(c['x'][-1,:,0],c['x'][-1,:,1],
                 bins=36,range=[[-extent,extent],[-extent,extent]])
    relative=hist/(hist.sum()/hist.size)
    fig,ax=plt.subplots(figsize=(7.1,6.5))
    im=ax.imshow(relative.T,extent=[-extent,extent,-extent,extent],origin='lower',
                 interpolation='nearest',cmap=S.DENSITY,vmin=0)
    ax.set_aspect('equal');ax.set_title('Spatial number density / grid mean',fontsize=12)
    fig.colorbar(im,ax=ax,shrink=.78)
    save(fig,out/'density.png',dpi=175)
    # Valid metrics vary by boundary. No fictitious wall statistics for periodic.
    metrics=[('P_heading','Global polarization P'),('local_alignment','Neighbor alignment Q'),
             ('mean_speed','Measured speed / v0')]
    if b!='periodic':
        metrics += [('edge_fraction','Fraction in inner edge strip')]
        if b=='soft':metrics += [('outside_fraction','Fraction outside nominal square')]
    fig,axs=plt.subplots(len(metrics),1,figsize=(9.2,2.15*len(metrics)+.8),sharex=True)
    axs=np.atleast_1d(axs)
    for ax,(key,title) in zip(axs,metrics):
        t,y=field(c,key);ax.plot(t,y,color=S.DOT,lw=1.4)
        ax.set_ylabel(title,fontsize=9);ax.grid(alpha=.13)
        ax.spines[['right','top']].set_visible(False)
    axs[-1].set_xlabel('Simulation time t')
    fig.suptitle(f'{NAMES[b]} | K={c["K"]:g}',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.96]);save(fig,out/'metrics.png',dpi=150)


def encode(anim,path,fps=15,dpi=180):
    if shutil.which('ffmpeg'):
        anim.save(str(path),writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=5500,
                         extra_args=['-pix_fmt','yuv420p','-crf','18','-movflags','+faststart']),dpi=dpi)
    else:
        path=path.with_suffix('.gif')
        print('ffmpeg unavailable: GIF fallback',path,flush=True)
        anim.save(str(path),writer=PillowWriter(fps=fps),dpi=85)
    return path


def movie(c,heading=False,max_frames=120):
    idx=np.unique(np.linspace(0,len(c['t'])-1,min(max_frames,len(c['t']))).round().astype(int))
    fig,ax=plt.subplots(figsize=(6.0,6.4));fig.subplots_adjust(left=.05,right=.95,bottom=.02,top=.89)
    sc=dots(ax,c,idx[0],heading=heading,marker_area=3.8)
    ax.set_title(f'{NAMES[c["boundary"]]}  |  N={c["cfg"]["N"]:,}  |  K={c["K"]:g}',fontsize=12)
    tx=fig.text(.5,.958,'',ha='center',fontsize=11,color=S.TEXT)
    def update(j):
        i=idx[j];sc.set_offsets(c['x'][i])
        if heading:sc.set_array(np.mod(c['theta'][i],2*np.pi))
        tx.set_text(f't = {c["t"][i]:.1f}')
        return sc,tx
    anim=FuncAnimation(fig,update,frames=len(idx),interval=1000/15,blit=False)
    target=encode(anim,c['folder']/('heading.mp4' if heading else 'particles.mp4'))
    plt.close(fig);print('VIDEO',target,flush=True)


def comparison(cases,folder,movies=False,max_frames=120,label=None):
    cases=list(cases)
    if not cases:return
    folder.mkdir(parents=True,exist_ok=True)
    t=min(float(c['t'][-1]) for c in cases)
    fig,ax=plt.subplots(1,len(cases),figsize=(4.4*len(cases),4.65),squeeze=False)
    for a,c in zip(ax[0],cases):
        i=int(np.abs(c['t']-t).argmin());dots(a,c,i,marker_area=4.0)
        title=f'K={c["K"]:g}' if label=='K' else NAMES[c['boundary']]
        a.set_title(title,fontsize=12)
    fig.suptitle(f'GROUP02 | {label or "Square boundary"} | common t={t:.1f}',fontsize=15)
    save(fig,folder/'comparison_final.png',dpi=220)
    fig,ax=plt.subplots(1,2,figsize=(10.5,4))
    for j,c in enumerate(cases):
        name=f'K={c["K"]:g}' if label=='K' else NAMES[c['boundary']]
        for a,key,title in zip(ax,('P_heading','local_alignment'),('Polar order P','Local alignment Q')):
            tt,v=field(c,key);keep=tt<=t+1.e-8
            a.plot(tt[keep],v[keep],label=name,color=S.METRIC_COLORS[j%len(S.METRIC_COLORS)])
            a.set_title(title);a.set_xlabel('t');a.grid(alpha=.14);a.spines[['right','top']].set_visible(False)
    ax[1].legend(frameon=False,fontsize=8)
    save(fig,folder/'comparison_metrics.png',dpi=200)
    if movies:
        times=np.linspace(0,t,min(max_frames,120))
        fig,axs=plt.subplots(1,len(cases),figsize=(4.0*len(cases),4.4),squeeze=False)
        scat=[]
        for a,c in zip(axs[0],cases):
            scat.append(dots(a,c,0,marker_area=3.6))
            a.set_title(f'K={c["K"]:g}' if label=='K' else NAMES[c['boundary']],fontsize=10)
        time_label=fig.text(.5,.975,'',ha='center',fontsize=12,color=S.TEXT)
        def update(j):
            for c,s in zip(cases,scat):
                i=int(np.abs(c['t']-times[j]).argmin());s.set_offsets(c['x'][i])
            time_label.set_text(f'Common time t = {times[j]:.1f}')
            return scat+[time_label]
        anim=FuncAnimation(fig,update,frames=len(times),interval=1000/15,blit=False)
        target=encode(anim,folder/'comparison.mp4',dpi=115)
        plt.close(fig);print('COMPARISON',target,flush=True)


def render(root,boundaries=None,movies=False,heading_movies=False,max_frames=120):
    root=Path(root)
    if boundaries is None:boundaries=list(NAMES)
    allcases={}
    for name in boundaries:
        cases=[]
        for folder in sorted((root/name).glob('K_*_seed*')) if (root/name).exists() else []:
            if not (folder/'history.npz').exists() or not (folder/'config.json').exists():continue
            c=load_case(folder)
            if c['cfg']['shape']!='square' or c['cfg']['omega_mean']!=0:continue
            cases.append(c)
            images(c)
            if movies:movie(c,max_frames=max_frames)
            if heading_movies:movie(c,heading=True,max_frames=max_frames)
        cases.sort(key=lambda c:c['K'])
        if cases:
            comparison(cases,root/name,movies=movies,max_frames=max_frames,label='K')
            allcases[name]=cases
    allK=sorted({c['K'] for group in allcases.values() for c in group})
    for k in allK:
        cases=[next((c for c in allcases.get(b,[]) if c['K']==k),None) for b in NAMES]
        cases=[c for c in cases if c is not None]
        if len(cases)>1:comparison(cases,root/'compare_boundaries'/f'K_{k:g}',
                                   movies=movies,max_frames=max_frames,label='boundary')
    html_lines=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
     '<meta name="viewport" content="width=device-width,initial-scale=1">',
     '<title>GROUP02: Square straight particles</title>',
     '<style>body{font-family:system-ui,Arial,sans-serif;color:#222;background:white;max-width:1200px;margin:25px auto;padding:0 15px}img,video{max-width:100%;height:auto}a{color:#2878a8}section{border:1px solid #dde3e8;border-radius:10px;padding:16px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}</style>',
     '<h1>第二组：纯直线粒子 · 方形边界对照</h1>',
     '<p>五种边界：镜面、切向滑移、反射/滑移混合、软势、周期性。Ω=0；点粒子，不含粒子间机械排斥。白底高清。</p>',
     '<p><strong>说明：</strong>周期框仅为基本胞，并非真实墙面。软壁虚线表示势开始作用的区域；外实线为名义边界，粒子可能短暂越过。图中t均以各模拟实际记录为准，t=300并不等于已证实稳态。</p>']
    for k in allK:
        f=root/'compare_boundaries'/f'K_{k:g}'
        if (f/'comparison_final.png').exists():
            r=f.relative_to(root).as_posix()
            html_lines.append(f'<section><h2>五种边界比较，K={k:g}</h2>')
            for fn in ('comparison.mp4','comparison_final.png','comparison_metrics.png'):
                if (f/fn).exists():
                    html_lines.append(f'<video controls muted playsinline src="{r}/{fn}"></video>' if fn.endswith('mp4') else f'<img src="{r}/{fn}">')
            html_lines.append('</section>')
    for b,cases in allcases.items():
        html_lines.append(f'<section><h2>{html.escape(NAMES[b])}</h2>')
        rel=b
        for fn in ('comparison.mp4','comparison_final.png','comparison_metrics.png'):
            if (root/b/fn).exists():
                html_lines.append(f'<video controls muted playsinline src="{rel}/{fn}"></video>' if fn.endswith('mp4') else f'<img src="{rel}/{fn}">')
        html_lines.append('<div class="grid">')
        for c in cases:
            loc=c['folder'].relative_to(root).as_posix()
            html_lines.append(f'<div><h3>K={c["K"]:g}，当前实际 t={c["t"][-1]:g}</h3>')
            for fn in ('particles.mp4','heading.mp4','particles.gif','heading.gif'):
                if (c['folder']/fn).exists():
                    html_lines.append(f'<video controls muted playsinline loop src="{loc}/{fn}"></video>' if fn.endswith('.mp4') else f'<img src="{loc}/{fn}">')
            for fn in ('snapshots.png','particles_final.png','heading_final.png','sampled_tracks.png','density.png','metrics.png'):
                if (c['folder']/fn).exists():html_lines.append(f'<img loading="lazy" src="{loc}/{fn}">')
            html_lines.append('</div>')
        html_lines.append('</div></section>')
    html_lines.append('</html>')
    (root/'index.html').write_text('\n'.join(html_lines),encoding='utf-8')
    print('OPEN',root/'index.html',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/'N5000_square_runs')
    ap.add_argument('--boundaries',nargs='+',choices=list(NAMES),default=list(NAMES))
    ap.add_argument('--movies',action='store_true');ap.add_argument('--heading-movies',action='store_true')
    ap.add_argument('--max-frames',type=int,default=120)
    a=ap.parse_args();render(a.root,a.boundaries,a.movies,a.heading_movies,a.max_frames)
