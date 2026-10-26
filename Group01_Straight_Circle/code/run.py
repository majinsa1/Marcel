"""Run one case, save from t=0, recheck late-time windows with further data."""
from pathlib import Path
from dataclasses import asdict
from collections import deque
import argparse
import csv
import json
import time
import numpy as np
from model import Params, Simulation
from diagnostics import measure, stationarity

DEFAULT_RUN = dict(mode='fixed', end=60., max_time=4800., window=200.,
                   record_time=200., sample_dt=1., early_time=60.,
                   mean_tol=.04, std_tol=.04, tv_tol=.08, checkpoint_seconds=120., render=True)


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run_case(config, output, resume=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    p = Params(**config.get('params', {})); p.validate()
    unknown=set(config.get('run',{}))-set(DEFAULT_RUN)
    if unknown: raise ValueError(f'Unknown run settings: {sorted(unknown)}')
    run = dict(DEFAULT_RUN); run.update(config.get('run', {}))
    if run['mode'] not in ('fixed', 'steady'):
        raise ValueError('run.mode must be fixed or steady')
    time_keys=('end','max_time','window','record_time','sample_dt','checkpoint_seconds')
    if any(not np.isfinite(run[k]) or run[k]<=0 for k in time_keys):
        raise ValueError('Time settings must be positive')
    for k in ('mean_tol','std_tol','tv_tol','early_time'):
        if not np.isfinite(run[k]) or run[k]<0: raise ValueError(f'{k} must be finite and nonnegative')
    if run['sample_dt'] > run['window']/10:
        raise ValueError('Need >=10 observations per stationarity window')
    if run['mode'] == 'steady' and run['record_time'] < run['window']:
        raise ValueError('Confirmation record_time must be at least one window')
    if run['mode']=='steady':
        for key in ('window','record_time'):
            ratio=run[key]/run['sample_dt']
            if not np.isclose(ratio,round(ratio),atol=1e-8,rtol=0):
                raise ValueError(f'{key} must be an integer multiple of sample_dt')
    cfg_path = output/'config.json'
    if (output/'checkpoint.npz').exists() and not resume:
        raise FileExistsError(f'{output} already has data; use --resume or a new output')
    started = time.perf_counter()
    rows=[]; radial=[]; spatial=[]; angle=[]; early=[]; timeline=[]; late=deque()
    track_t=[]; track_x=[]; checks=[]; candidate=None; timeline_stride=1
    frequency_samples=deque()
    if resume:
        sim = Simulation.restore(output/'checkpoint.npz')
        if asdict(sim.p) != asdict(p):
            # JSON lists vs tuples are equivalent in the probability field.
            if json.dumps(asdict(sim.p), sort_keys=True) != json.dumps(asdict(p), sort_keys=True):
                raise ValueError('Cannot change physical parameters when resuming')
        # IMPORTANT on Windows: close NpzFile before any later os.replace().
        # Keeping this file open locks history.npz and triggers WinError 5.
        with np.load(output/'history.npz', allow_pickle=False) as z:
            rows = json.loads(str(z['rows_json']))
            radial=list(z['radial']); spatial=list(z['spatial']); angle=list(z['angle'])
            track_t=list(z['track_t']); track_x=list(z['track_x']); tracks=z['tracks'].copy()
            for name, dest in [('early', early), ('timeline', timeline), ('late', late)]:
                for t, x, th in zip(z[name+'_t'], z[name+'_x'], z[name+'_theta']):
                    dest.append((float(t), x.copy(), th.copy()))
            meta=json.loads(str(z['meta_json'])); checks=meta['checks']; candidate=meta['candidate']
            timeline_stride=meta['timeline_stride']
            for ft,ca,wa in zip(z['frequency_t'],z['frequency_cont'],z['frequency_wall']):
                frequency_samples.append((float(ft),ca.copy(),wa.copy()))
        oldrun=json.loads(cfg_path.read_text(encoding='utf-8'))['run']
        for key in ('sample_dt','window','record_time','mean_tol','std_tol','tv_tol'):
            if run[key] != oldrun[key]:
                raise ValueError(f'Cannot change {key} when resuming stored diagnostics')
        if abs(rows[-1]['t']-sim.t) > 1e-8:
            raise ValueError('Checkpoint/history mismatch; restore matching files')
    else:
        sim=Simulation(p)
        tracks=np.sort(np.random.default_rng(p.seed+932).choice(p.N, min(32,p.N), replace=False))
    end = run['end'] if run['mode']=='fixed' else run['max_time']
    if resume and end<=sim.t+1e-9:
        raise ValueError('New end/max_time must be later than the stored simulation time')
    cfg_path.write_text(json.dumps({'params': asdict(p), 'run': run}, indent=2), encoding='utf-8')
    dt_obs=run['sample_dt']; status='unconfirmed'; validated_start=None
    def frame():
        return (float(sim.t), sim.x.astype(np.float32).copy(), sim.theta.astype(np.float32).copy())
    def sample():
        nonlocal timeline_stride
        row, rh, sh, ah=measure(sim)
        if sim.t == 0:
            # No displacement measured at t=0. Avoid claiming initial zero speed.
            for key in ('mean_speed','stopped_fraction','P_velocity','C_velocity','C_edge','tangent_order_edge'):
                row[key]=float('nan')
        rows.append(row); radial.append(rh); spatial.append(sh); angle.append(ah)
        f=frame(); late.append(f)
        frequency_samples.append((sim.t,sim.continuous_angle.copy(),sim.wall_angle.copy()))
        while frequency_samples and frequency_samples[0][0] < sim.t-run['record_time']-1e-8:
            frequency_samples.popleft()
        while late and late[0][0] < sim.t-run['record_time']-1e-8:
            late.popleft()
        if sim.t <= run['early_time']+1e-8:
            early.append(f)
        index=len(rows)-1
        if index % timeline_stride == 0:
            timeline.append(f)
        if len(timeline)>300:
            timeline[:]=timeline[::2]; timeline_stride*=2
    def save():
        sim.checkpoint(output/'checkpoint.tmp.npz')
        arrays=dict(rows_json=json.dumps(rows), radial=np.array(radial), spatial=np.array(spatial),
                    angle=np.array(angle), track_t=np.array(track_t), track_x=np.array(track_x), tracks=tracks,
                    omega=sim.omega, continuous_angle=sim.continuous_angle, wall_angle=sim.wall_angle,
                    frequency_t=np.array([a[0] for a in frequency_samples]),
                    frequency_cont=np.array([a[1] for a in frequency_samples]),
                    frequency_wall=np.array([a[2] for a in frequency_samples]),
                    meta_json=json.dumps(dict(checks=checks,candidate=candidate,timeline_stride=timeline_stride)))
        for name, seq in [('early',early),('timeline',timeline),('late',late)]:
            arrays[name+'_t']=np.array([s[0] for s in seq])
            arrays[name+'_x']=np.array([s[1] for s in seq])
            arrays[name+'_theta']=np.array([s[2] for s in seq])
        np.savez_compressed(output/'history.tmp.npz', **arrays)
        # Replace each completed file atomically; reject a mismatched pair on restart.
        (output/'checkpoint.tmp.npz').replace(output/'checkpoint.npz')
        (output/'history.tmp.npz').replace(output/'history.npz')
        write_csv(output/'observables.csv',rows)
        (output/'stationarity_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    if not rows:
        sample(); track_t.append(0.); track_x.append(sim.x[tracks].astype(np.float32).copy())
        save()  # a consistent t=0 recovery point is always available
    # Tracks sample more densely than frames, resolving the fastest intrinsic rotation.
    track_dt=min(.2, dt_obs, .12/max(abs(sim.omega).max(initial=0)+abs(p.K),1e-12))
    last_log=time.perf_counter(); last_save=sim.t; last_save_wall=time.perf_counter()
    try:
        while sim.t < end-1e-9:
            next_obs=min((np.floor((sim.t+1e-8)/dt_obs)+1)*dt_obs,end)
            while sim.t < next_obs-1e-9:
                target=min(sim.t+track_dt,next_obs)
                sim.advance_to(target)
                track_t.append(sim.t); track_x.append(sim.x[tracks].astype(np.float32).copy())
            sample()
            if run['mode']=='steady' and sim.t >= 3*run['window']-1e-8:
                at_window=abs(sim.t/run['window']-round(sim.t/run['window'])) < 1e-7
                at_confirm=candidate is not None and sim.t >= candidate+run['record_time']-1e-8
                if at_window or at_confirm:
                    check=stationarity(rows,radial,spatial,run['window'],run['mean_tol'],run['std_tol'],run['tv_tol'])
                    checks.append(check)
                    if candidate is None and check['passed']:
                        candidate=sim.t
                    elif candidate is not None:
                        if not check['passed']:
                            candidate=None
                        elif at_confirm:
                            status='finite_window_confirmed'; validated_start=candidate; break
            if time.perf_counter()-last_log>25:
                print(f'{output.name}: t={sim.t:.1f}, steps={sim.steps}, candidate={candidate}',flush=True)
                last_log=time.perf_counter()
            if sim.t-last_save >= max(20.,run['window']) or time.perf_counter()-last_save_wall >= run['checkpoint_seconds']:
                save(); last_save=sim.t; last_save_wall=time.perf_counter()
    except KeyboardInterrupt:
        # All writes use the last completed observation state, ensuring consistent resume.
        # On interrupt mid-step, save a standalone emergency state, not mismatched history.
        sim.checkpoint(output/'interrupted_state.npz')
        raise
    if not timeline or abs(timeline[-1][0]-sim.t)>1e-8:
        timeline.append(frame())
    save()
    # Keep continuous rotation and wall jumps separate in both full-run and late windows.
    ft,ca,wa=frequency_samples[0]; duration=max(sim.t-ft,1e-12)
    late_cont=(sim.continuous_angle-ca)/duration; late_wall=(sim.wall_angle-wa)/duration
    per_particle=dict(id=np.arange(p.N),omega_intrinsic=sim.omega,
        angular_rate_continuous_total=sim.continuous_angle/max(sim.t,1e-12),
        angular_rate_wall_total=sim.wall_angle/max(sim.t,1e-12),
        angular_rate_continuous_late=late_cont,angular_rate_wall_late=late_wall)
    with open(output/'particle_frequencies.csv','w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(per_particle)
        writer.writerows(zip(*per_particle.values()))
    report=dict(status=status,record_start=float(late[0][0]),record_end=sim.t,
        confirmed_start=validated_start,N=p.N,steps=sim.steps,minimum_dt=sim.smallest_dt,
        elapsed_seconds=time.perf_counter()-started,omega_realized_mean=float(sim.omega.mean()),
        omega_realized_std=float(sim.omega.std()),omega_min=float(sim.omega.min()),
        omega_max=float(sim.omega.max()),
        model_type='point_alignment' if p.repulsion_k==0 else 'alignment_with_soft_repulsion',
        nominal_packing_fraction=float(p.N*np.pi*p.diameter**2/(4*p.side**2)) if p.repulsion_k else None,
        frequency_window_start=ft,frequency_window_end=sim.t,
        late_continuous_frequency_std=float(np.std(late_cont)),
        warning='Finite-window screening is not a proof of asymptotic steady state. Fixed runs are unconfirmed.')
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # One row per particle makes the actual quenched sample easy to inspect.
    np.savetxt(output/'omega_samples.csv',np.column_stack((np.arange(p.N),sim.omega)),
               delimiter=',',header='particle_id,omega',comments='',fmt=['%d','%.17g'])
    if run['render']:
        from render import render_case
        render_case(output)
    print(json.dumps(dict(case=str(output),**report)),flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/straight.json'); ap.add_argument('--out',default='results/custom')
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--no-render',action='store_true')
    args=ap.parse_args(); cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    from configuration import resolve_config
    cfg=resolve_config(cfg)
    if args.no_render: cfg.setdefault('run',{})['render']=False
    run_case(cfg,args.out,args.resume)
