"""Distinguish polar order, circulating flow, wall residence and synchronization."""
import numpy as np
from model import interactions


def measure(sim):
    p, x, theta = sim.p, sim.x, sim.theta
    e = np.column_stack((np.cos(theta), np.sin(theta)))
    inter = interactions(x, theta, p)
    align, force, degree, contacts, energy, overlap, cos_sum, pairs = inter
    r = np.linalg.norm(x, axis=1)
    if p.shape == 'circle':
        distance = p.radius-r
        tangent = np.divide(np.column_stack((-x[:, 1], x[:, 0])), r[:, None],
                            out=np.zeros_like(x), where=r[:, None] > 1e-12)
        area_fraction = 1-(1-p.edge_width/p.radius)**2
        q = r/p.radius
    else:
        distance = p.half-np.max(np.abs(x), axis=1)
        normal = np.zeros_like(x); axis = np.argmax(np.abs(x), axis=1)
        normal[np.arange(p.N), axis] = np.sign(x[np.arange(p.N), axis])
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        area_fraction = 1-(1-2*p.edge_width/p.side)**2
        q = np.max(np.abs(x), axis=1)/p.half
    inside = distance >= -1e-8
    edge = inside & (distance <= p.edge_width)
    speed = np.linalg.norm(sim.velocity, axis=1)
    vhat = np.divide(sim.velocity, speed[:, None], out=np.zeros_like(x), where=speed[:, None] > 1e-8*p.v0)
    vt = np.sum(sim.velocity*tangent, axis=1)/p.v0
    et = np.sum(e*tangent, axis=1)
    movable = speed > 1e-3*p.v0
    nematic = 2*np.sum(vhat*tangent, axis=1)**2-1
    P = float(np.linalg.norm(e.mean(axis=0)))
    row = dict(t=sim.t, P_heading=P,
        P_velocity=float(np.linalg.norm(vhat[movable].mean(axis=0))) if movable.any() else 0.,
        local_alignment=cos_sum/pairs if pairs else np.nan,
        mean_speed=float(speed.mean()/p.v0), stopped_fraction=float(np.mean(~movable)),
        edge_fraction=float(edge.mean()), enrichment=float(edge.mean()/area_fraction),
        C_velocity=float(vt.mean()), C_heading=float(et.mean()),
        C_edge=float(vt[edge].mean()) if edge.any() else np.nan,
        tangent_order_edge=float(nematic[edge & movable].mean()) if (edge & movable).any() else np.nan,
        mean_degree=float(degree.mean()), contact_fraction=float(np.mean(contacts > 0)),
        repulsion_energy_per_particle=energy/p.N, max_overlap_fraction=overlap,
        mean_force=float(np.linalg.norm(force, axis=1).mean()),
        continuous_rate_mean=float(np.mean(sim.omega+align)),
        continuous_rate_std=float(np.std(sim.omega+align)),
        outside_fraction=float(np.mean(~inside)))
    if p.boundary == 'periodic':
        for key in ('edge_fraction', 'enrichment', 'C_velocity', 'C_heading', 'C_edge',
                    'tangent_order_edge', 'outside_fraction'):
            row[key] = np.nan
    if p.repulsion_k == 0:
        # Point-particle baseline: diameter and contact/overlap are not physical inputs.
        row['contact_fraction']=np.nan
        row['max_overlap_fraction']=np.nan
    # Probabilities used for stationarity; radial density later divide by shell area.
    radial = np.r_[np.histogram(q, bins=np.linspace(0, 1, 25))[0], np.sum(q > 1)]/p.N
    lim = p.radius if p.shape == 'circle' else p.half
    spatial, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=12, range=[[-lim, lim]]*2)
    spatial = np.r_[spatial.ravel(), p.N-spatial.sum()]/p.N
    relative = np.arctan2(e[:, 0]*tangent[:, 1]-e[:, 1]*tangent[:, 0], et)
    angle = np.histogram(relative[edge], bins=np.linspace(-np.pi, np.pi, 25))[0].astype(float)
    angle /= max(1, edge.sum())
    return row, radial, spatial, angle


def stationarity(rows, radial, spatial, window=200., mean_tol=.04,
                 std_tol=.04, tv_tol=.08):
    """Finite-window screen, NOT a proof of asymptotic stationarity.

    Three nonoverlapping windows. Compare means, standard deviations, slopes,
    radial and spatial averaged distributions. Further data over at least one
    later window are required; rolling rechecks overlap the earlier screen.
    Slow cycles can fail this conservative screen; it does not auto-classify them.
    """
    t = np.array([r['t'] for r in rows]); end = t[-1]
    if end-t[0] < 3*window-1e-8:
        return {'passed': False, 'reason': 'insufficient_time', 'end': float(end)}
    masks = [(t > end-(3-i)*window+1e-8) & (t <= end-(2-i)*window+1e-8) for i in range(3)]
    if any(m.sum() < 10 for m in masks):
        return {'passed': False, 'reason': 'too_few_samples', 'end': float(end)}
    report = {}; passed = True
    keys = ('P_heading', 'local_alignment', 'mean_speed', 'stopped_fraction',
            'edge_fraction', 'C_velocity', 'contact_fraction', 'continuous_rate_std')
    for key in keys:
        y = np.array([r[key] for r in rows]); stats = []
        selected=np.logical_or.reduce(masks)
        if not np.isfinite(y[selected]).any():
            continue
        for m in masks:
            good = m & np.isfinite(y)
            if good.sum() < 10:
                return {'passed': False, 'reason': 'insufficient_finite_observations', 'key': key}
            stats.append([float(y[good].mean()), float(y[good].std()),
                          float(np.polyfit(t[good]-t[good].mean(), y[good], 1)[0]*window)])
        a = np.array(stats)
        ok = bool(np.ptp(a[:, 0]) <= mean_tol and np.ptp(a[:, 1]) <= std_tol
                  and np.max(np.abs(a[:, 2])) <= 2*mean_tol)
        report[key] = {'mean_std_drift': stats, 'passed': ok}; passed &= ok
    for key, hist in [('radial_distribution', radial), ('spatial_distribution', spatial)]:
        av = [np.array(hist)[m].mean(axis=0) for m in masks]
        tv = [float(np.sum(np.abs(av[i]-av[j]))/2) for i in range(3) for j in range(i+1, 3)]
        ok = max(tv) <= tv_tol; report[key] = {'TV_distances': tv, 'passed': ok}; passed &= ok
    return {'passed': bool(passed), 'end': float(end), 'window': window,
            'mean_tol': mean_tol, 'std_tol': std_tol, 'tv_tol': tv_tol, 'checks': report}
