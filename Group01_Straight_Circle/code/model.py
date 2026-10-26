"""Baseline: pure-population particles with local alignment and fixed chirality.

Lengths are in alignment-radius units, speeds in v0 units. Angular frequencies
are quenched particle properties. No temporal noise or artificial wall torque.
The hard-wall law is a specified hybrid rule; see README for its interpretation.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Params:
    N: int = 5000
    side: float = 50.0
    v0: float = 1.0
    K: float = 1.5
    align_radius: float = 1.0
    include_self: bool = True
    diameter: float = 0.5
    repulsion_k: float = 0.0  # OPTIONAL extension; baseline has no mechanical force
    mobility: float = 1.0
    omega_mean: float = 0.0
    omega_width: float = 0.0
    omega_law: str = 'uniform'  # uniform half-width; normal standard deviation
    allow_opposite_sign: bool = False
    shape: str = 'circle'
    boundary: str = 'specular'
    # left, right, bottom, top. Draw only on a NEW contact, retain sliding state.
    wall_p_spec: tuple = (0.5, 0.5, 0.5, 0.5)
    wall_width: float = 2.0
    wall_k: float = 4.0
    edge_width: float = 2.0
    dt: float = 0.02  # maximum step; contact stiffness can reduce this
    step_scale: float = 1.0  # scales the adaptive bound, for meaningful refinement
    seed: int = 20261022
    init: str = 'random'  # random centers, or random nonoverlapping centers

    @property
    def radius(self):
        return self.side / math.sqrt(math.pi)

    @property
    def half(self):
        return self.side / 2

    def validate(self):
        for k, v in asdict(self).items():
            if isinstance(v, (float, int)) and not np.isfinite(v):
                raise ValueError(f'{k} must be finite')
        if not isinstance(self.N, int) or self.N < 1:
            raise ValueError('N must be a positive integer')
        if min(self.side, self.v0, self.align_radius, self.diameter,
               self.mobility, self.dt) <= 0:
            raise ValueError('Lengths, speed, mobility and dt must be positive')
        if not 0 < self.step_scale <= 1:
            raise ValueError('step_scale must be in (0,1]')
        if self.repulsion_k < 0 or self.wall_k <= 0 or self.omega_width < 0:
            raise ValueError('Invalid force or frequency parameters')
        if self.shape not in ('circle', 'square'):
            raise ValueError('Unknown shape')
        if self.boundary not in ('specular', 'sliding', 'mixed', 'soft', 'periodic'):
            raise ValueError('Unknown boundary')
        if self.boundary in ('mixed', 'periodic') and self.shape != 'square':
            raise ValueError('Mixed / periodic controls require a square')
        if not 0 < self.edge_width < self.half or not 0 < self.wall_width < self.half:
            raise ValueError('Invalid layer width')
        if len(self.wall_p_spec) != 4 or not all(0 <= p <= 1 for p in self.wall_p_spec):
            raise ValueError('wall_p_spec order: left,right,bottom,top; range [0,1]')
        if self.omega_law not in ('uniform', 'normal', 'constant'):
            raise ValueError('Unknown omega_law')
        if self.omega_law == 'constant' and self.omega_width != 0:
            raise ValueError('constant law requires omega_width=0')
        if not self.allow_opposite_sign:
            if self.omega_law == 'normal' and self.omega_width > 0:
                raise ValueError('Unbounded normal may cross zero; explicitly allow it')
            if self.omega_width > abs(self.omega_mean):
                raise ValueError('Distribution crosses zero: enable allow_opposite_sign')
        if self.init not in ('random', 'nonoverlap'):
            raise ValueError('Unknown initialization')
        cutoff=max(self.align_radius, self.diameter) if self.repulsion_k else self.align_radius
        if self.boundary == 'periodic' and cutoff >= self.half:
            raise ValueError('Periodic cutoff must be smaller than side/2')


def initialise(p):
    p.validate()
    posrng, angrng, omrng, wallrng = [np.random.default_rng(s)
        for s in np.random.SeedSequence(p.seed).spawn(4)]
    def points(n):
        u = posrng.random((n, 2))
        if p.shape == 'square':
            return p.side * (u - 0.5)
        r, a = p.radius * np.sqrt(u[:, 0]), 2*np.pi*u[:, 1]
        return np.column_stack((r*np.cos(a), r*np.sin(a)))
    if p.init == 'random':
        x = points(p.N)
    else:
        # Random sequential placement. Fail explicitly at too high a density.
        x = np.empty((p.N, 2)); cells = {}; attempts = 0; n = 0
        if p.boundary == 'periodic':
            raise ValueError('nonoverlap initialization not implemented for periodic; use random')
        while n < p.N and attempts < 1000*p.N:
            q = points(1)[0]; attempts += 1
            cell = tuple(np.floor(q/p.diameter).astype(int))
            near = [j for a in range(-1, 2) for b in range(-1, 2)
                    for j in cells.get((cell[0]+a, cell[1]+b), [])]
            if near and np.any(np.linalg.norm(x[near]-q, axis=1) < p.diameter):
                continue
            x[n] = q; cells.setdefault(cell, []).append(n); n += 1
        if n < p.N:
            raise RuntimeError('Nonoverlap placement failed; lower density or use random')
    theta = angrng.uniform(-np.pi, np.pi, p.N)
    if p.omega_law == 'normal':
        z = omrng.normal(size=p.N)
    else:
        z = omrng.uniform(-1, 1, p.N)
    omega = p.omega_mean + (-1 if p.omega_mean < 0 else 1)*p.omega_width*z
    return x, theta, omega, wallrng


def pair_batches(tree, cutoff, max_pairs=1_000_000):
    """Bound the memory used by dense point-particle clusters.

    Exact neighborhoods; no neighbor cap, downsampling or grid approximation.
    """
    n=tree.n
    estimate=(int(tree.count_neighbors(tree,cutoff))-n)//2
    if estimate <= max_pairs:
        yield tree.query_pairs(cutoff,output_type='ndarray')
    else:
        for start in range(0,n,64):
            neighbors=tree.query_ball_point(tree.data[start:start+64],cutoff)
            left=[]; right=[]
            for offset,js in enumerate(neighbors):
                i=start+offset; j=np.asarray(js,dtype=np.int64); j=j[j>i]
                if len(j):
                    left.append(np.full(len(j),i,dtype=np.int64)); right.append(j)
            if left:
                yield np.column_stack((np.concatenate(left),np.concatenate(right)))


def interactions(x, theta, p):
    """One undirected neighbor search; pair forces summed, never normalized.

    Alignment is neighbor-normalized (self contributes 0 to numerator).
    Forces obey F_ij = -F_ji, including the deterministic coincident-point rule.
    """
    cutoff = max(p.align_radius, p.diameter) if p.repulsion_k else p.align_radius
    if p.boundary == 'periodic':
        tree = cKDTree(np.remainder(x+p.half, p.side), boxsize=p.side)
    else:
        tree = cKDTree(x)
    force = np.zeros_like(x); degree = np.zeros(p.N, dtype=int)
    contact_degree = np.zeros(p.N, dtype=int); torque = np.zeros(p.N)
    energy = 0.0; max_overlap = 0.0; cos_sum = 0.0; edge_count = 0
    for pairs in pair_batches(tree,cutoff):
        if not len(pairs):
            continue
        i, j = pairs.T
        dx = x[i]-x[j]
        if p.boundary == 'periodic':
            dx -= p.side*np.rint(dx/p.side)
        r = np.linalg.norm(dx, axis=1)
        mask = r < p.align_radius
        a, b = i[mask], j[mask]
        sine = np.sin(theta[b]-theta[a])
        torque += (np.bincount(a, weights=sine, minlength=p.N)
                  - np.bincount(b, weights=sine, minlength=p.N))
        degree += np.bincount(a, minlength=p.N)+np.bincount(b, minlength=p.N)
        cos_sum += float(np.cos(theta[b]-theta[a]).sum()); edge_count += len(a)
        if not p.repulsion_k:
            continue
        mask = r < p.diameter
        a, b = i[mask], j[mask]; d = dx[mask].copy(); rr = r[mask]
        contact_degree += np.bincount(a, minlength=p.N)+np.bincount(b, minlength=p.N)
        overlap = p.diameter-rr
        max_overlap = max(max_overlap,float(overlap.max(initial=0)/p.diameter))
        # At exactly coincident centers a central-force direction is undefined.
        # A reproducible pair direction resolves only this degeneracy.
        zero = rr < 1e-14
        if zero.any():
            phi = np.remainder((a[zero]+1)*2.399963229728653 + (b[zero]+1)*1.61803398875, 2*np.pi)
            d[zero] = np.column_stack((np.cos(phi), np.sin(phi)))
        unit = d / np.where(zero, 1, rr)[:, None]
        f = p.repulsion_k*overlap[:, None]*unit
        for axis in range(2):
            force[:, axis] += (np.bincount(a, weights=f[:, axis], minlength=p.N)
                              - np.bincount(b, weights=f[:, axis], minlength=p.N))
        energy += float(0.5*p.repulsion_k*np.sum(overlap**2))
    denom = degree+int(p.include_self)
    align = p.K*np.divide(torque, denom, out=np.zeros(p.N), where=denom > 0)
    return align, force, degree, contact_degree, energy, max_overlap, cos_sum, edge_count


def wall_force(x, p):
    if p.boundary != 'soft':
        return np.zeros_like(x)
    if p.shape == 'square':
        q = np.maximum(np.abs(x)-(p.half-p.wall_width), 0)
        return -p.wall_k*q*np.sign(x)
    r = np.linalg.norm(x, axis=1)
    q = np.maximum(r-(p.radius-p.wall_width), 0)
    return -p.wall_k*q[:, None]*np.divide(x, r[:, None], out=np.zeros_like(x), where=r[:, None] > 0)


def wrap_angle(a):
    return (a+np.pi) % (2*np.pi)-np.pi


def circle_billiard(x, heading, distance, radius):
    """Exact frozen-heading disk billiard, including many grazing chords.

    This fast path is used by the point-particle baseline. It avoids a fixed
    collision-count cutoff when particles organize close to the circular wall.
    """
    R=radius; d=distance
    out=x+d*heading; eo=heading.copy(); jump=np.zeros(len(x))
    dot=np.sum(x*heading,axis=1)
    first=np.maximum(0.,-dot+np.sqrt(np.maximum(0.,dot*dot+R*R-np.sum(x*x,axis=1))))
    hit=(first <= d+1e-12)&(np.linalg.norm(out,axis=1)>=R-1e-12)
    ids=np.flatnonzero(hit)
    if not len(ids): return out,eo,jump
    h=x[ids]+np.minimum(first[ids],d)[:,None]*heading[ids]
    h*= (R/np.linalg.norm(h,axis=1))[:,None]; n=h/R
    rem=np.maximum(0.,d-first[ids])
    reflected=heading[ids]-2*np.sum(heading[ids]*n,axis=1)[:,None]*n
    jump[ids]=wrap_angle(np.arctan2(reflected[:,1],reflected[:,0])-np.arctan2(heading[ids,1],heading[ids,0]))
    chord=np.maximum(0.,-2*np.sum(h*reflected,axis=1))
    regular=chord>1e-9*R
    count=np.zeros(len(ids),dtype=np.int64)
    count[regular]=np.floor(rem[regular]/chord[regular]).astype(np.int64)
    nxt=h+chord[:,None]*reflected
    turn=np.arctan2(h[:,0]*nxt[:,1]-h[:,1]*nxt[:,0],np.sum(h*nxt,axis=1))
    angle=np.remainder(count*turn,2*np.pi); c,s=np.cos(angle),np.sin(angle)
    wall=np.column_stack((c*h[:,0]-s*h[:,1],s*h[:,0]+c*h[:,1]))
    direction=np.column_stack((c*reflected[:,0]-s*reflected[:,1],s*reflected[:,0]+c*reflected[:,1]))
    rest=np.maximum(0.,rem-count*chord)
    out[ids]=wall+rest[:,None]*direction; eo[ids]=direction
    jump[ids]+=count*turn
    if (~regular).any():
        g=ids[~regular]; hg=h[~regular]; rg=reflected[~regular]
        sign=np.sign(hg[:,0]*rg[:,1]-hg[:,1]*rg[:,0]); sign[sign==0]=1
        phi=np.arctan2(hg[:,1],hg[:,0])+sign*rem[~regular]/R
        out[g]=R*np.column_stack((np.cos(phi),np.sin(phi)))
        eo[g]=sign[:,None]*np.column_stack((-np.sin(phi),np.cos(phi)))
        jump[g]+=sign*rem[~regular]/R
    if np.max(np.linalg.norm(out,axis=1))>R+1e-8:
        raise RuntimeError('Circle billiard geometric error')
    return out,eo,jump


def transport(x, theta, velocity, dt, p, rng, contacts):
    """Freeze total overdamped velocity in a substep and transport to walls.

    Reflection applies the same normal reflection to total velocity and heading.
    Sliding removes outward TOTAL velocity, does not rotate heading, and is not
    renormalized. All hard walls constrain particle centers (not disk surfaces).
    Returns new position, heading, persistent contacts, collision-angle jumps.
    """
    if p.boundary == 'soft':
        return x+dt*velocity, theta.copy(), contacts.copy(), np.zeros(p.N)
    if p.boundary == 'periodic':
        return (x+dt*velocity+p.half) % p.side-p.half, theta.copy(), contacts.copy(), np.zeros(p.N)
    e = np.column_stack((np.cos(theta), np.sin(theta)))
    if (p.boundary=='specular' and p.shape=='circle' and p.repulsion_k==0
            and np.allclose(velocity,p.v0*e,rtol=1e-12,atol=1e-12)):
        out,eo,jump=circle_billiard(x,e,p.v0*dt,p.radius)
        return out,np.arctan2(eo[:,1],eo[:,0]),contacts.copy(),jump
    out = x.copy(); eo = e.copy(); jump = np.zeros(p.N); state = contacts.copy()
    if p.shape == 'square':
        for axis in range(2):
            comp = velocity[:, axis]
            if p.boundary == 'specular':
                raw = x[:, axis]+dt*comp
                q = (raw+p.half)/p.side; cell = np.floor(q).astype(np.int64)
                odd = cell % 2 != 0; frac = q-cell
                out[:, axis] = np.where(odd, p.half-frac*p.side, -p.half+frac*p.side)
                eo[odd, axis] *= -1
                state[:, axis] = 0
            else:
                active = ((state[:, axis] != 0)
                          & (np.abs(x[:, axis]-state[:, axis]*p.half) < 1e-9)
                          & (comp*state[:, axis] >= 0))
                state[~active, axis] = 0
                raw = x[:, axis]+dt*comp; raw[active] = x[active, axis]
                hit = ((raw >= p.half) | (raw <= -p.half)) & ~active & (abs(comp) > 1e-14)
                sign = np.where(raw >= p.half, 1, -1)
                refl = np.zeros(p.N, dtype=bool); ids = np.flatnonzero(hit)
                if p.boundary == 'mixed' and len(ids):
                    wall_id = 2*axis+(sign[ids] > 0).astype(int)
                    prob = np.asarray(p.wall_p_spec)[wall_id]
                    refl[ids] = rng.random(len(ids)) < prob
                slip = hit & ~refl
                raw[refl] = 2*sign[refl]*p.half-raw[refl]
                eo[refl, axis] *= -1; state[refl, axis] = 0
                raw[slip] = sign[slip]*p.half; state[slip, axis] = sign[slip]
                out[:, axis] = raw
        newtheta = np.arctan2(eo[:, 1], eo[:, 0])
        jump = wrap_angle(newtheta-theta)
    else:
        R = p.radius
        speed = np.linalg.norm(velocity, axis=1)
        u = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity), where=speed[:, None] > 0)
        remaining = speed*dt
        # Usually zero or one collision; grazing rays can have many short chords.
        for iteration in range(128):
            dot = np.sum(out*u, axis=1)
            disc = np.maximum(0, dot*dot+R*R-np.sum(out*out, axis=1))
            distance = np.maximum(0, -dot+np.sqrt(disc))
            hit = (remaining > 1e-13) & (distance <= remaining+1e-12)
            travel = np.where(hit, np.minimum(distance, remaining), remaining)
            out += travel[:, None]*u; remaining -= travel
            ids = np.flatnonzero(hit)
            if not len(ids):
                break
            n = out[ids]/np.linalg.norm(out[ids], axis=1)[:, None]
            out[ids] = R*n
            if p.boundary == 'sliding':
                phi = np.arctan2(n[:, 1], n[:, 0])
                direction = np.arctan2(u[ids, 1], u[ids, 0])
                delta = wrap_angle(direction-phi)
                delta_new = 2*np.arctan(np.tan(delta/2)*np.exp(-remaining[ids]/R))
                phi_new = direction-delta_new
                out[ids] = R*np.column_stack((np.cos(phi_new), np.sin(phi_new)))
                remaining[ids] = 0
            else:
                oldangle = np.arctan2(eo[ids, 1], eo[ids, 0])
                eo[ids] -= 2*np.sum(eo[ids]*n, axis=1)[:, None]*n
                jump[ids] += wrap_angle(np.arctan2(eo[ids, 1], eo[ids, 0])-oldangle)
                un = np.sum(u[ids]*n, axis=1)
                u[ids] -= 2*un[:, None]*n
                # Only the machine-precision tangency limit is regularized.
                tangent = np.abs(un) < 1e-10
                if tangent.any():
                    g = ids[tangent]; ng = n[tangent]
                    sg = np.sign(ng[:, 0]*u[g, 1]-ng[:, 1]*u[g, 0]); sg[sg == 0] = 1
                    rot = sg*remaining[g]/R; c, s = np.cos(rot), np.sin(rot)
                    out[g] = R*np.column_stack((c*ng[:, 0]-s*ng[:, 1], s*ng[:, 0]+c*ng[:, 1]))
                    eg = eo[g].copy()
                    eo[g] = np.column_stack((c*eg[:, 0]-s*eg[:, 1], s*eg[:, 0]+c*eg[:, 1]))
                    jump[g] += rot; remaining[g] = 0
            if np.all(remaining < 1e-13):
                break
        if np.max(remaining) > 1e-10:
            raise RuntimeError('Too many grazing reflections; reduce dt')
        newtheta = np.arctan2(eo[:, 1], eo[:, 0])
    if p.shape == 'circle':
        outside = np.max(np.linalg.norm(out, axis=1))-p.radius
    else:
        outside = np.max(np.abs(out))-p.half
    if outside > 1e-8:
        raise RuntimeError('Hard boundary violation; reduce dt')
    return out, newtheta, state, jump


class Simulation:
    def __init__(self, p):
        self.p = p
        self.x, self.theta, self.omega, self.rng = initialise(p)
        self.contacts = np.zeros((p.N, 2), np.int8)
        self.t = 0.0; self.steps = 0; self.smallest_dt = p.dt
        self.continuous_angle = np.zeros(p.N); self.wall_angle = np.zeros(p.N)
        self.velocity = np.zeros_like(self.x)

    def advance_to(self, target):
        p = self.p
        # The last difference to a requested output time can be as small as
        # 1e-11 after thousands of floating-point increments.  That is not
        # physical stiffness.  Snap only such a negligible CLOCK residual to
        # the requested output time; do not hide a truly small physical dt.
        clock_tol = max(1e-10, 32*np.finfo(float).eps*max(1.0, abs(target)))
        if target < self.t-clock_tol:
            raise ValueError(f'Cannot advance backwards: t={self.t}, target={target}')
        while self.t < target-clock_tol:
            if p.K == 0 and p.repulsion_k == 0:
                rate=self.omega.copy(); force=np.zeros_like(self.x)
                contact_degree=np.zeros(p.N,dtype=int)
            else:
                inter = interactions(self.x, self.theta, p)
                rate, force, _, contact_degree = inter[:4]
                rate = self.omega+rate
            wf = wall_force(self.x, p)
            # Guard angular, force-relaxation and displacement scales separately.
            vmax = p.v0+p.mobility*np.max(np.linalg.norm(force+wf, axis=1))
            stiffness = p.mobility*(2*p.repulsion_k*contact_degree.max(initial=0)
                                    + (p.wall_k if p.boundary == 'soft' else 0))
            length_scale=min(p.diameter,p.align_radius) if p.repulsion_k else p.align_radius
            bound = min(p.dt, .04/max(np.max(np.abs(rate)), 1e-12),
                        .15/max(stiffness, 1e-12), .10*length_scale/max(vmax, 1e-12))
            dt = min(p.step_scale*bound, target-self.t)
            if dt < 1e-10:
                raise RuntimeError(
                    f'Adaptive physical dt too small at t={self.t:.16g}: '
                    f'bound={bound:.4g}, remaining={target-self.t:.4g}, '
                    f'max_rate={np.max(np.abs(rate)):.4g}, '
                    f'vmax={vmax:.4g}, stiffness={stiffness:.4g}. '
                    'Inspect physical forces and time-step requirements.')
            theta = wrap_angle(self.theta+dt*rate)
            v = p.v0*np.column_stack((np.cos(theta), np.sin(theta)))+p.mobility*(force+wf)
            newx, newtheta, contacts, jump = transport(self.x, theta, v, dt, p, self.rng, self.contacts)
            dx = newx-self.x
            if p.boundary == 'periodic':
                dx -= p.side*np.rint(dx/p.side)
            self.velocity = dx/dt  # final-substep displacement velocity, not movie-frame speed
            self.x, self.theta, self.contacts = newx, newtheta, contacts
            self.continuous_angle += dt*rate; self.wall_angle += jump
            self.t += dt; self.steps += 1; self.smallest_dt = min(self.smallest_dt, dt)
        # A discrepancy below clock_tol is floating-point bookkeeping only.
        # Do not take an unphysical microstep merely to land on a decimal time.
        self.t = float(target)

    def checkpoint(self, path):
        import json
        np.savez_compressed(path, x=self.x, theta=self.theta, omega=self.omega,
            contacts=self.contacts, t=self.t, steps=self.steps, smallest_dt=self.smallest_dt,
            continuous_angle=self.continuous_angle, wall_angle=self.wall_angle,
            velocity=self.velocity, rng_json=json.dumps(self.rng.bit_generator.state),
            params_json=json.dumps(asdict(self.p)))

    @classmethod
    def restore(cls, path):
        import json
        # Close checkpoint file promptly so Windows can replace it on save.
        with np.load(path, allow_pickle=False) as z:
            obj = cls(Params(**json.loads(str(z['params_json']))))
            for k in ('x', 'theta', 'omega', 'contacts', 'continuous_angle', 'wall_angle', 'velocity'):
                setattr(obj, k, z[k].copy())
            obj.t = float(z['t']); obj.steps = int(z['steps']); obj.smallest_dt = float(z['smallest_dt'])
            obj.rng.bit_generator.state = json.loads(str(z['rng_json']))
        return obj
