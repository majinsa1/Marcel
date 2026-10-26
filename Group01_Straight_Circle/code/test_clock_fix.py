"""Quick numeric regression test for the Group01 time-step bug.
Run from code: python test_clock_fix.py
"""
import math
import numpy as np
from model import Params, Simulation


def test_case(K):
    p = Params(N=4, K=K, omega_mean=0.0, omega_width=0.0,
               omega_law='constant', repulsion_k=0.0, dt=0.02,
               boundary='specular', shape='circle')
    sim = Simulation(p)
    while sim.t < 300 - 1e-9:
        next_obs = min((math.floor((sim.t + 1e-8) / 1.0) + 1.0), 300.0)
        while sim.t < next_obs - 1e-9:
            sim.advance_to(min(sim.t + 0.08, next_obs))
        assert abs(sim.t-next_obs)<1e-8, (sim.t, next_obs)
        assert np.isfinite(sim.x).all() and np.isfinite(sim.theta).all()
        assert np.linalg.norm(sim.x,axis=1).max()<=p.radius+1e-7
    assert abs(sim.t-300.0)<1e-8
    assert sim.smallest_dt > 1e-8, sim.smallest_dt
    print(f'PASS: K={K:g}  t={sim.t:.1f}  steps={sim.steps}  min_dt={sim.smallest_dt:.8f}')

if __name__ == '__main__':
    test_case(0.0)
    test_case(1.5)
    print('Clock endpoint handling passed both tests.')
