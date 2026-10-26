"""Focused physical and reproducibility checks for experiment group A."""
from dataclasses import replace
import numpy as np
from model import Params, Simulation, initialise, interactions, circle_billiard


def test():
    count=0
    def ok(label, cond):
        nonlocal count
        if not cond:raise AssertionError(label)
        count+=1;print('PASS:',label)
    p=Params(N=35,K=1.5,side=20,shape='circle',boundary='specular',
             omega_mean=0,omega_width=0,omega_law='constant',repulsion_k=0)
    x,t,omega,_=initialise(p)
    x2,t2,omega2,_=initialise(replace(p,K=0))
    ok('Same seed gives identical initial positions and headings for every K',
       np.array_equal(x,x2) and np.array_equal(t,t2))
    ok('Every particle has omega=0',np.all(omega==0) and np.array_equal(omega,omega2))
    ok('Uniform random initialization stays in disk',np.all(np.linalg.norm(x,axis=1)<=p.radius))
    a,force,degree,*_=interactions(x,t,p)
    ok('No mechanical forces',np.array_equal(force,np.zeros_like(force)))
    ref=np.zeros(p.N)
    for i in range(p.N):
        close=np.linalg.norm(x-x[i],axis=1)<p.align_radius
        close[i]=False
        ref[i]=p.K*np.sin(t[close]-t[i]).sum()/(close.sum()+1)
    ok('KDTree local alignment equals direct sine-neighbor sum, denominator n+1',np.allclose(a,ref))
    R=p.radius
    start=np.array([[R-.1,0.],[0.,0.]])
    e=np.array([[1.,0.],[0.,1.]])
    out,direction,_=circle_billiard(start,e,.3,R)
    ok('Mirror collision preserves remaining traveling distance',np.allclose(out[0],[R-.2,0],atol=1e-9))
    ok('Mirror collision changes heading, not speed',np.allclose(direction[0],[-1,0]))
    ok('No contact means uniform straight transport',np.allclose(out[1],[0,.3]) and np.allclose(direction[1],[0,1]))
    sim=Simulation(p);sim.advance_to(3.)
    ok('All simulated particles remain inside circular wall',np.max(np.linalg.norm(sim.x,axis=1))<R+1e-9)
    ok('Simulation is finite',np.isfinite(sim.x).all() and np.isfinite(sim.theta).all())
    sim2=Simulation(p);sim2.advance_to(3.)
    ok('Same seed reproduces trajectories',np.array_equal(sim.x,sim2.x) and np.array_equal(sim.theta,sim2.theta))
    print(f'Passed {count} focused checks')

if __name__=='__main__':test()
