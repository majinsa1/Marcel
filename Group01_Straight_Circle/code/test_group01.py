#!/usr/bin/env python3
"""Focused unit and small-N integration tests for all 3 circular wall types."""
from dataclasses import replace
import numpy as np
from model import Params,Simulation,initialise,interactions,circle_billiard,wall_force,transport


def test():
    count=0
    def check(label,good):
        nonlocal count
        if not good:raise AssertionError(label)
        count+=1
        print('PASS:',label,flush=True)
    p=Params(N=50,K=1.,shape='circle',boundary='specular',side=20.,omega_mean=0.,omega_width=0.,omega_law='constant',repulsion_k=0)
    x,theta,omega,rng=initialise(p)
    x2,t2,omega2,_=initialise(replace(p,K=2.))
    check('Identical seed => identical starting positions and headings',np.array_equal(x,x2) and np.array_equal(theta,t2))
    check('No intrinsic chirality',np.array_equal(omega,np.zeros(p.N)))
    check('Uniform points inside disk',np.max(np.linalg.norm(x,axis=1))<=p.radius)
    align,forces,*_=interactions(x,theta,p)
    ref=np.empty(p.N)
    for i in range(p.N):
        nb=np.linalg.norm(x-x[i],axis=1)<p.align_radius
        nb[i]=False
        ref[i]=p.K*np.sin(theta[nb]-theta[i]).sum()/(nb.sum()+1)
    check('KDTree neighbor alignment matches direct O(N^2) sum',np.allclose(align,ref,atol=1e-12))
    check('No mechanical force',np.all(forces==0))
    R=p.radius
    pos=np.array([[R-.1,0.],[0.,0.]])
    heading=np.array([[1.,0.],[0.,1.]])
    out,direction,jump=circle_billiard(pos,heading,.4,R)
    check('Specular reflection uses leftover distance',np.allclose(out[0],[R-.3,0],atol=1e-10))
    check('Specular reflection turns heading without changing its norm',np.allclose(direction[0],[-1,0]) and np.allclose(np.linalg.norm(direction,axis=1),1))
    check('Uncollided particle goes straight',np.allclose(out[1],[0,.4]))
    # For K=0 the circular billiard keeps x cross heading constant.
    pos=x.copy();e=np.column_stack((np.cos(theta),np.sin(theta)))
    angular0=pos[:,0]*e[:,1]-pos[:,1]*e[:,0]
    for _ in range(80):pos,e,_=circle_billiard(pos,e,0.3,R)
    angular1=pos[:,0]*e[:,1]-pos[:,1]*e[:,0]
    check('Circle specular billiard angular momentum invariant for K=0',np.max(abs(angular0-angular1))<1e-8)
    for wall in ['specular','sliding','soft']:
        sim=Simulation(replace(p,boundary=wall))
        sim.advance_to(3)
        check(wall+' finite particle state',np.isfinite(sim.x).all() and np.isfinite(sim.theta).all())
        if wall != 'soft':
            check(wall+' remains inside circle',np.linalg.norm(sim.x,axis=1).max()<=R+1e-8)
    # Soft-wall restoring force is inward in annulus R-w<r.
    sp=replace(p,boundary='soft',wall_width=2.,wall_k=4.)
    force=wall_force(np.array([[R-.5,0],[0.,0.]]),sp)
    check('Soft wall acts inward near radius',force[0,0]<0 and np.allclose(force[1],0))
    # Sliding at wall, outward radial heading remains outward while tangential translation is constrained.
    slide=replace(p,boundary='sliding')
    xx=np.zeros((p.N,2));xx[:,0]=R-.05
    th=np.zeros(p.N)
    v=np.column_stack((np.ones(p.N),np.zeros(p.N)))
    after,th_after,_,_=transport(xx,th,v,.1,slide,np.random.default_rng(3),np.zeros((p.N,2),dtype=np.int8))
    check('Sliding constrains normal displacement',np.max(np.linalg.norm(after,axis=1))<=R+1e-8)
    check('Sliding does not spuriously rotate intrinsic heading',np.allclose(th_after,th))
    print(f'ALL {count} TESTS PASSED')

if __name__=='__main__':test()
