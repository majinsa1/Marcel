"""Regression tests for five square boundaries, local alignment and resume."""
import json
import tempfile
from pathlib import Path
import numpy as np
from model import Params,Simulation,interactions,transport,wall_force
from run import run_case,DEFAULT_RUN


def p_for(b,N=24,K=1.5):
    return Params(N=N,side=20.,K=K,shape='square',boundary=b,dt=.02,
                  omega_law='constant',omega_mean=0.,omega_width=0.,seed=12)


def test_periodic_wrap():
    p=p_for('periodic',N=1)
    x=np.array([[9.9,-9.8]]);theta=np.array([.2]);v=np.array([[1.,-1.]])
    xx,tt,_,jump=transport(x,theta,v,1.,p,np.random.default_rng(1),np.zeros((1,2),dtype=np.int8))
    assert np.allclose(xx,[[-9.1,9.2]]) and np.allclose(tt,theta) and not jump.any()


def test_periodic_minimum_image():
    p=p_for('periodic',2,2);a=np.array([[-9.8,0],[9.8,0]])
    theta=np.array([0.,np.pi/2]);torque,*_=interactions(a,theta,p)
    assert np.allclose(torque,[1.,-1.]),torque
    p.boundary='specular';torque,*_=interactions(a,theta,p)
    assert np.allclose(torque,0.)


def test_mirror_reflection():
    p=p_for('specular',1)
    x=np.array([[9.9,0]]);theta=np.array([0.]);v=np.array([[1.,0.]])
    xx,tt,_,_=transport(x,theta,v,1.,p,np.random.default_rng(1),np.zeros((1,2),np.int8))
    assert np.allclose(xx,[[9.1,0]]) and np.allclose(np.cos(tt),[-1.])


def test_sliding_normal_blocked():
    p=p_for('sliding',1)
    x=np.array([[9.9,0]]);theta=np.array([0.]);v=np.array([[1.,0.]])
    xx,tt,c,_=transport(x,theta,v,1.,p,np.random.default_rng(1),np.zeros((1,2),np.int8))
    assert np.allclose(xx,[[10,0]]) and np.allclose(tt,theta) and c[0,0]==1


def test_mixed_probability_endpoints():
    x=np.array([[9.9,0]]);theta=np.array([0.]);v=np.array([[1.,0.]])
    for prob,expect in ((1.,9.1),(0.,10.)):
        p=p_for('mixed',1);p.wall_p_spec=(prob,)*4
        xx,_,_,_=transport(x,theta,v,1.,p,np.random.default_rng(1),np.zeros((1,2),np.int8))
        assert abs(xx[0,0]-expect)<1.e-8


def test_soft_force_vs_periodic():
    p=p_for('soft',2)
    x=np.array([[9.5,0.],[0.,0.]])
    f=wall_force(x,p)
    assert f[0,0]<0 and f[1,0]==0 and f[0,1]==0
    p.boundary='periodic';assert not wall_force(x,p).any()


def test_same_initial_conditions():
    p=p_for('periodic',32);x1=Simulation(p)
    p.boundary='specular';x2=Simulation(p)
    assert np.array_equal(x1.x,x2.x) and np.array_equal(x1.theta,x2.theta)


def test_advance_and_checkpoint():
    for boundary in ('specular','sliding','mixed','soft','periodic'):
        p=p_for(boundary,60)
        s=Simulation(p);s.advance_to(1.);s.advance_to(2.)
        assert np.isfinite(s.x).all() and np.isfinite(s.theta).all()
        if boundary in ('specular','sliding','mixed','periodic'):
            assert np.max(np.abs(s.x))<=p.half+1.e-8
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'state.npz';s.checkpoint(cp);rr=Simulation.restore(cp)
            assert np.array_equal(s.x,rr.x) and s.t==rr.t
            rr.advance_to(3.)
            assert abs(rr.t-3)<1.e-10


def test_short_run_resume():
    with tempfile.TemporaryDirectory() as td:
        p=p_for('periodic',16,0.5)
        from dataclasses import asdict
        out=Path(td)/'case'
        opts=dict(DEFAULT_RUN,mode='fixed',end=2.,sample_dt=1.,render=False)
        run_case({'params':asdict(p),'run':opts},out)
        opts['end']=3.;run_case({'params':asdict(p),'run':opts},out,resume=True)
        with np.load(out/'history.npz') as data:
            r=json.loads(str(data['rows_json']))
        assert abs(r[-1]['t']-3)<1.e-9

if __name__=='__main__':
    tests=[(k,v) for k,v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for k,fun in tests:
        fun();print('PASS',k,flush=True)
    print('TOTAL',len(tests),'TESTS PASSED',flush=True)
