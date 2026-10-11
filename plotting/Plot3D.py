#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:32:49 2026

@author: pv
"""
import anywidget
import traitlets
import numpy as np
from pathlib import Path
from IPython.display import display
from ..utilities import ThreeJSMeshSaver
from time import time

def Plot3D(*Ps, xs=None, ys=None, ymin=0, ymax=15, yref=450, xstride=6, 
           time_check=False):
    if xs is None:
        xs = Ps[0].Slab.all_Xs[::xstride]
    if ys is None:
        ys = np.linspace(ymin, ymax, yref)
        
    ymin, ymax = np.min(ys), np.max(ys)

    if time_check:
        print('Taking Meshgrid of xs and ys.\n')
        a = time()

    Xs, Ys = np.meshgrid(xs, ys)

    if time_check:
        b = time()
        print("Time taken : %.4f.\n"%(b-a))

    if time_check:
        print('Creating meshsaver.\n')
        a = time()

    MS = ThreeJSMeshSaver(Xs, Ys)

    if time_check:
        b = time()
        print("Time taken : %.4f.\n"%(b-a))

    trigs = MS.trig_indices
    basepts = MS.base_vertices

    if time_check:
        print('Applying Propagator to xs and ys.\n')
        a = time()

    gs = np.array([P.Guided(xs,ys) for P in Ps])
    ps = np.array([P.Propagating(xs, ys) for P in Ps])
    evs = np.array([P.Evanescent(xs, ys) for P in Ps])
    f0s = np.array(Ps[0].f0(xs), dtype=np.float32)
    Nmesh = np.array(len(Ps), dtype=np.uint32)

    if time_check:
        b = time()
        print("Time taken : %.4f.\n"%(b-a))

    if time_check:
        print('Creating height arrays from data.\n')
        a = time()

    guided = MS.create_height_array_multi_mesh_complex(gs)
    propagating = MS.create_height_array_multi_mesh_complex(ps)
    evanescent = MS.create_height_array_multi_mesh_complex(evs)

    if time_check:
        b = time()
        print("Time taken : %.4f.\n"%(b-a))

    Rhos = np.array(Ps[0].Slab.Rhos, dtype=np.float32)
    
    class HelloWidget(anywidget.AnyWidget):
        _esm = Path(__file__).parent / "widget.js"
        _css = Path(__file__).parent / "styles.css"
        _trig_bytes = traitlets.Bytes(trigs.tobytes()).tag(sync=True)
        _base_bytes = traitlets.Bytes(basepts.tobytes()).tag(sync=True)
        _f0_bytes = traitlets.Bytes(f0s.tobytes()).tag(sync=True)
        _x0_bytes = traitlets.Bytes(
            np.array(xs, dtype=np.float32).tobytes()).tag(sync=True)
        _y0_bytes = traitlets.Bytes(
            np.array(ys, dtype=np.float32).tobytes()).tag(sync=True)
        _guided_bytes = traitlets.Bytes(guided.tobytes()).tag(sync=True)
        _prop_bytes = traitlets.Bytes(propagating.tobytes()).tag(sync=True)
        _ev_bytes = traitlets.Bytes(evanescent.tobytes()).tag(sync=True)
        _rho_bytes = traitlets.Bytes(Rhos.tobytes()).tag(sync=True)
        _Nmesh = traitlets.Bytes(Nmesh.tobytes()).tag(sync=True)

    if time_check:
        print('Instantiating widget.\n')
        a = time()
    three_viewer = HelloWidget()
    if time_check:
        b = time()
        print("Time taken : %.4f.\n"%(b-a))

    if time_check:
        print('Displaying widget.\n')
        a = time()
    display(three_viewer)
    if time_check:
        b = time()
        print("Time taken : %.4f.\n"%(b-a))
    # return three_viewer

    # return HelloWidget()
