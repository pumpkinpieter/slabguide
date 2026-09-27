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


def Plot3D(P, xs=None, ys=None, ymin=0, ymax=15, yref=450, xstride=6):
    if xs is None:
        xs = P.Slab.all_Xs[::xstride]
    if ys is None:
        ys = np.linspace(ymin, ymax, yref)
        
    ymin, ymax = np.min(ys), np.max(ys)
    ylims = np.array([ymin, ymax], dtype=np.float32)

    Xs, Ys = np.meshgrid(xs, ys)
    MS = ThreeJSMeshSaver(Xs, Ys)

    trigs = MS.trig_indices
    basepts = MS.base_vertices

    gs = P.Guided(xs, ys)
    ps = P.Propagating(xs, ys)
    evs = P.Evanescent(xs, ys)
    f0s = np.array(P.f0(xs), dtype=np.float32)

    guided = MS.create_height_array_from_complex(gs)
    propagating = MS.create_height_array_from_complex(ps)
    evanescent = MS.create_height_array_from_complex(evs)
    Rhos = np.array(P.Slab.Rhos, dtype=np.float32)
    
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

    three_viewer = HelloWidget()
    display(three_viewer)

    # return HelloWidget()
