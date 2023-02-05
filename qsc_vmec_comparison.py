#!/usr/bin/env python
import os
import time
import numpy as np
from mpi4py import MPI
from pathlib import Path
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec
import inputs
import matplotlib.pyplot as plt
#############################################
mpi = MpiPartition(MPI.COMM_WORLD.Get_size())
def pprint(*args, **kwargs):
    if mpi.proc0_world: print(*args, **kwargs)
pprint("Starting qsc vmec benchmark")
#############################################
# Initialize folders and variables
#############################################
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'comparison_qsc_vmec_r_min{inputs.r_min}r_ARIES{inputs.r_ARIES}')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
#############################################
# Compare geometries
#############################################
minor_radius_array = np.linspace(inputs.r_min,inputs.r_ARIES,inputs.n_minor_radius)
nfp = inputs.field_nearaxis.nfp
n_surfaces = [16, 31, 51]
ntheta = 200
nradius = 8
for i, minor_radius in enumerate(minor_radius_array):
    pprint(f"Running minor_radius = {minor_radius:.2f}")
    vmec_input = os.path.join(OUT_DIR, f'input.na_A{minor_radius:.2f}')
    start_time = time.time()
    if mpi.proc0_world: inputs.field_nearaxis.to_vmec(filename=vmec_input,r=minor_radius, params={"ntor":7, "mpol":7, "ns_array":n_surfaces,"niter_array":[5000,5000,5000],"ftol_array":[1e-12,1e-14,1e-14]}, ntheta=14, ntorMax=7)
    pprint(f"  Creating VMEC input took {(time.time() - start_time):.2f}s")
    vmec = Vmec(vmec_input, verbose=False)
    start_time = time.time()
    vmec.run()
    pprint(f"  VMEC ran in {(time.time() - start_time):.2f}s")

    theta = np.linspace(0,2*np.pi,num=ntheta)
    iradii = np.linspace(0,n_surfaces[-1]-1,num=nradius).round()
    iradii = [int(i) for i in iradii]
    R = np.zeros((ntheta,nradius))
    Z = np.zeros((ntheta,nradius))
    izeta = 0
    zeta = 0
    for itheta in range(ntheta):
        for iradius in range(nradius):
            for imode in range(len(vmec.wout.xn)):
                angle = vmec.wout.xm[imode]*theta[itheta] - vmec.wout.xn[imode]*zeta
                R[itheta,iradius] += vmec.wout.rmnc[imode,iradii[iradius]]*np.cos(angle)
                Z[itheta,iradius] += vmec.wout.zmns[imode,iradii[iradius]]*np.sin(angle)
    Raxis = 0
    Zaxis = 0
    for n in range(vmec.wout.ntor+1):
        angle = -n*nfp*zeta
        Raxis += vmec.wout.raxis_cc[n]*np.cos(angle)
        Zaxis += vmec.wout.zaxis_cs[n]*np.sin(angle)

    pprint(f'Difference between Raxis={Raxis} and qscRaxis={inputs.field_nearaxis.R0_func(zeta)} is {Raxis-inputs.field_nearaxis.R0_func(zeta)}')

    fig = plt.figure()
    fig.set_size_inches(14,7)
    fig.patch.set_facecolor('white')
    for iradius in range(nradius):
        plt.plot(R[:,iradius], Z[:,iradius], '-')
    ### IMPORTANT -> the toroidal angle phi on-axis is not equal to the cylindrical toroidal angle phi off-axis
    plt.plot([Raxis],[Zaxis],'xr')
    plt.plot([inputs.field_nearaxis.R0_func(zeta)],[inputs.field_nearaxis.Z0_func(zeta)],'ok')
    plt.gca().set_aspect('equal',adjustable='box')
    plt.xlabel('R', fontsize=10)
    plt.ylabel('Z', fontsize=10)
    plt.title(r'$\phi$ = '+str(round(zeta,2)))
    plt.tight_layout()
    plt.show()