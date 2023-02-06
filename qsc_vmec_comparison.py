#!/usr/bin/env python
import os
import glob
import time
import shutil
import inputs
import numpy as np
from mpi4py import MPI
from pathlib import Path
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from simsopt.util import MpiPartition
from scipy.interpolate import CubicSpline as spline
#############################################
mpi = MpiPartition(MPI.COMM_WORLD.Get_size())
def pprint(*args, **kwargs):
    if mpi.proc0_world: print(*args, **kwargs)
phi0 = inputs.varphi_initial
R0axis = 0
Z0axis = 0
RZPhi_initial = np.empty((3,))
if mpi.proc0_world:
    from neat.fields import StellnaQS
    from scipy.interpolate import CubicSpline as spline
    field_nearaxis = StellnaQS(rc=inputs.rc*inputs.Rmajor_ARIES, zs=inputs.zs*inputs.Rmajor_ARIES, etabar=inputs.eta_bar/inputs.Rmajor_ARIES, B2c=inputs.B2c*(inputs.b0_ARIES/inputs.Rmajor_ARIES/inputs.Rmajor_ARIES), B0=inputs.b0_ARIES, nfp=inputs.nfp, order='r3', nphi=111)
    nu_of_varphi = spline(np.append(field_nearaxis.varphi,2*np.pi/inputs.nfp), np.append(field_nearaxis.varphi-field_nearaxis.phi,0), bc_type='periodic')
    phi0 = inputs.varphi_initial-nu_of_varphi(inputs.varphi_initial)
    RZPhi_initial = np.ravel(field_nearaxis.to_RZ([[inputs.r_initial,inputs.theta_initial,phi0]]))
    R0axis = field_nearaxis.R0_func(phi0)
    Z0axis = field_nearaxis.Z0_func(phi0)
mpi.comm_world.bcast(phi0, root=0)
mpi.comm_world.bcast(R0axis, root=0)
mpi.comm_world.bcast(Z0axis, root=0)
mpi.comm_world.Bcast(RZPhi_initial, root=0)
pprint("Starting qsc vmec benchmark")
#############################################
# Initialize folders and variables
#############################################
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'comparison_qsc_vmec_r_min{inputs.r_min:.2f}r_ARIES{inputs.r_ARIES:.2f}')
os.makedirs(OUT_DIR, exist_ok=True)
if mpi.proc0_world: shutil.copyfile(os.path.join(Path(__file__).parent.resolve(),'inputs.py'), os.path.join(OUT_DIR,'copy_inputs.py'))
os.chdir(OUT_DIR)
#############################################
# Compare geometries
#############################################
ntheta = 200
nradius = 6
pprint(f'Varphi0 = {inputs.varphi_initial}')
pprint(f'Phi0 = {phi0}')
raxis_r0axis_relerror_array = []
zaxis_z0axis_relerror_array = []
R_VMEC_r_nearaxis_relerror_array = []
Z_VMEC_z_nearaxis_relerror_array = []
numRows = int(np.floor(np.sqrt(inputs.n_minor_radius)))
numCols = int(np.ceil(inputs.n_minor_radius / numRows))
plotNum = 1
fig, axs = plt.subplots(numRows, numCols)
axs = axs.ravel()
for i, minor_radius in enumerate(inputs.minor_radius_array):
    pprint(f"Running minor_radius = {minor_radius:.3f}")
    vmec_input = os.path.join(OUT_DIR, f'input.na_A{minor_radius:.1}')
    start_time = time.time()
    if mpi.proc0_world: field_nearaxis.to_vmec(filename=vmec_input,r=minor_radius, params={"ntor":7, "mpol":7, "ns_array":inputs.ns_array,"niter_array":[5000,5000,7000],"ftol_array":[1e-12,1e-14,1e-14]}, ntheta=14, ntorMax=7)
    pprint(f"  Creating VMEC input took {(time.time() - start_time):.2f}s")
    vmec = Vmec(vmec_input, verbose=False)
    start_time = time.time()
    vmec.run()
    aspect_ratio = vmec.aspect()
    pprint(f"  VMEC ran in {(time.time() - start_time):.2f}s")

    theta = np.linspace(0,2*np.pi,num=ntheta)
    iradii = np.linspace(0,np.max(inputs.ns_array)-1,num=nradius).round()
    iradii = [int(i) for i in iradii]
    R = np.zeros((ntheta,nradius))
    Z = np.zeros((ntheta,nradius))
    for itheta in range(ntheta):
        for iradius in range(nradius):
            for imode in range(len(vmec.wout.xn)):
                angle = vmec.wout.xm[imode]*theta[itheta] - vmec.wout.xn[imode]*phi0
                R[itheta,iradius] += vmec.wout.rmnc[imode,iradii[iradius]]*np.cos(angle)
                Z[itheta,iradius] += vmec.wout.zmns[imode,iradii[iradius]]*np.sin(angle)
    Raxis = 0
    Zaxis = 0
    for n in range(vmec.wout.ntor+1):
        angle = -n*inputs.nfp*phi0
        Raxis += vmec.wout.raxis_cc[n]*np.cos(angle)
        Zaxis += vmec.wout.zaxis_cs[n]*np.sin(angle)

    toroidalflux_initial = (inputs.r_initial**2)*np.pi*inputs.b0_ARIES
    phiedge = (minor_radius**2)*np.pi*inputs.b0_ARIES
    s_initial = toroidalflux_initial/phiedge
    iradus_initial = np.abs(vmec.wout.phi-toroidalflux_initial).argmin()
    R_VMEC_initial = 0
    Z_VMEC_initial = 0
    R_VMEC_theta0 = 0
    Z_VMEC_theta0 = 0
    R_VMEC_initial_array = np.zeros((ntheta,))
    Z_VMEC_initial_array = np.zeros((ntheta,))
    for imode in range(len(vmec.wout.xn)):
        angle = vmec.wout.xm[imode]*(np.pi-inputs.theta_initial) - vmec.wout.xn[imode]*RZPhi_initial[2]
        R_VMEC_initial += vmec.wout.rmnc[imode,iradus_initial]*np.cos(angle)
        Z_VMEC_initial += vmec.wout.zmns[imode,iradus_initial]*np.sin(angle)
    for imode in range(len(vmec.wout.xn)):
        angle = vmec.wout.xm[imode]*0 - vmec.wout.xn[imode]*RZPhi_initial[2]
        R_VMEC_theta0 += vmec.wout.rmnc[imode,iradus_initial]*np.cos(angle)
        Z_VMEC_theta0 += vmec.wout.zmns[imode,iradus_initial]*np.sin(angle)
    for itheta in range(ntheta):
        for imode in range(len(vmec.wout.xn)):
            angle = vmec.wout.xm[imode]*theta[itheta] - vmec.wout.xn[imode]*RZPhi_initial[2]
            R_VMEC_initial_array[itheta] += vmec.wout.rmnc[imode,iradus_initial]*np.cos(angle)
            Z_VMEC_initial_array[itheta] += vmec.wout.zmns[imode,iradus_initial]*np.sin(angle)

    R_VMEC_r_nearaxis_relerror = np.abs((R_VMEC_initial-RZPhi_initial[0])/R_VMEC_initial)
    Z_VMEC_z_nearaxis_relerror = np.abs((Z_VMEC_initial-RZPhi_initial[1])/(Z_VMEC_initial+1e-15))
    R_VMEC_r_nearaxis_relerror_array.append(R_VMEC_r_nearaxis_relerror)
    Z_VMEC_z_nearaxis_relerror_array.append(Z_VMEC_z_nearaxis_relerror)
    pprint(f"  Initial point in psi={toroidalflux_initial}, corresponding to VMEC phi={vmec.wout.phi[iradus_initial]}")
    pprint(f"  True (R,Z,phi)={[RZPhi_initial[0],RZPhi_initial[1],RZPhi_initial[2]]}")
    pprint(f"  VMEC (R,Z,phi)={[R_VMEC_initial,Z_VMEC_initial,RZPhi_initial[2]]}")

    raxis_r0axis_relerror = np.abs((Raxis-R0axis)/Raxis)
    zaxis_z0axis_relerror = np.abs((Zaxis-Z0axis)/(Zaxis+1e-15))
    raxis_r0axis_relerror_array.append(raxis_r0axis_relerror)
    zaxis_z0axis_relerror_array.append(zaxis_z0axis_relerror)
    pprint(f'  Rel error Raxis and R0axis is {raxis_r0axis_relerror:.2}')
    pprint(f'  Rel error Zaxis and Z0axis is {zaxis_z0axis_relerror:.2}')

    if mpi.proc0_world:
        # plt.subplot(numRows,numCols,plotNum)
        # plotNum += 1
        for iradius in range(nradius):
            axs[i].plot(R[:,iradius], Z[:,iradius], '-', label = '_nolegend_')
        axs[i].plot(R_VMEC_initial_array, Z_VMEC_initial_array, '-', label='Initial flux surface')
        axs[i].plot([Raxis],[Zaxis],'.b', label='VMEC axis')
        axs[i].plot([R0axis],[Z0axis],'*g', label='Near-Axis (true) axis')
        axs[i].plot([R_VMEC_initial],[Z_VMEC_initial],'xr',label='VMEC Initial Position')
        axs[i].plot([RZPhi_initial[0]],[RZPhi_initial[1]],'ok',label='Near-Axis (true) Initial Position')
        # axs[i].plot([R_VMEC_theta0],[Z_VMEC_theta0],'Xb')
        axs[i].set_xlabel('R', fontsize=10)
        axs[i].set_ylabel('Z', fontsize=10)
        axs[i].title.set_text(f'aspect ratio = {aspect_ratio:.2f}')
        # if i==len(inputs.minor_radius_array)-1: axs[i].legend(fontsize=8)

if mpi.proc0_world:
    handles, labels = axs[len(inputs.minor_radius_array)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=8)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.tight_layout()
    plt.savefig('qsc_vmec_location.png')
    # plt.show()

if mpi.proc0_world:
    fig, ax = plt.subplots()
    # plt.subplot(numRows,numCols,plotNum)
    plt.plot(inputs.minor_radius_array, raxis_r0axis_relerror_array, label='(raxis-r0axis)/raxis')
    plt.plot(inputs.minor_radius_array, zaxis_z0axis_relerror_array, label='(zaxis-z0axis)/zaxis')
    plt.plot(inputs.minor_radius_array, R_VMEC_r_nearaxis_relerror_array, label='(R_VMEC - R_nearaxis)/R_VMEC')
    plt.plot(inputs.minor_radius_array, Z_VMEC_z_nearaxis_relerror_array, label='(Z_VMEC - Z_nearaxis)/Z_VMEC')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Minor Radius of the Plasma Boundary')
    plt.ylabel('Error in Location')
    plt.legend()
    plt.tight_layout()
    plt.savefig('qsc_vmec_relerrors.png')
    # plt.show()

    for objective_file in glob.glob(os.path.join(OUT_DIR,f"input.*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"wout_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"threed1*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"parvmecinfo*")): os.remove(objective_file)
    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"fort*")): os.remove(objective_file)
