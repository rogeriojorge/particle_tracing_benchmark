#!/usr/bin/env python3
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
from neat.fields import Simple, Vmec as Vmec_NEAT
from scipy.interpolate import CubicSpline as spline
from neat.tracing import ChargedParticle, ParticleOrbit
from desc.equilibrium import Equilibrium
from desc.objectives import get_fixed_boundary_constraints
from desc.vmec import VMECIO
#############################################
mpi = MpiPartition(MPI.COMM_WORLD.Get_size())
mayavi_loaded = True
if mpi.proc0_world:
    try:
        from mayavi import mlab
    except Exception as e:
        print(e)
        mayavi_loaded = False
mpi.comm_world.barrier()
mpi.comm_world.bcast(mayavi_loaded, root=0)
def pprint(*args, **kwargs):
    if mpi.proc0_world: print(*args, **kwargs)
#############################################
plot3D=False
min_surface_radius=0.1
#############################################
phi0 = inputs.varphi_initial
R0axis = 0
Z0axis = 0
iota0 = np.empty((3,))
RZPhi_initial = np.empty((3,))
if mpi.proc0_world:
    from neat.fields import StellnaQS
    from scipy.interpolate import CubicSpline as spline
    field_nearaxis_basis = StellnaQS.from_paper("precise QH", B0=inputs.b0_ARIES, nphi=201)
    field_nearaxis = StellnaQS(rc=field_nearaxis_basis.rc*inputs.Rmajor_ARIES, zs=field_nearaxis_basis.zs*inputs.Rmajor_ARIES, etabar=field_nearaxis_basis.etabar/inputs.Rmajor_ARIES, B2c=field_nearaxis_basis.B2c*(inputs.b0_ARIES/inputs.Rmajor_ARIES/inputs.Rmajor_ARIES),\
                                B0=inputs.b0_ARIES, nfp=field_nearaxis_basis.nfp, order='r3', nphi=111)
    print(field_nearaxis.rc)
    # field_desc = Equilibrium.from_near_axis(field_nearaxis, r= inputs.Aminor_ARIES, L=8, M=8, N=8)
    # constraints = get_fixed_boundary_constraints(iota=False)
    # field_desc.solve(verbose=3, ftol=1e-2,objective="force",maxiter=100,xtol=1e-6,constraints=constraints)
    nu_of_varphi = spline(np.append(field_nearaxis.varphi,2*np.pi/inputs.nfp), np.append(field_nearaxis.varphi-field_nearaxis.phi,0), bc_type='periodic')
    phi0 = inputs.varphi_initial-nu_of_varphi(inputs.varphi_initial)
    RZPhi_initial = np.ravel(field_nearaxis.to_RZ([[inputs.r_initial,inputs.theta_initial,phi0]]))
    R0axis = field_nearaxis.R0_func(phi0)
    Z0axis = field_nearaxis.Z0_func(phi0)
    iota0 = np.array([field_nearaxis.iota,0,0])
mpi.comm_world.Bcast(iota0, root=0)
iota0=iota0[0]
mpi.comm_world.bcast(phi0, root=0)
mpi.comm_world.bcast(R0axis, root=0)
mpi.comm_world.bcast(Z0axis, root=0)
mpi.comm_world.Bcast(RZPhi_initial, root=0)
pprint("Starting qsc desc benchmark")
#############################################
# Initialize folders and variables
#############################################
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'comparison_qsc_desc_r_min{inputs.r_min:.2f}r_ARIES{inputs.r_ARIES:.2f}_nsVMEC{np.max(inputs.ns_array)}')
os.makedirs(OUT_DIR, exist_ok=True)
if mpi.proc0_world: shutil.copyfile(os.path.join(Path(__file__).parent.resolve(),'inputs.py'), os.path.join(OUT_DIR,'copy_inputs.py'))
os.chdir(OUT_DIR)
inputs.minor_radius_array = inputs.minor_radius_array[::-1]
#############################################
# Show orbit of particle layered VMEC
#############################################
if plot3D:
    pprint('3D rendering particle orbits')
    minor_radius_plot=inputs.minor_radius_array[0]
    vmec_input_plot = os.path.join(OUT_DIR, f'input.na_A{minor_radius_plot:.2f}')
    vmec_output_plot = os.path.join(OUT_DIR, f'wout_na_A{minor_radius_plot:.2f}.nc')
    pprint('  Creating VMEC input')
    if mpi.proc0_world: field_nearaxis.to_vmec(filename=vmec_input_plot,r=minor_radius_plot, params={"ntor":7, "mpol":7, "ns_array":inputs.ns_array,"niter_array":inputs.niter_array,"ftol_array":inputs.ftol_array}, ntheta=14, ntorMax=7)
    mpi.comm_world.barrier()
    if mpi.proc0_world and mayavi_loaded:
        particle = ChargedParticle(r_initial=inputs.minor_radius_array[0]/2, theta_initial=inputs.theta_initial, phi_initial=inputs.varphi_initial, Lambda=inputs.Lambda)
        fig = mlab.figure(bgcolor=(1,1,1), size=(1100,800))
        pprint('  Running near-axis orbit')
        orbit_nearaxis = ParticleOrbit(particle, field_nearaxis, nsamples=inputs.nsamples, tfinal=inputs.tfinal, constant_b20=inputs.constant_b20)
        orbit_rpos_cartesian = orbit_nearaxis.rpos_cartesian
        mlab.plot3d(orbit_rpos_cartesian[0], orbit_rpos_cartesian[1], orbit_rpos_cartesian[2], tube_radius=0.025, color=(0.6, 0.1, 0.1)) #red tracing
        pprint('  Readjusting particle position')
        particle.r_initial = ((inputs.minor_radius_array[0]/2)**2)/(minor_radius_plot**2) # keep initializing the particle at the same location as the near-axis one
        particle.phi_initial = RZPhi_initial[2]
        particle.theta_initial = -inputs.theta_initial
        pprint('  Running gyronimo orbit')
        # VMECIO.save(field_desc, vmec_output_plot)
        vmec_NEAT = Vmec_NEAT(wout_filename=vmec_output_plot, maximum_s=inputs.maximum_s_particle)
        orbit_gyronimo = ParticleOrbit(particle, vmec_NEAT, nsamples=inputs.nsamples, tfinal=inputs.tfinal)
        orbit_rpos_cartesian = orbit_gyronimo.rpos_cartesian
        mlab.plot3d(orbit_rpos_cartesian[0], orbit_rpos_cartesian[1], orbit_rpos_cartesian[2], tube_radius=0.025, color=(0.1, 0.6, 0.1)) #green tracing
        pprint('  Running SIMPLE orbit')
        # particle.theta_initial = inputs.theta_initial
        field_simple = Simple(wout_filename=vmec_output_plot, B_scale=1, Aminor_scale=1, multharm=3,ns_s=3,ns_tp=3)
        orbit_simple = ParticleOrbit(particle, field_simple, nsamples=inputs.nsamples, tfinal=inputs.tfinal)
        orbit_rpos_cartesian = orbit_simple.rpos_cartesian
        # mlab.plot3d(orbit_rpos_cartesian[0], orbit_rpos_cartesian[1], orbit_rpos_cartesian[2], tube_radius=0.025, color=(0.1, 0.1, 0.6)) #blue tracing
        pprint('  Plotting surfaces')
        opacity_array = np.sqrt(np.linspace(0.8,1.0,inputs.surfaces_plot_qsc_vmec))[::-1]
        minor_radius_plot_array = np.linspace(min_surface_radius,inputs.minor_radius_array[0],inputs.surfaces_plot_qsc_vmec,endpoint=True)
        for i, minor_radius_plot in enumerate(minor_radius_plot_array):
            nphi_plot = 80
            ntheta_plot = 30
            if i==0: phimin=0
            else: phimin=np.pi/2
            phimax=2*np.pi
            x_2D_plot, y_2D_plot, z_2D_plot, R_2D_plot = field_nearaxis.get_boundary(r=minor_radius_plot, ntheta=ntheta_plot, nphi=nphi_plot, ntheta_fourier=14, mpol=7, ntor=7, phimin=phimin, phimax=phimax)
            theta1D = np.linspace(0, 2 * np.pi, ntheta_plot)
            phi1D = np.linspace(0, 2 * np.pi, nphi_plot)
            phi2D, theta2D = np.meshgrid(phi1D, theta1D)
            Bmag = field_nearaxis.B_mag(r=minor_radius_plot, theta=theta2D, phi=phi2D)
            Out=mlab.mesh(x_2D_plot, y_2D_plot, z_2D_plot, scalars=Bmag, colormap='blue-red', opacity=opacity_array[i])
            Out.actor.property.lighting = False
            # Out.scene.renderer.use_depth_peeling = True
            # Out.scene.renderer.maximum_number_of_peels = 8
            mlab.view(azimuth=30, elevation=70, focalpoint=(-0.15,0,0), figure=fig)
        cb = mlab.colorbar(orientation='horizontal', title='|B| [T]', nb_labels=7)
        cb.scalar_bar.unconstrained_font_size = True
        cb.label_text_property.font_family = 'times'
        cb.label_text_property.bold = 0
        cb.label_text_property.font_size=20
        cb.label_text_property.color=(0,0,0)
        cb.title_text_property.font_family = 'times'
        cb.title_text_property.font_size=24
        cb.title_text_property.color=(0,0,0)
        cb.title_text_property.bold = 1
        mlab.savefig(filename=os.path.join(OUT_DIR,'3D_stells.png'), figure=fig)
        mlab.show()
#############################################
# Compare geometries
#############################################
ntheta = 200
nradius = 6
numRows = int(np.floor(np.sqrt(inputs.n_minor_radius)))
numCols = int(np.ceil(inputs.n_minor_radius / numRows))
plotNum = 1
raxis_r0axis_relerror_array = []
zaxis_z0axis_relerror_array = []
R_VMEC_r_nearaxis_relerror_array = []
Z_VMEC_z_nearaxis_relerror_array = []
iota_relerror_array = []
aspect_ratio_array = []
iota_array = []
mpi.comm_world.barrier()
if mpi.proc0_world:
    fig, axs = plt.subplots(numRows, numCols, sharex=True, sharey=True)
    axs = axs.ravel()
for i, minor_radius in enumerate(inputs.minor_radius_array):
    pprint(f"Running minor_radius = {minor_radius:.3f}")
    vmec_input = os.path.join(OUT_DIR, f'input.na_A{minor_radius:.2f}')
    vmec_output=os.path.join(OUT_DIR, f'wout_na_A{minor_radius:.2f}.nc')
    start_time = time.time()
    if mpi.proc0_world: #to_vmec was here
        # field_desc = Equilibrium.from_near_axis(field_nearaxis, r=minor_radius , L=8, M=8, N=8)
        # constraints = get_fixed_boundary_constraints(iota=False)
        # field_desc.solve(verbose=3, ftol=1e-2,objective="force",maxiter=100,xtol=1e-6,constraints=constraints)
        pprint(f"  Creating DESC input took {(time.time() - start_time):.2f}s")
        pprint("Saving DESC equilibrium in VMEC style output file")
        # VMECIO.save(field_desc, vmec_output)
        vmec_NEAT = Vmec_NEAT(wout_filename=vmec_output, maximum_s=inputs.maximum_s_particle)
        pprint("Loading VMEC style file")
        vmec = Vmec(vmec_output, verbose=False)
        aspect_ratio = vmec.aspect()
        aspect_ratio_array.append(aspect_ratio)
        iota=vmec.wout.iotaf[1]
        iota_array.append(iota)
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
        angle = vmec.wout.xm[imode]*(-inputs.theta_initial) - vmec.wout.xn[imode]*RZPhi_initial[2]
        R_VMEC_initial += vmec.wout.rmnc[imode,iradus_initial]*np.cos(angle)
        Z_VMEC_initial += vmec.wout.zmns[imode,iradus_initial]*np.sin(angle)
    for imode in range(len(vmec.wout.xn)):
        angle = vmec.wout.xm[imode]*0 - vmec.wout.xn[imode]*RZPhi_initial[2]
        R_VMEC_theta0 += vmec.wout.rmnc[imode,iradus_initial]*np.cos(angle)
        Z_VMEC_theta0 += vmec.wout.zmns[imode,iradus_initial]*np.sin(angle)
    for itheta in range(ntheta):
        for imode in range(len(vmec.wout.xn)):
            angle = vmec.wout.xm[imode]*(theta[itheta]) - vmec.wout.xn[imode]*phi0 # RZPhi_initial[2] was the correct one but other surfaces are approximated
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

    iota_relerror = np.abs((np.abs(iota)-np.abs(iota0))/iota)
    iota_relerror_array.append(iota_relerror)
    pprint(f'  iota is {iota:.2}')
    pprint(f'  iota0 is {iota0:.2}')
    pprint(f'  Rel error iota and iota is {iota_relerror:.2}')

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
    plt.savefig('qsc_vmec_location.pdf')
    # plt.show()

if mpi.proc0_world:
    fig, ax = plt.subplots()
    # plt.subplot(numRows,numCols,plotNum)
    plt.plot(aspect_ratio_array, raxis_r0axis_relerror_array, label=r'$(Raxis_{VMEC}-Raxis_{near-axis})/Raxis_{VMEC}$')
    plt.plot(aspect_ratio_array, zaxis_z0axis_relerror_array, label=r'$(Zaxis_{VMEC}-Zaxis_{near-axis})/Zaxis_{VMEC}$')
    plt.plot(aspect_ratio_array, R_VMEC_r_nearaxis_relerror_array, label=r'$(R_{VMEC} - R_{near-axis})/R_{VMEC}$')
    plt.plot(aspect_ratio_array, Z_VMEC_z_nearaxis_relerror_array, label=r'$(Z_{VMEC} - Z_{near-axis})/Z_{VMEC}$')
    plt.plot(aspect_ratio_array, iota_relerror_array, label=r'$(\iota_{VMEC}-\iota_{near-axis})/\iota_{VMEC}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Aspect Ratio of the Plasma Boundary')
    plt.ylabel('Relative Error in Location')
    plt.legend()
    plt.tight_layout()
    plt.savefig('qsc_vmec_relerrors.pdf')
    # plt.show()

    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"input.*")): os.remove(objective_file)
    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"wout_*")): os.remove(objective_file)
    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"threed1*")): os.remove(objective_file)
    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"parvmecinfo*")): os.remove(objective_file)
    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"fort*")): os.remove(objective_file)
