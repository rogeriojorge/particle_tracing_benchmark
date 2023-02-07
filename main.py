#!/usr/bin/env python
import os
import time
import glob
import shutil
import inputs
import numpy as np
from mpi4py import MPI
from pathlib import Path
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from simsopt.util import MpiPartition
from create_figures import create_figures_func
from neat.fields import Simple, Vmec as Vmec_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit
mpi = MpiPartition(MPI.COMM_WORLD.Get_size())
def pprint(*args, **kwargs):
    if mpi.proc0_world: print(*args, **kwargs)
#### Only loading pyQSC field on processor 0 for efficiency
particle_phi_initial = inputs.varphi_initial
if mpi.proc0_world:
    from neat.fields import StellnaQS
    from scipy.interpolate import CubicSpline as spline
    field_nearaxis = StellnaQS(rc=inputs.rc*inputs.Rmajor_ARIES, zs=inputs.zs*inputs.Rmajor_ARIES, etabar=inputs.eta_bar/inputs.Rmajor_ARIES, B2c=inputs.B2c*(inputs.b0_ARIES/inputs.Rmajor_ARIES/inputs.Rmajor_ARIES), B0=inputs.b0_ARIES, nfp=inputs.nfp, order='r3', nphi=111)
    nu_of_varphi = spline(np.append(field_nearaxis.varphi,2*np.pi/field_nearaxis.nfp), np.append(field_nearaxis.varphi-field_nearaxis.phi,0), bc_type='periodic')
    particle_phi_initial = field_nearaxis.to_RZ([[inputs.r_initial,inputs.theta_initial,inputs.varphi_initial-nu_of_varphi(inputs.varphi_initial)]])[2][0]
mpi.comm_world.bcast(particle_phi_initial, root=0)
#######################################################
pprint("Starting patricle_tracing benchmark")
# Initialize folders and variables
particle = ChargedParticle(r_initial=inputs.r_initial, theta_initial=inputs.theta_initial, phi_initial=inputs.varphi_initial, Lambda=inputs.Lambda, charge=inputs.charge)
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'out_constantb20{inputs.constant_b20}_r{inputs.r_initial:.2f}_theta{inputs.theta_initial:.2f}_phi{inputs.varphi_initial:.2f}_lambda{inputs.Lambda:.2f}_nsVMEC{np.max(inputs.ns_array)}')
os.makedirs(OUT_DIR, exist_ok=True)
if mpi.proc0_world: shutil.copyfile(os.path.join(Path(__file__).parent.resolve(),'inputs.py'), os.path.join(OUT_DIR,'copy_inputs.py'))
os.chdir(OUT_DIR)
# Near-axis orbit
start_time = time.time()
orbit_nearaxis_solution=np.empty((inputs.nsamples+1,12));orbit_nearaxis_rpos_cylindrical=np.empty((3,inputs.nsamples+1))
if mpi.proc0_world:
    orbit_nearaxis = ParticleOrbit(particle, field_nearaxis, nsamples=inputs.nsamples, tfinal=inputs.tfinal, constant_b20=inputs.constant_b20)
    orbit_nearaxis_solution = orbit_nearaxis.solution
    orbit_nearaxis_rpos_cylindrical = orbit_nearaxis.rpos_cylindrical
    orbit_nearaxis.plot_orbit_contourB(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_contourB_nearaxis.pdf'))
    orbit_nearaxis.plot(show=False, savefig=os.path.join(OUT_DIR,f'plot_nearaxis.pdf'))
    orbit_nearaxis.plot_orbit(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_nearaxis.pdf'))
    orbit_nearaxis.plot_orbit_3d(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_3d_nearaxis.pdf'))
    plt.close()
mpi.comm_world.Bcast(orbit_nearaxis_solution, root=0);mpi.comm_world.Bcast(orbit_nearaxis_rpos_cylindrical, root=0)
pprint(f"Particle tracer gyronimo for near-axis (stellna) took {(time.time() - start_time):.2f}s")
orbit_gyronimo_solution_array=np.empty((inputs.n_minor_radius,inputs.nsamples+1,15));orbit_gyronimo_rpos_cylindrical_array=np.empty((inputs.n_minor_radius,3,inputs.nsamples+1))
orbit_simple_solution_array=np.empty((inputs.n_minor_radius,inputs.nsamples+1,15));orbit_simple_rpos_cylindrical_array=np.empty((inputs.n_minor_radius,3,inputs.nsamples+1))
orbit_gyronimo_solution=np.empty((inputs.nsamples+1,15));orbit_gyronimo_rpos_cylindrical=np.empty((3,inputs.nsamples+1))
orbit_simple_solution=np.empty((inputs.nsamples+1,15));orbit_simple_rpos_cylindrical=np.empty((3,inputs.nsamples+1))
# def remove_rows(solution_array):
    # new_solution = []
    # for i, solution in enumerate(solution_array[:,:,1]):
    #     idx = np.flatnonzero((solution == 0) | (solution > inputs.maximum_s_particle))
    #     if idx.any(): new_solution.append(solution_array[i, :idx[0], :])
    #     else: new_solution.append(solution_array[i, :, :])
    # return new_solution
# Calculate SIMPLE and Gyronimo orbits
aspect_ratio_array = []
for i, minor_radius in enumerate(inputs.minor_radius_array):
    pprint(f"Running minor_radius = {minor_radius:.2f}")
    vmec_input = os.path.join(OUT_DIR, f'input.na_A{minor_radius:.2f}')
    start_time = time.time()
    if mpi.proc0_world: field_nearaxis.to_vmec(filename=vmec_input,r=minor_radius, params={"ntor":7, "mpol":7, "ns_array":inputs.ns_array,"niter_array":[1000,3000,7000],"ftol_array":inputs.ftol_array}, ntheta=14, ntorMax=7)
    pprint(f"  Creating VMEC input took {(time.time() - start_time):.2f}s")
    vmec = Vmec(vmec_input, verbose=False)
    start_time = time.time()
    vmec.run()
    aspect_ratio_array.append(vmec.aspect())
    pprint(f"  VMEC ran in {(time.time() - start_time):.2f}s")
    vmec_NEAT = Vmec_NEAT(wout_filename=vmec.output_file, maximum_s=inputs.maximum_s_particle)
    particle.r_initial = (inputs.r_initial**2)/(minor_radius**2) # keep initializing the particle at the same location as the near-axis one
    particle.phi_initial = particle_phi_initial
    particle.theta_initial = np.pi-inputs.theta_initial
    particle.vpp_sign = 1
    if mpi.proc0_world:
        orbit_gyronimo = ParticleOrbit(particle, vmec_NEAT, nsamples=inputs.nsamples, tfinal=inputs.tfinal, add_zeros=True)
        orbit_gyronimo_solution = orbit_gyronimo.solution
        orbit_gyronimo_rpos_cylindrical = orbit_gyronimo.rpos_cylindrical
        orbit_gyronimo.plot_orbit_contourB(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_contourB_gyronimo_aspect{vmec.aspect():.2f}.pdf'))
        orbit_gyronimo.plot(show=False, savefig=os.path.join(OUT_DIR,f'plot_gyronimo_aspect{vmec.aspect():.2f}.pdf'))
        orbit_gyronimo.plot_orbit(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_gyronimo_aspect{vmec.aspect():.2f}.pdf'))
        orbit_gyronimo.plot_orbit_3d(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_3d_gyronimo_aspect{vmec.aspect():.2f}.pdf'))
        plt.close()
    mpi.comm_world.Bcast(orbit_gyronimo_solution, root=0);mpi.comm_world.Bcast(orbit_gyronimo_rpos_cylindrical, root=0)
    orbit_gyronimo_solution_array[i]=orbit_gyronimo_solution
    orbit_gyronimo_rpos_cylindrical_array[i]=orbit_gyronimo_rpos_cylindrical
    pprint(f"  Particle tracer gyronimo for minor_radius = {minor_radius:.2f} took {(time.time() - start_time):.2f}s")
    field_simple = Simple(wout_filename=vmec.output_file, B_scale=1, Aminor_scale=1, multharm=3,ns_s=3,ns_tp=3)
    particle.theta_initial = inputs.theta_initial
    start_time = time.time()
    particle.vpp_sign = -1
    if mpi.proc0_world:
        orbit_simple = ParticleOrbit(particle, field_simple, nsamples=inputs.nsamples, tfinal=inputs.tfinal, add_zeros=True)
        orbit_simple_solution = orbit_simple.solution
        orbit_simple_rpos_cylindrical = orbit_simple.rpos_cylindrical
        orbit_simple.plot_orbit_contourB(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_contourB_simple_aspect{vmec.aspect():.2f}.pdf'))
        orbit_simple.plot(show=False, savefig=os.path.join(OUT_DIR,f'plot_simple_aspect{vmec.aspect():.2f}.pdf'))
        orbit_simple.plot_orbit(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_simple_aspect{vmec.aspect():.2f}.pdf'))
        orbit_simple.plot_orbit_3d(show=False, savefig=os.path.join(OUT_DIR,f'plot_orbit_3d_simple_aspect{vmec.aspect():.2f}.pdf'))
        plt.close()
    mpi.comm_world.Bcast(orbit_simple_solution, root=0);mpi.comm_world.Bcast(orbit_simple_rpos_cylindrical, root=0)
    orbit_simple_solution_array[i]=orbit_simple_solution
    orbit_simple_rpos_cylindrical_array[i]=orbit_simple_rpos_cylindrical
    pprint(f"  Particle tracer simple for minor_radius = {minor_radius:.2f} took {(time.time() - start_time):.2f}s")

orbit_gyronimo_solution_array[(orbit_gyronimo_solution_array[:,:,1] == 0) | (orbit_gyronimo_solution_array[:,:,1] > inputs.maximum_s_particle),:] = np.nan
orbit_simple_solution_array[(orbit_simple_solution_array[:,:,1] == 0) | (orbit_simple_solution_array[:,:,1] > inputs.maximum_s_particle),:] = np.nan
# Analyze results
if mpi.proc0_world:
    create_figures_func(inputs, pprint, aspect_ratio_array,
        orbit_simple_rpos_cylindrical_array, orbit_gyronimo_rpos_cylindrical_array, orbit_nearaxis_rpos_cylindrical,
        orbit_simple_solution_array, orbit_gyronimo_solution_array, orbit_nearaxis_solution, show=True, savefig=True)

    for objective_file in glob.glob(os.path.join(OUT_DIR,f"input.*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"wout_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"threed1*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"parvmecinfo*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"healaxis*")): os.remove(objective_file)
    # for objective_file in glob.glob(os.path.join(OUT_DIR,f"fort*")): os.remove(objective_file)