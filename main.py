#!/usr/bin/env python
import os
import time
import numpy as np
from mpi4py import MPI
from pathlib import Path
from simsopt.util import MpiPartition
from neat.fields import Simple, Vmec as Vmec_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit
from simsopt.mhd import Vmec
import inputs
mpi = MpiPartition(MPI.COMM_WORLD.Get_size())
def pprint(*args, **kwargs):
    if mpi.proc0_world: print(*args, **kwargs)
pprint("Starting patricle_tracing benchmark")
# Initialize folders and variables
particle = ChargedParticle(r_initial=inputs.r_initial, theta_initial=inputs.theta_initial, phi_initial=inputs.varphi_initial, Lambda=inputs.Lambda)
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'out_constantb20{inputs.constant_b20}_r{inputs.r_initial}_theta{inputs.theta_initial}_phi{inputs.varphi_initial}_lambda{inputs.Lambda}')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
# Near-axis orbit
start_time = time.time()
orbit_nearaxis_solution=np.empty((inputs.nsamples+1,12));orbit_nearaxis_rpos_cylindrical=np.empty((3,inputs.nsamples+1))
if mpi.proc0_world:
    orbit_nearaxis = ParticleOrbit(particle, inputs.field_nearaxis, nsamples=inputs.nsamples, tfinal=inputs.tfinal, constant_b20=inputs.constant_b20)
    orbit_nearaxis_solution = orbit_nearaxis.solution
    orbit_nearaxis_rpos_cylindrical = orbit_nearaxis.rpos_cylindrical
mpi.comm_world.Bcast(orbit_nearaxis_solution, root=0);mpi.comm_world.Bcast(orbit_nearaxis_rpos_cylindrical, root=0)
pprint(f"Particle tracer simple for near-axis took {(time.time() - start_time):.2f}s")
orbit_gyronimo_solution_array=np.empty((inputs.n_minor_radius,inputs.nsamples+1,15));orbit_gyronimo_rpos_cylindrical_array=np.empty((inputs.n_minor_radius,3,inputs.nsamples+1))
orbit_simple_solution_array=np.empty((inputs.n_minor_radius,inputs.nsamples+1,15));orbit_simple_rpos_cylindrical_array=np.empty((inputs.n_minor_radius,3,inputs.nsamples+1))
orbit_gyronimo_solution=np.empty((inputs.nsamples+1,15));orbit_gyronimo_rpos_cylindrical=np.empty((3,inputs.nsamples+1))
orbit_simple_solution=np.empty((inputs.nsamples+1,15));orbit_simple_rpos_cylindrical=np.empty((3,inputs.nsamples+1))
# Calculate SIMPLE and Gyronimo orbits
minor_radius_array = np.linspace(inputs.r_min,inputs.r_ARIES,inputs.n_minor_radius)
for i, minor_radius in enumerate(minor_radius_array):
    pprint(f"Running minor_radius = {minor_radius:.2f}")
    vmec_input = os.path.join(OUT_DIR, f'input.na_A{minor_radius:.2f}')
    start_time = time.time()
    if mpi.proc0_world: inputs.field_nearaxis.to_vmec(filename=vmec_input,r=minor_radius, params={"ntor":7, "mpol":7, "ns_array":[16,51,101],"niter_array":[1000,2500,5000],"ftol_array":[1e-12,1e-14,1e-14]}, ntheta=14, ntorMax=7)
    pprint(f"  Creating VMEC input took {(time.time() - start_time):.2f}s")
    vmec = Vmec(vmec_input, verbose=False)
    start_time = time.time()
    vmec.run()
    pprint(f"  VMEC ran in {(time.time() - start_time):.2f}s")
    vmec_NEAT = Vmec_NEAT(wout_filename=vmec.output_file, maximum_s=inputs.maximum_s_gyronimo)
    particle.r_initial = (inputs.r_initial**2)/(minor_radius**2) # keep initializing the particle at the same location as the near-axis one
    from scipy.interpolate import CubicSpline as spline
    nu_of_varphi=spline(np.append(inputs.field_nearaxis.varphi,2*np.pi/inputs.field_nearaxis.nfp), np.append(inputs.field_nearaxis.varphi-inputs.field_nearaxis.phi,0), bc_type='periodic')
    particle.phi_initial = inputs.field_nearaxis.to_RZ([[inputs.r_initial,inputs.theta_initial,inputs.varphi_initial-nu_of_varphi(inputs.varphi_initial)]])[2][0]
    if mpi.proc0_world:
        orbit_gyronimo = ParticleOrbit(particle, vmec_NEAT, nsamples=inputs.nsamples, tfinal=inputs.tfinal)
        orbit_gyronimo_solution = orbit_gyronimo.solution
        orbit_gyronimo_rpos_cylindrical = orbit_gyronimo.rpos_cylindrical
    mpi.comm_world.Bcast(orbit_gyronimo_solution, root=0);mpi.comm_world.Bcast(orbit_gyronimo_rpos_cylindrical, root=0)
    orbit_gyronimo_solution_array[i]=orbit_gyronimo_solution
    orbit_gyronimo_rpos_cylindrical_array[i]=orbit_gyronimo_rpos_cylindrical
    pprint(f"  Particle tracer gyronimo for minor_radius = {minor_radius:.2f} took {(time.time() - start_time):.2f}s")
    field_simple = Simple(wout_filename=vmec.output_file, B_scale=1, Aminor_scale=1, multharm=3,ns_s=3,ns_tp=3)
    start_time = time.time()
    if mpi.proc0_world:
        orbit_simple = ParticleOrbit(particle, field_simple, nsamples=inputs.nsamples, tfinal=inputs.tfinal)
        orbit_simple_solution = orbit_simple.solution
        orbit_simple_rpos_cylindrical = orbit_simple.rpos_cylindrical
    mpi.comm_world.Bcast(orbit_simple_solution, root=0);mpi.comm_world.Bcast(orbit_simple_rpos_cylindrical, root=0)
    orbit_simple_solution_array[i]=orbit_simple_solution
    orbit_simple_rpos_cylindrical_array[i]=orbit_simple_rpos_cylindrical
    pprint(f"  Particle tracer simple for minor_radius = {minor_radius:.2f} took {(time.time() - start_time):.2f}s")
# Analyze results
import matplotlib.pyplot as plt
if mpi.proc0_world:
    fig, ax = plt.subplots()
    plt.plot(minor_radius_array,orbit_simple_rpos_cylindrical_array[:,0,0],label='SIMPLE')
    plt.plot(minor_radius_array,orbit_gyronimo_rpos_cylindrical_array[:,0,0],label='gyronimo')
    ax.axhline(y=orbit_nearaxis_rpos_cylindrical[0,0], color='black', lw=2, label='Near-Axis')
    plt.xlabel('Minor Radius of the Plasma Boundary')
    plt.ylabel('Initial R')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(minor_radius_array,orbit_simple_rpos_cylindrical_array[:,1,0],label='SIMPLE')
    plt.plot(minor_radius_array,orbit_gyronimo_rpos_cylindrical_array[:,1,0],label='gyronimo')
    ax.axhline(y=orbit_nearaxis_rpos_cylindrical[1,0], color='black', lw=2, label='Near-Axis')
    plt.xlabel('Minor Radius of the Plasma Boundary')
    plt.ylabel('Initial Z')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(minor_radius_array,orbit_simple_rpos_cylindrical_array[:,2,0],label='SIMPLE')
    plt.plot(minor_radius_array,orbit_gyronimo_rpos_cylindrical_array[:,2,0],label='gyronimo')
    ax.axhline(y=orbit_nearaxis_rpos_cylindrical[2,0], color='black', lw=2, label='Near-Axis')
    plt.xlabel('Minor Radius of the Plasma Boundary')
    plt.ylabel(f'Initial $\phi$')
    plt.legend()
    
    fig, ax = plt.subplots()
    for i, minor_radius in enumerate(minor_radius_array):
        if i==0: label_legends=['SIMPLE','gyronimo']
        else: label_legends=['_nolegend_','_nolegend_']
        plt.plot(orbit_simple_solution_array[i, :, 0], np.sqrt(orbit_simple_solution_array[i,:,1])*minor_radius, '--b', label=label_legends[0])
        plt.plot(orbit_gyronimo_solution_array[i, :, 0], np.sqrt(orbit_gyronimo_solution_array[i,:,1])*minor_radius, ':r', label=label_legends[1])
    plt.plot(orbit_nearaxis_solution[:, 0], orbit_nearaxis_solution[:,1], label='Near-Axis')
    plt.xlabel('time')
    plt.ylabel(r'$r_{\mathrm{near-axis}}$')
    plt.legend()

    plt.show()