#!/usr/bin/env python
import os
import time
import numpy as np
from mpi4py import MPI
from pathlib import Path
from simsopt.util import MpiPartition
from neat.fields import StellnaQS, Simple, Vmec as Vmec_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit
from simsopt.mhd import Vmec
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:  # only pprint on rank 0
        print(*args, **kwargs)
mpi = MpiPartition(MPI.COMM_WORLD.Get_size())
"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""
# Initialize an alpha particle at a radius = r_initial
aspect_ratio_array = [10] # 1/r_max (near-axis r)
r_initial = 0.06  # meters
theta_initial = 0 * np.pi / 2  # initial Boozer poloidal angle
phi_initial = np.pi  # initial cylindrical toroidal angle
B0 = 5.7  # Tesla, mean magnetic field on-axis
Lambda = 0.883  # = mu * B0 / energy
nsamples = 1000  # resolution in time
tfinal = 4e-5  # seconds
stell_index = 4
constant_b20 = True  # use a constant B20 (mean value) or the real function
# Initialize folders and variables
particle = ChargedParticle(r_initial=r_initial, theta_initial=theta_initial, phi_initial=phi_initial, Lambda=Lambda)
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'out_stell{stell_index}_constantb20{constant_b20}_r{r_initial}_theta{theta_initial}_phi{phi_initial}_lambda{Lambda}')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
# Near-axis calculation
field_nearaxis = StellnaQS.from_paper(stell_index, B0=B0, nphi=71)
start_time = time.time()
if mpi.proc0_world:
    orbit_nearaxis = ParticleOrbit(particle, field_nearaxis, nsamples=nsamples, tfinal=tfinal, constant_b20=constant_b20)
pprint(f"Particle tracer simple for near-axis took {(time.time() - start_time):.2f}s")
# SIMPLE calculation
for aspect_ratio in aspect_ratio_array:
    pprint(f"Running aspect_ratio = {aspect_ratio}")
    vmec_input = os.path.join(OUT_DIR, f'input.na_stell{stell_index}_A{aspect_ratio}')
    start_time = time.time()
    if mpi.proc0_world: field_nearaxis.to_vmec(filename=vmec_input,r=1/aspect_ratio, params={"ntor":5, "mpol":5, "ns_array":[16,51],"niter_array":[1000,2000],"ftol_array":[1e-12,1e-14]}, ntheta=10, ntorMax=5)
    pprint(f"  Creating VMEC input took {(time.time() - start_time):.2f}s")
    vmec = Vmec(vmec_input, verbose=False)
    start_time = time.time()
    vmec.run()
    pprint(f"  VMEC ran in {(time.time() - start_time):.2f}s")
    # # Gyronimo Field
    # vmec_NEAT = Vmec_NEAT(wout_filename=vmec.output_file)
    # if mpi.proc0_world: orbit_gyronimo = ParticleOrbit(particle, vmec_NEAT, nsamples=nsamples, tfinal=tfinal)
    # pprint(f"  Particle tracer gyronimo for aspect_ratio = {aspect_ratio} took {(time.time() - start_time):.2f}s")
    # Simple Field
    field_simple = Simple(wout_filename=vmec.output_file, B_scale=1, Aminor_scale=1, multharm=3,ns_s=3,ns_tp=3)
    start_time = time.time()
    if mpi.proc0_world: orbit_simple = ParticleOrbit(particle, field_simple, nsamples=nsamples, tfinal=tfinal)
    pprint(f"  Particle tracer simple for aspect_ratio = {aspect_ratio} took {(time.time() - start_time):.2f}s")
