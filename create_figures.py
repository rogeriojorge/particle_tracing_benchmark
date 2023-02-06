import numpy as np
import matplotlib.pyplot as plt

def create_figures_func(inputs, pprint, aspect_ratio_array,
        orbit_simple_rpos_cylindrical_array, orbit_gyronimo_rpos_cylindrical_array, orbit_nearaxis_rpos_cylindrical,
        orbit_simple_solution_array, orbit_gyronimo_solution_array, orbit_nearaxis_solution, show=True, savefig=True):

    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle('Single-Particle Initial Conditions in Real Space')
    axs[0].axhline(y=orbit_nearaxis_rpos_cylindrical[0,0], color='black', lw=2, label='Near-Axis')
    axs[0].plot(aspect_ratio_array,orbit_simple_rpos_cylindrical_array[:,0,0], 'b-.', lw=2, label='SIMPLE')
    axs[0].plot(aspect_ratio_array,orbit_gyronimo_rpos_cylindrical_array[:,0,0], 'r:', lw=2, label='gyronimo')
    axs[0].set_ylabel(r'Initial $R$')
    axs[0].legend()
    axs[1].axhline(y=orbit_nearaxis_rpos_cylindrical[1,0], color='black', lw=2, label='Near-Axis')
    axs[1].plot(aspect_ratio_array,orbit_simple_rpos_cylindrical_array[:,1,0], 'b-.', lw=2,label='SIMPLE')
    axs[1].plot(aspect_ratio_array,orbit_gyronimo_rpos_cylindrical_array[:,1,0], 'r:', lw=2,label='gyronimo')
    axs[1].set_ylabel(r'Initial $Z$')
    axs[2].axhline(y=orbit_nearaxis_rpos_cylindrical[2,0], color='black', lw=2, label='Near-Axis')
    axs[2].plot(aspect_ratio_array,orbit_simple_rpos_cylindrical_array[:,2,0], 'b-.', lw=2,label='SIMPLE')
    axs[2].plot(aspect_ratio_array,orbit_gyronimo_rpos_cylindrical_array[:,2,0], ':', lw=2,label='gyronimo')
    axs[2].set_ylabel(r'Initial $\phi$')
    plt.xlabel('Aspect Ratio of the Plasma Boundary')
    if savefig: plt.savefig('particle_initial_conditions_RZphi.png')

    # fig, ax = plt.subplots()
    # np.array(orbit_simple_rpos_cylindrical_array,dtype=object)[:,0,7]
    # plt.plot(inputs.minor_radius_array,orbit_simple_rpos_cylindrical_array[:,0,7],label='SIMPLE')
    # plt.plot(inputs.minor_radius_array,orbit_gyronimo_rpos_cylindrical_array[:,0,7],label='gyronimo')
    # ax.axhline(y=orbit_nearaxis_rpos_cylindrical[1,0], color='black', lw=2, label='Near-Axis')
    # plt.xlabel('Minor Radius of the Plasma Boundary')
    # plt.ylabel('Initial vparallel')
    # plt.legend()

    fig, ax = plt.subplots()
    for i, minor_radius in enumerate(inputs.minor_radius_array):
        if i==0: label_legends=['SIMPLE','gyronimo']
        else: label_legends=['_nolegend_','_nolegend_']
        plt.plot(orbit_simple_solution_array[i][:,0], np.sqrt(orbit_simple_solution_array[i][:,1])*minor_radius, '--b', label=label_legends[0])
        plt.plot(orbit_gyronimo_solution_array[i][:,0], np.sqrt(orbit_gyronimo_solution_array[i][:,1])*minor_radius, ':r', label=label_legends[1])
    plt.plot(orbit_nearaxis_solution[:,0], orbit_nearaxis_solution[:,1], label='Near-Axis')
    plt.xlabel('time')
    plt.ylabel(r'$r_{\mathrm{near-axis}}$')
    plt.legend()
    if savefig: plt.savefig('r_nearaxis_comparison.png')

    fig, ax = plt.subplots()
    for i, minor_radius in enumerate(inputs.minor_radius_array):
        if i==0: label_legends=['SIMPLE','gyronimo']
        else: label_legends=['_nolegend_','_nolegend_']
        plt.plot(orbit_simple_solution_array[i][:,0], orbit_simple_solution_array[i][:,7], '--b', label=label_legends[0])
        plt.plot(orbit_gyronimo_solution_array[i][:,0], orbit_gyronimo_solution_array[i][:,7], ':r', label=label_legends[1])
    plt.plot(orbit_nearaxis_solution[:,0], orbit_nearaxis_solution[:,7], label='Near-Axis')
    plt.xlabel('time')
    plt.ylabel(r'$v_\parallel$')
    plt.legend()
    if savefig: plt.savefig('vparallel_nearaxis_comparison.png')

    fig, ax = plt.subplots()
    for i, minor_radius in enumerate(inputs.minor_radius_array):
        if i==0: label_legends=['SIMPLE','gyronimo']
        else: label_legends=['_nolegend_','_nolegend_']
        # plt.plot(orbit_simple_solution_array[i][:,0], orbit_simple_solution_array[i][:,6], '--b', label=label_legends[0])
        plt.plot(orbit_gyronimo_solution_array[i][:,0], orbit_gyronimo_solution_array[i][:,6], ':r', label=label_legends[1])
    plt.plot(orbit_nearaxis_solution[:,0], orbit_nearaxis_solution[:,6], label='Near-Axis')
    plt.xlabel('time')
    plt.ylabel(r'$|B|$')
    plt.legend()
    if savefig: plt.savefig('modB_nearaxis_comparison.png')

    fig, ax = plt.subplots()
    for i, minor_radius in enumerate(inputs.minor_radius_array):
        if i==0: label_legends=['SIMPLE','gyronimo']
        else: label_legends=['_nolegend_','_nolegend_']
        plt.plot(orbit_simple_solution_array[i][:,0], orbit_simple_solution_array[i][:,1], '--b', label=label_legends[0])
        plt.plot(orbit_gyronimo_solution_array[i][:,0], orbit_gyronimo_solution_array[i][:,1], ':r', label=label_legends[1])
    plt.xlabel('time')
    plt.ylabel(r'$s=\psi/\psi_a$')
    plt.legend()
    if savefig: plt.savefig('s_nearaxis_comparison.png')

    pprint(f'Initial vparallel nearaxis = {orbit_nearaxis_solution[0,7]}')
    pprint(f'Initial |B| nearaxis = {orbit_nearaxis_solution[0,6]}')
    pprint(f'Initial r nearaxis = {orbit_nearaxis_solution[0,1]}')
    for i, minor_radius in enumerate(inputs.minor_radius_array):
        pprint(f'Minor radius = {minor_radius:.2f} (s={((inputs.r_initial**2)/(minor_radius**2)):.2f})')
        pprint(f'  vparallel gyronimo = {orbit_gyronimo_solution_array[i][0,7]}')
        pprint(f'  vparallel simple = {orbit_simple_solution_array[i][0,7]}')
        pprint(f'  |B| gyronimo = {orbit_gyronimo_solution_array[i][0,6]}')
        pprint(f'  r gyronimo = {np.sqrt(orbit_gyronimo_solution_array[i][0,1])*minor_radius}')
        pprint(f'  r simple = {np.sqrt(orbit_simple_solution_array[i][0,1])*minor_radius}')
        pprint(f'  s gyronimo = {orbit_gyronimo_solution_array[i][0,1]}')
        pprint(f'  s simple = {orbit_simple_solution_array[i][0,1]}')

    # plt.show()