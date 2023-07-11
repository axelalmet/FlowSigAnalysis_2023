### Script to simulate the evolution of a ligand-receptor model. As a first step, we're considering just one ligand, one receptor,
### and one complex on a network.
import keyword
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy import integrate
import numpy as np
from itertools import count, product
from timeit import default_timer as timer
from numbalsoda import lsoda_sig, lsoda, address_as_void_pointer
from numba import njit, cfunc, carray, types
import numba as nb
import pickle
import seaborn as sns

# Set a bunch of parameters that we need
num_variables = 5
pert_scale = 10.0
knockout_scale = 1e-3

# ODE parameters (for the solver evaluation)
end_time = 10.0
num_timepoints = 2
num_realisations = 5

# SHH parameters
diff_shh = 100.0 # Diffusion of SHH
prod_scale_shh = 0.1 #Production multiplier
bind_shh = 2.0  # Binding rate
beta_shh = 0.1 # Degradation rate

# BMP4 parameters
diff_bmp4 = 100.0 # Diffusion of SHH
prod_scale_bmp4 = 0.1 #Production multiplier
alpha_bmp4 = 2.0
beta_bmp4 = 0.1 # Degradation rate

# FOXF1 TF parameters
K_shh_foxf1  = 25.0 # Binding affinity of SHH to GLI
K_foxf1_bmp4 = 10.0
alpha_foxf1 = 2.0 # Production rate of foxf1
beta_foxf1 = 2.0 # Degradation rate of foxf1

bws_ligand = [30.0, 70.0] # How we initialise the ligand distribution

# Some helper functions
def set_graph_attributes(G, prods, lig='shh'): # Really only need to set the production rate of Shh

    # Add the Graph attributes
    index = 0
    for node in G.nodes():

        i, j = node
        pos = np.array([10*i, 10*j])
        G.nodes[i, j]['pos'] = pos
        G.nodes[i, j]['index'] = index
        G.nodes[i, j]['prod_' + lig]  = prods[index] # Auto-production term of SHH

        index += 1

def set_ode_parameters(G, initial_states, domain_height, domain_width, bws_ligand):

    num_nodes = len(G.nodes())

    # Define SHH first
    shh_x, shh_r = np.random.rand(2)

    shh_r = bws_ligand[0] + (bws_ligand[1] - bws_ligand[0])*shh_r # Adjust the blob widths of the ligand and receptor
    shh_x *= (domain_width / 6.0)  # Scale so that Shh initially sits on the left 1/6

    # Define BMP4 now
    bmp4_x, bmp4_r = np.random.rand(2)

    bmp4_r = bws_ligand[0] + (bws_ligand[1] - bws_ligand[0])*bmp4_r # Adjust the blob widths of the ligand and receptor
    bmp4_x *= (domain_width / 6.0)  # Scale so that Shh initially sits on the left 1/6
    bmp4_x += 5.0 * (domain_width / 6.0)

    for node in G.nodes():

        i, j = node

        # Get the cell type and position
        position = G.nodes[i, j]['pos']
        index = G.nodes[i, j]['index']

        # Set the ligand, receptor, and complex concentration for this node
        shh = np.exp( -(position[0] - shh_x)**2.0 / (2.0 * shh_r)**2.0)
        bmp4 = np.exp( -(position[0] - bmp4_x)**2.0 / (2.0 * bmp4_r)**2.0)

        # Shh is produced at the left
        if position[0] < domain_width / 6.0:
            shh = 1.0

        # Bmp4 is produced at the right
        if position[0] > 5.0 * domain_width / 6.0:
            bmp4 = 1.0

        shh_rec = np.random.random(1) # Uniformly random receptor expression? (Could pick something else)

        # Set the complex to be zero always
        shh_bound = 0.0

        # foxf1 should be 0 while bmp4 should be 1
        foxf1 = 0.0

        # We should be able to set the initial state now
        initial_states[num_variables * index] = shh
        initial_states[num_variables * index + 1] = shh_rec
        initial_states[num_variables * index+ 2] = shh_bound
        initial_states[num_variables * index + 3] = foxf1
        initial_states[num_variables * index + 4] = bmp4

        G.nodes[i, j]['SHH'] = shh
        G.nodes[i, j]['PTCH1'] = shh_rec
        G.nodes[i, j]['SHH_BOUND'] = shh_bound
        G.nodes[i, j]['FOXF1'] = foxf1
        G.nodes[i, j]['BMP4'] = bmp4

def set_ode_sol_at_time(G, time, all_timepoints, sol):

    timeIndex = np.where(all_timepoints == time)[0][0]  
    node_pairs = list(G.nodes())

    for node in node_pairs:

        i, j = node
        index = G.nodes[i, j]['index']

        G.nodes[i, j]['SHH'] = float(sol[timeIndex, num_variables * index])
        G.nodes[i, j]['PTCH1'] = float(sol[timeIndex, num_variables * index + 1])
        G.nodes[i, j]['SHH_BOUND'] = float(sol[timeIndex, num_variables * index + 2])
        G.nodes[i, j]['FOXF1'] = float(sol[timeIndex, num_variables * index + 3])
        G.nodes[i, j]['BMP4'] = float(sol[timeIndex, num_variables * index + 4])
        
def plot_ode_sols(G, ode_sol_matrices, fig_width, fig_height, v_min, v_max):

    plot_count = 0
    node_coordinates = dict(G.nodes(data='pos')) # Store as node coordinates for later
    state_names = ['SHH', 'PTCH1', 'SHH_BOUND', 'FOXF1', 'BMP4']

    num_rows = 3
    num_cols = 2
    # Initialise figure
    fig, axs = plt.subplots(3, 3)

    for i, j in product(range(num_rows), range(num_cols)):
        
        if plot_count < len(state_names):
            state = state_names[plot_count]

            node_values = set(nx.get_node_attributes(G, state).values())
            mapping = dict(zip(sorted(node_values), count()))
            nodes = G.nodes()
            colours = [mapping[G.nodes[node][state]] for node in nodes]
            colours_norm = plt.Normalize(min(colours), max(colours))

            sns.heatmap(ode_sol_matrices[state].T, cmap='Spectral_r', ax=axs[i, j]).set(title=state)
            # ec = nx.draw_networkx_edges(G, node_coordinates, alpha = 0.2)
            # nc = nx.draw_networkx_nodes(G, node_coordinates, nodelist=nodes, node_color = colours, node_size=10, cmap = plt.cm.Spectral_r, vmin=v_min, vmax=v_max)
            # plt.colorbar(nc)
            plt.axis('off')
            plot_count += 1 # Update plot count

    plt.show() 

def construct_ode_sol(G, times, ode_sol):

    num_times = len(times)
    
    node_pairs = list(G.nodes())
    num_nodes = len(node_pairs)

    # Number of columns comes from: time, x, y, Shh, Ptch1, Bound_Shh, Foxf1, Bmp4 (x N)
    output_sol = np.zeros((num_times*num_nodes, 3 + num_variables))

    # Set positions
    for t in range(num_times):

        time = times[t]


        for node in G.nodes():

            i, j = node
            index = G.nodes[i, j]['index']
            position = G.nodes[i, j]['pos']

            output_sol[t*num_nodes + index, 0] = time
            output_sol[t*num_nodes + index, 1] = position[0].copy()
            output_sol[t*num_nodes + index, 2] = position[1].copy()
            output_sol[t*num_nodes + index, 3] = ode_sol[t, num_variables * index]
            output_sol[t*num_nodes + index, 4] = ode_sol[t, num_variables * index + 1]
            output_sol[t*num_nodes + index, 5] = ode_sol[t, num_variables * index+ 2]
            output_sol[t*num_nodes + index, 6] = ode_sol[t, num_variables * index + 3]
            output_sol[t*num_nodes + index, 7] = ode_sol[t, num_variables * index + 4]

    return output_sol

def construct_ode_sol_matrices_at_time(G, time, all_timepoints, sol):

    timeIndex = np.where(all_timepoints == time)[0][0]  
    node_pairs = list(G.nodes())

    state_matrices = {}
    state_matrices['SHH'] = np.zeros((num_cols, num_rows))
    state_matrices['PTCH1'] = np.zeros((num_cols, num_rows))
    state_matrices['SHH_BOUND'] = np.zeros((num_cols, num_rows))
    state_matrices['FOXF1'] = np.zeros((num_cols, num_rows))
    state_matrices['BMP4'] = np.zeros((num_cols, num_rows))

    for node in node_pairs:
        
        state_sol = np.zeros((num_cols, num_rows))
        i, j = node
        index = G.nodes[i, j]['index']

        state_matrices['SHH'][i, j] = float(sol[timeIndex, num_variables * index])
        state_matrices['PTCH1'][i, j] = float(sol[timeIndex, num_variables * index + 1])
        state_matrices['SHH_BOUND'][i, j] = float(sol[timeIndex, num_variables * index + 2])
        state_matrices['FOXF1'][i, j]  = float(sol[timeIndex, num_variables * index + 3])
        state_matrices['BMP4'][i, j]  = float(sol[timeIndex, num_variables * index + 4])

    return state_matrices


### Set-up for analysis
simulation_path = '../output/'

# Generate a 2D point
num_rows = 10
num_cols = 90
num_nodes = num_rows*num_cols # Total number of nodes
domain_width = 10.0*num_cols
domain_height = 10.0*num_rows

# Generate 2D grid with 4 nearest neighbours
G = nx.grid_2d_graph(num_cols, num_rows)
D = -1.0 * nx.laplacian_matrix(G).toarray().astype('float64') # Define the Laplacian matrix.
A = nx.adjacency_matrix(G).toarray() # Adjacency

prod_rates_shh = np.zeros(num_nodes)
prod_rates_bmp4 = np.zeros(num_nodes)

# Set all the graph attributes to get the index and positions
set_graph_attributes(G, prod_rates_shh, lig='SHH')
set_graph_attributes(G, prod_rates_bmp4, lig='BMP4')

# Define the node pairs
node_pairs = list(G.nodes())

for node in node_pairs:
    i, j = node
    index = G.nodes[i, j]['index']
    pos = G.nodes[i, j]['pos']

    prod_rates_shh[index] = prod_scale_shh
    prod_rates_bmp4[index] = prod_scale_bmp4

    # Only have a few cells 
    if pos[0] > domain_width / 18.0:
        prod_rates_shh[index] = 0.0

    if pos[0] < 17.0 * domain_width / 18.0:
         prod_rates_bmp4[index] = 0.0
    

set_graph_attributes(G, prod_rates_shh, lig='SHH') # To set teh production rates
set_graph_attributes(G, prod_rates_bmp4, lig='BMP4') # To set teh production rates


# Generate the initial state

def grn_rhs(t, Y, dY, D, params):

    diff_shh, diff_bmp4, bind_shh, \
    beta_shh, K_shh_foxf1, K_foxf1_bmp4,\
    alpha_foxf1, beta_foxf1, alpha_bmp4 = params

    # Helper function to define reaction rates later
    def hill(x):
        return x / (1 + x)
    
    for index in range(num_nodes):

        prod_rate_shh = prod_rates_shh[index]
        prod_rate_bmp4 = prod_rates_bmp4[index]

        # Get the kinetic parameters for the ligand-receptor binding
        shh = Y[num_variables * index] # SHH
        shh_rec = Y[num_variables * index + 1] # PTCH1
        shh_bound = Y[num_variables * index + 2] # Bound complex
        foxf1 = Y[num_variables * index + 3] # FOXF1 
        bmp4 = Y[num_variables * index + 4] # BMP4

        # We want D[index, j] * Shh[j]
        shh_flux = np.sum(np.array([D[index, n]*Y[num_variables*(n)] for n in range(num_nodes)]))

        # We want D[index, j] * Bmp4[j]
        bmp4_flux = np.sum(np.array([D[index, n]*Y[num_variables*(n + 1) - 1] for n in range(num_nodes)]))

        # Ligand-receptor binding
        dY[num_variables * index] = diff_shh * shh_flux\
                                                 - bind_shh * shh * shh_rec\
                                                 + prod_rate_shh\
                                                 - beta_shh * shh # Shh
        dY[num_variables * index + 1] = - bind_shh * shh * shh_rec # Ptch1
        dY[num_variables * index + 2] = bind_shh * shh * shh_rec # Bound complex

        # Activation of Foxf1 TFs due to binding of Shh
        dY[num_variables * index + 3] = alpha_foxf1 * hill(1.0 + K_shh_foxf1 * shh_bound) - beta_foxf1 * foxf1# Gli activator

        # Activation of Bmp4 due to Foxf1 activation
        dY[num_variables * index + 4] = diff_bmp4 * bmp4_flux\
                                        + prod_rate_bmp4\
                                        + alpha_bmp4 * hill(1.0 + K_foxf1_bmp4 * foxf1)\
                                         - beta_bmp4 * bmp4

# Define the argument types of the different parameters. We need to specify the array types and the shape parametesr too
args_dtype = types.Record.make_c_struct([
                    ('D_p', types.int64),
                    ('D_shape_0', types.int64),
                    ('D_shape_1', types.int64), 
                     ('model_params_p', types.int64), 
                     ('model_params_shape_0', types.int64)])

# this function will create the numba function to pass to lsoda.
def create_jit_rhs(rhs, args_dtype):
    jitted_rhs = njit(rhs)
    @nb.cfunc(types.void(types.double,
             types.CPointer(types.double),
             types.CPointer(types.double),
             types.CPointer(args_dtype)))
    def wrapped(t, u, du, user_data_p):
        # unpack p and arr from user_data_p
        user_data = nb.carray(user_data_p, 1)
        D_p = nb.carray(address_as_void_pointer(user_data[0].D_p),(user_data[0].D_shape_0, user_data[0].D_shape_1), dtype=np.float64)
        model_params_p = nb.carray(address_as_void_pointer(user_data[0].model_params_p),(user_data[0].model_params_shape_0,), dtype=np.float64)

        # then we call the jitted rhs function, passing in data
        jitted_rhs(t, u, du, D_p, model_params_p) 
    return wrapped

# JIT the ODE RHS
rhs_cfunc = create_jit_rhs(grn_rhs, args_dtype)

t_eval = np.linspace(0, end_time, num_timepoints)


# np.random.seed(1) # Set a random seed for reproducibility
for seed in range(num_realisations):

    initial_states = np.zeros(num_variables*num_nodes)

    set_ode_parameters(G, initial_states, domain_height, domain_width, bws_ligand)

    model_params = np.array([diff_shh, diff_bmp4, bind_shh, \
                            beta_shh, K_shh_foxf1, K_foxf1_bmp4,\
                            alpha_foxf1, beta_foxf1, alpha_bmp4])

    args_obs = np.array((D.ctypes.data, D.shape[0], D.shape[1], model_params.ctypes.data, model_params.shape[0]),dtype=args_dtype)

    funcptr = rhs_cfunc.address
    ode_sol, success = lsoda(funcptr, initial_states, t_eval, data=args_obs)

    # Save the model parameters
    ode_params = {'seed':seed,
                    'num_rows':num_rows,
                    'num_cols':num_cols,
                    'num_variables':num_variables,
                    'diff_shh':diff_shh,
                    'diff_bmp4':diff_bmp4,
                    'prod_scale_shh':prod_scale_shh,
                    'prod_scale_bmp4':prod_scale_bmp4,
                    'bind_shh':bind_shh,
                    'beta_shh':beta_shh,
                    'K_shh_foxf1': K_shh_foxf1,
                    'K_foxf1_bmp4': K_foxf1_bmp4,
                    'alpha_foxf1': alpha_foxf1,
                    'beta_foxf1': beta_foxf1,
                    'alpha_bmp4': alpha_bmp4,
                    'beta_bmp4': beta_bmp4,
                    'bws_ligand': bws_ligand}
    
    f = open('%s/meta_data_' % simulation_path + str(seed) + '.pkl',"wb")
    pickle.dump(ode_params, f)
    f.close()

    # Save the ODE sol
    ode_sol_output = construct_ode_sol(G, t_eval, ode_sol)

    # Add the ODE sol to G
    set_ode_sol_at_time(G, end_time, t_eval, ode_sol)

    state_matrices = construct_ode_sol_matrices_at_time(G, end_time, t_eval, ode_sol)

    # Plot this shit
    # plot_ode_sols(G, state_matrices, 9, 9, 0.0001, 1.0)
    
    np.savetxt(simulation_path + "/SHH_FOXF1_BMP4/simple_shh_foxf1_bmp4_grn_network_ode_sol_" + str(seed) + ".csv", ode_sol_output, delimiter=',', header="t,x,y,Shh,Ptch1,Shh_bound,Foxf1,Bmp4", comments="")

    # Generate interventions by perturbing bind_shh, K_shh_foxf1 and K_foxf1_bmp4
    # bind_shh_pert = bind_shh * np.random.lognormal(mean=0.0, sigma=pert_scale)
    alpha_foxf1_pert = alpha_foxf1 * knockout_scale
    K_shh_foxf1_pert = K_shh_foxf1
    # K_foxf1_bmp4_pert = K_foxf1_bmp4 * np.random.lognormal(mean=0.0, sigma=pert_scale)
    bind_shh_pert = bind_shh * knockout_scale
    # K_shh_foxf1_pert = K_shh_foxf1
    alpha_bmp4_pert = alpha_bmp4 * knockout_scale
    K_foxf1_bmp4_pert = K_foxf1_bmp4

    # Save the model parameters
    ode_params = {'seed':seed,
                    'num_rows':num_rows,
                    'num_cols':num_cols,
                    'num_variables':num_variables,
                    'pert_scale': pert_scale,
                    'diff_shh':diff_shh,
                    'diff_bmp4':diff_bmp4,
                    'prod_scale_shh':prod_scale_shh,
                    'prod_scale_bmp4':prod_scale_bmp4,
                    'bind_shh':bind_shh_pert,
                    'beta_shh':beta_shh,
                    'K_shh_foxf1': K_shh_foxf1_pert,
                    'K_foxf1_bmp4': K_foxf1_bmp4_pert,
                    'alpha_foxf1': alpha_foxf1_pert,
                    'beta_foxf1': beta_foxf1,
                    'alpha_bmp4': alpha_bmp4_pert,
                    'beta_bmp4': beta_bmp4,
                    'bws_ligand': bws_ligand}
    
    f = open('%s/SHH_FOXF1_BMP4/meta_data_' % simulation_path + 'pert_' + str(seed) + '.pkl',"wb")
    pickle.dump(ode_params, f)
    f.close()

    # initial_states = np.zeros(num_variables*num_nodes)
    set_ode_parameters(G, initial_states, domain_height, domain_width, bws_ligand)

    model_params = np.array([diff_shh, diff_bmp4, bind_shh, \
                            bind_shh_pert, K_shh_foxf1_pert, K_foxf1_bmp4_pert,\
                            alpha_foxf1_pert, beta_foxf1, alpha_bmp4_pert])

    args_obs = np.array((D.ctypes.data, D.shape[0], D.shape[1], model_params.ctypes.data, model_params.shape[0]),dtype=args_dtype)

    funcptr = rhs_cfunc.address
    ode_sol, success = lsoda(funcptr, initial_states, t_eval, data=args_obs)

    # Save the ODE sol
    ode_sol_output = construct_ode_sol(G, t_eval, ode_sol)

    # Add the ODE sol to G
    set_ode_sol_at_time(G, end_time, t_eval, ode_sol)

    state_matrices = construct_ode_sol_matrices_at_time(G, end_time, t_eval, ode_sol)

    # Plot this shit
    # plot_ode_sols(G, state_matrices, 9, 9, 0.0001, 1.0)
    
    np.savetxt(simulation_path + "/SHH_FOXF1_BMP4/simple_shh_foxf1_bmp4_grn_network_ode_pert_sol_" + str(seed) + ".csv", ode_sol_output, delimiter=',', header="t,x,y,Shh,Ptch1,Shh_bound,Foxf1,Bmp4", comments="")