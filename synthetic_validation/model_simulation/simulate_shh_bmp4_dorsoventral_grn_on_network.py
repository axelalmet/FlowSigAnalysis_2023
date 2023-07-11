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
num_variables = 9
pert_scale = 1.0
knockout_scale = 1e-3

# ODE parameters (for the solver evaluation)
end_time = 10.0
num_timepoints = 2
num_realisations = 5

# FGF8 parameters
diff_shh = 200.0 # Diffusion of FGF8
prod_scale_shh = 0.1 #Production multiplier
bind_shh = 2.0  # Binding rate
beta_shh = 0.1 # Degradation rate

# BMP4 parameters
diff_bmp4 = 200.0 # Diffusion of FGF8
prod_scale_bmp4 = 0.1 #Production multiplier
bind_bmp4 = 2.0
beta_bmp4 = 0.1 # Degradation rate

# TF degradation parameters
beta_d = 2.0
beta_i = 2.0
beta_v = 2.0

# Morphogen-TF factors
alpha_d = 2.0
alpha_i = 2.0
alpha_v = 2.0

K_shh_i = 25.0
K_shh_v = 25.0
K_bmp4_d = 10.0
K_bmp4_i = 10.0

# TF-TF factors
K_d_i = 15.0
K_i_d = 10.0
K_d_v = 10.0
K_v_d = 15.0
K_v_i = 15.0
K_i_v = 15.0

# TF-morphogen factors

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
        G.nodes[i, j]['prod_' + lig]  = prods[index] # Auto-production term of FGF8

        index += 1

def set_ode_parameters(G, initial_states, domain_height, domain_width, bws_ligand):

    num_nodes = len(G.nodes())

    # Define FGF8 first
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
        bmp4_rec = np.random.random(1) # Uniformly random receptor expression? (Could pick something else)

        # Set the complex to be zero always
        shh_bound = 0.0
        bmp4_bound = 0.0

        # Set the direct target TFs to be 0
        d = 1.0
        i = 1.0
        v = 1.0

        # We should be able to set the initial state now
        initial_states[num_variables * index] = shh
        initial_states[num_variables * index + 1] = shh_rec
        initial_states[num_variables * index+ 2] = shh_bound
        initial_states[num_variables * index + 3] = bmp4
        initial_states[num_variables * index + 4] = bmp4_rec
        initial_states[num_variables * index + 5] = bmp4_bound
        initial_states[num_variables * index + 6] = d
        initial_states[num_variables * index + 7] = i
        initial_states[num_variables * index + 8] = v

        G.nodes[i, j]['SHH'] = shh
        G.nodes[i, j]['PTCH1_SMO'] = shh_rec
        G.nodes[i, j]['SHH_bound'] = shh_bound
        G.nodes[i, j]['BMP4'] = bmp4
        G.nodes[i, j]['BMPR1A_BMPR2'] = bmp4_rec
        G.nodes[i, j]['BMP4_bound'] = bmp4_bound
        G.nodes[i, j]['D'] = d
        G.nodes[i, j]['I'] = i
        G.nodes[i, j]['V'] = v

def set_ode_sol_at_time(G, time, all_timepoints, sol):

    timeIndex = np.where(all_timepoints == time)[0][0]  
    node_pairs = list(G.nodes())

    for node in node_pairs:

        i, j = node
        index = G.nodes[i, j]['index']

        G.nodes[i, j]['SHH'] = float(sol[timeIndex, num_variables * index])
        G.nodes[i, j]['PTCH1_SMO'] = float(sol[timeIndex, num_variables * index + 1])
        G.nodes[i, j]['SHH_bound'] = float(sol[timeIndex, num_variables * index + 2])
        G.nodes[i, j]['BMP4'] = float(sol[timeIndex, num_variables * index + 3])
        G.nodes[i, j]['BMPR1A_BMPR2'] = float(sol[timeIndex, num_variables * index + 4])
        G.nodes[i, j]['BMP4_bound'] = float(sol[timeIndex, num_variables * index + 5])
        G.nodes[i, j]['D'] = float(sol[timeIndex, num_variables * index + 6])
        G.nodes[i, j]['I'] = float(sol[timeIndex, num_variables * index + 7])
        G.nodes[i, j]['V'] = float(sol[timeIndex, num_variables * index + 8])

def plot_ode_sols(G, ode_sol_matrices, fig_width, fig_height, num_rows, num_cols, v_min, v_max):

    plot_count = 0
    node_coordinates = dict(G.nodes(data='pos')) # Store as node coordinates for later
    state_names = ['SHH', 'PTCH1_SMO', 'SHH_bound',\
                   'BMP4', 'BMPR1A_BMPR2', 'BMP4_bound',\
                   'D', 'I', 'V']

    # Initialise figure
    fig, axs = plt.subplots(num_rows, num_cols)

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
            output_sol[t*num_nodes + index, 8] = ode_sol[t, num_variables * index + 5]
            output_sol[t*num_nodes + index, 9] = ode_sol[t, num_variables * index + 6]
            output_sol[t*num_nodes + index, 10] = ode_sol[t, num_variables * index + 7]
            output_sol[t*num_nodes + index, 11] = ode_sol[t, num_variables * index + 8]

    return output_sol

def construct_ode_sol_matrices_at_time(G, time, all_timepoints, sol):

    timeIndex = np.where(all_timepoints == time)[0][0]  
    node_pairs = list(G.nodes())

    state_matrices = {}
    state_matrices['SHH'] = np.zeros((num_cols, num_rows))
    state_matrices['PTCH1_SMO'] = np.zeros((num_cols, num_rows))
    state_matrices['SHH_bound'] = np.zeros((num_cols, num_rows))
    state_matrices['BMP4'] = np.zeros((num_cols, num_rows))
    state_matrices['BMPR1A_BMPR2'] = np.zeros((num_cols, num_rows))
    state_matrices['BMP4_bound'] = np.zeros((num_cols, num_rows))
    state_matrices['D'] = np.zeros((num_cols, num_rows))
    state_matrices['I'] = np.zeros((num_cols, num_rows))
    state_matrices['V'] = np.zeros((num_cols, num_rows))

    for node in node_pairs:
        
        state_sol = np.zeros((num_cols, num_rows))
        i, j = node
        index = G.nodes[i, j]['index']

        state_matrices['SHH'][i, j] = float(sol[timeIndex, num_variables * index])
        state_matrices['PTCH1_SMO'][i, j] = float(sol[timeIndex, num_variables * index + 1])
        state_matrices['SHH_bound'][i, j] = float(sol[timeIndex, num_variables * index + 2])
        state_matrices['BMP4'][i, j]  = float(sol[timeIndex, num_variables * index + 3])
        state_matrices['BMPR1A_BMPR2'][i, j]  = float(sol[timeIndex, num_variables * index + 4])
        state_matrices['BMP4_bound'][i, j]  = float(sol[timeIndex, num_variables * index + 5])
        state_matrices['D'][i, j] = float(sol[timeIndex, num_variables * index + 6])
        state_matrices['I'][i, j] = float(sol[timeIndex, num_variables * index + 7])
        state_matrices['V'][i, j] = float(sol[timeIndex, num_variables * index + 8])

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

def grn_rhs(t, Y, dY, D, model_params):

    diff_shh, diff_bmp4, bind_shh,\
    bind_bmp4, beta_shh, beta_bmp4,\
    alpha_d, alpha_i, alpha_v,\
    K_shh_i, K_shh_v,\
    K_bmp4_d,K_bmp4_i,\
    K_d_i, K_i_d,\
    K_d_v, K_v_d,\
    K_v_i, K_i_v,\
    beta_d, beta_i, beta_v = model_params


    # Helper function to define reaction rates later
    def hill(x):
        return x / (1 + x)
    
    for index in range(num_nodes):

        prod_rate_shh = prod_rates_shh[index]
        prod_rate_bmp4 = prod_rates_bmp4[index]

        # Get the kinetic parameters for the ligand-receptor binding
        shh = Y[num_variables * index] # SHH
        shh_rec = Y[num_variables * index + 1] # PTCH1_SMO
        shh_bound = Y[num_variables * index + 2] # Bound complex
        bmp4 = Y[num_variables * index + 3] # BMP4 
        bmp4_rec = Y[num_variables * index + 4] # BMPR1A_BMPR2
        bmp4_bound = Y[num_variables * index + 5] # BMPR1A_BMPR2
        d = Y[num_variables * index + 6] # DORSAL PATTERN
        i = Y[num_variables * index + 7] # INTERMEDIATE
        v = Y[num_variables * index + 8] # VENTRAL

        # We want D[index, j] * Shh[j]
        shh_flux = np.sum(np.array([D[index, n]*Y[num_variables*(n)] for n in range(num_nodes)]))

        # We want D[index, j] * Bmp4[j]
        bmp4_flux = np.sum(np.array([D[index, n]*Y[num_variables*(n) + 3] for n in range(num_nodes)]))

        # FGF8 Ligand-receptor binding, plus inhibition due to emx2 (from BMP4)
        dY[num_variables * index] = diff_shh * shh_flux\
                                     - bind_shh * shh * shh_rec\
                                     + prod_rate_shh\
                                     - beta_shh * shh # FGF8
        dY[num_variables * index + 1] = - bind_shh * shh * shh_rec # FGF receptor
        dY[num_variables * index + 2] = bind_shh * shh * shh_rec # Bound complex

        # BMP4 ligand-receptor binding
        dY[num_variables * index + 3] = diff_bmp4 * bmp4_flux\
                                          - bind_bmp4 * bmp4 * bmp4_rec\
                                          + prod_rate_bmp4\
                                          - beta_bmp4 * bmp4 # BMP4
        dY[num_variables * index + 4] = - bind_bmp4 * bmp4 * bmp4_rec # BMP receptor
        dY[num_variables * index + 5] = bind_bmp4 * bmp4 * bmp4_rec # Bound complex

        # Activation of dorsal expression  due to BMP4 and inhibition by intermediate and ventral
        dY[num_variables * index + 6] = alpha_d * hill((1.0 + K_bmp4_d * bmp4_bound) \
                                         / ( ((1.0 + K_i_d * i)**2.0) \
                                        * ( (1.0 + K_v_d * v)**2.0 ) ) )\
                                        - beta_d * d
                                              
        # Activation of intermediate due to SHH and BMP4 and inhibition due to dorsal and ventral genes   
        dY[num_variables * index + 7] = alpha_i * hill( ( ( 1.0 + K_bmp4_i * bmp4_bound) \
                                        * (1.0 + K_shh_i * shh_bound) )\
                                        / ( ((1.0 + K_d_i * d)**2.0 )\
                                        * ( (1.0 + K_v_i * v)**2.0 ) ) )\
                                        - beta_i * i

        # Activation of ventral due to SHH and inhibition due to dorsal and intermediate genes
        dY[num_variables * index + 8] = alpha_v * hill((1.0 + K_shh_v * shh_bound)\
                                        / ( ((1.0 + K_d_v * d)**2.0 )\
                                        * ( (1.0 + K_i_v * i)**2.0 ) ) )\
                                        - beta_v * v

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

    model_params = np.array([diff_shh, diff_bmp4, bind_shh,\
                            bind_bmp4, beta_shh, beta_bmp4,\
                            alpha_d, alpha_i, alpha_v,\
                            K_shh_i, K_shh_v,\
                            K_bmp4_d,K_bmp4_i,\
                            K_d_i, K_i_d,\
                            K_d_v, K_v_d,\
                            K_v_i, K_i_v,\
                            beta_d, beta_i, beta_v])

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
                    'bind_bmp4':bind_bmp4,
                    'beta_shh':beta_shh,
                    'beta_bmp4':beta_bmp4,
                    'alpha_d': alpha_d,
                    'alpha_i': alpha_i,
                    'alpha_v': alpha_v,
                    'K_shh_i': K_shh_i, 
                    'K_shh_v': K_shh_v,
                    'K_bmp4_d': K_bmp4_d,
                    'K_bmp4_i': K_bmp4_i,
                    'K_d_i': K_d_i,
                    'K_i_d': K_i_d,
                    'K_d_v': K_d_v,
                    'K_v_d': K_v_d,
                    'K_v_i': K_v_i,
                    'K_i_v': K_i_v,
                    'beta_d': beta_d,
                    'beta_i': beta_i,
                    'beta_v': beta_v,
                    'bws_ligand': bws_ligand}
    
    f = open('%s/SHH_BMP4_DORSOVENTRAL/meta_data_' % simulation_path + str(seed) + '.pkl',"wb")
    pickle.dump(ode_params, f)
    f.close()

    # Save the ODE sol
    ode_sol_output = construct_ode_sol(G, t_eval, ode_sol)

    # Add the ODE sol to G
    set_ode_sol_at_time(G, end_time, t_eval, ode_sol)

    state_matrices = construct_ode_sol_matrices_at_time(G, end_time, t_eval, ode_sol)
    
    np.savetxt(simulation_path + "/SHH_BMP4_DORSOVENTRAL/shh_bmp4_dorsoventral_grn_network_ode_sol_" + str(seed) + ".csv",
               ode_sol_output,
               delimiter=',',
               header="t,x,y,Shh,Ptch1,Shh_bound,Bmp4,Bmpr1A_Bmpr2,Bmp4_bound,D,I,V", comments="")

    # Now perturb the regulatory parameters
    bind_shh_pert = bind_shh * knockout_scale
    bind_bmp4_pert = bind_bmp4 * knockout_scale
    K_shh_i_pert = K_shh_i * knockout_scale
    K_shh_v_pert = K_shh_v * knockout_scale
    K_bmp4_d_pert = K_bmp4_d * knockout_scale
    K_bmp4_i_pert = K_bmp4_i * knockout_scale
    K_d_i_pert = K_d_i * knockout_scale
    K_i_d_pert = K_i_d * knockout_scale
    K_d_v_pert = K_d_v * knockout_scale
    K_v_d_pert = K_v_d * knockout_scale
    K_v_i_pert = K_v_i * knockout_scale
    K_i_v_pert = K_i_v * knockout_scale

    initial_states = np.zeros(num_variables*num_nodes)

    model_params = np.array([diff_shh, diff_bmp4, bind_shh_pert,\
                            bind_bmp4_pert, beta_shh, beta_bmp4,\
                            alpha_d, alpha_i, alpha_v,\
                            K_shh_i_pert, K_shh_v_pert,\
                            K_bmp4_d_pert, K_bmp4_i_pert,\
                            K_d_i_pert, K_i_d_pert,\
                            K_d_v_pert, K_v_d_pert,\
                            K_v_i_pert, K_i_v_pert,\
                            beta_d, beta_i, beta_v])
    
    set_ode_parameters(G, initial_states, domain_height, domain_width, bws_ligand)

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
                    'bind_shh':bind_shh_pert,
                    'bind_bmp4':bind_bmp4_pert,
                    'beta_shh':beta_shh,
                    'beta_bmp4':beta_bmp4,
                    'alpha_d': alpha_d,
                    'alpha_i': alpha_i,
                    'alpha_v': alpha_v,
                    'K_shh_i': K_shh_i_pert, 
                    'K_shh_v': K_shh_v_pert,
                    'K_bmp4_d': K_bmp4_d_pert,
                    'K_bmp4_i': K_bmp4_i_pert,
                    'K_d_i': K_d_i_pert,
                    'K_i_d': K_i_d_pert,
                    'K_d_v': K_d_v_pert,
                    'K_v_d': K_v_d_pert,
                    'K_v_i': K_v_i_pert,
                    'K_i_v': K_i_v_pert,
                    'beta_d': beta_d,
                    'beta_i': beta_i,
                    'beta_v': beta_v,
                    'bws_ligand': bws_ligand}
    
    f = open('%s/SHH_BMP4_DORSOVENTRAL/meta_data_' % simulation_path + "pert_" + str(seed) + '.pkl',"wb")
    pickle.dump(ode_params, f)
    f.close()

    # Save the ODE sol
    ode_sol_output = construct_ode_sol(G, t_eval, ode_sol)

    # Add the ODE sol to G
    set_ode_sol_at_time(G, end_time, t_eval, ode_sol)

    state_matrices = construct_ode_sol_matrices_at_time(G, end_time, t_eval, ode_sol)
    
    np.savetxt(simulation_path + "/SHH_BMP4_DORSOVENTRAL/shh_bmp4_dorsoventral_grn_network_ode_sol_pert_" + str(seed) + ".csv",
               ode_sol_output,
               delimiter=',',
               header="t,x,y,Shh,Ptch1,Shh_bound,Bmp4,Bmpr1A_Bmpr2,Bmp4_bound,D,I,V", comments="")
