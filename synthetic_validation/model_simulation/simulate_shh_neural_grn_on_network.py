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
num_variables = 7
pert_scale = 1.0
knockout_scale = 0.001

# ODE parameters (for the solver evaluation)
end_time = 1.0
num_timepoints = 2
num_realisations = 5

# FGF8 parameters
diff_shh = 100.0 # Diffusion of FGF8
prod_scale_shh = 0.1 #Production multiplier
bind_shh = 2.0  # Binding rate
beta_shh = 0.1 # Degradation rate

# TF degradation parameters
beta_nkx22 = 2.0
beta_olig2 = 2.0
beta_pax6 = 2.0
beta_irx3 = 2.0

# Morphogen-TF factors
K_shh_nkx22 = 375.0
K_shh_olig2 = 20.0

# TF-TF factors
alpha_nkx22 = 2.0
alpha_olig2 = 2.0
alpha_pax6 = 2.0
alpha_irx3 = 2.0

K_nkx22_olig2 = 60.0
K_nkx22_pax6 = 25.0
K_nkx22_irx3 = 75.0

K_olig2_nkx22= 27.0
K_olig2_pax6 = 2.0
K_olig2_irx3 = 15.0

K_pax6_nkx22 = 5.0

K_irx3_nkx22 = 76.0
K_irx3_olig2 = 60.0

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

    for node in G.nodes():

        i, j = node

        # Get the cell type and position
        position = G.nodes[i, j]['pos']
        index = G.nodes[i, j]['index']

        # Set the ligand, receptor, and complex concentration for this node
        shh = np.exp( -(position[0] - shh_x)**2.0 / (2.0 * shh_r)**2.0)

        # Shh is produced at the left
        if position[0] < domain_width / 6.0:
            shh = 1.0

        shh_rec = np.random.random(1) # Uniformly random receptor expression? (Could pick something else)

        # Set the complex to be zero always
        shh_bound = 0.0

        # Just set the TFs to be 0
        nkx22 = 0.0
        olig2 = 0.0
        pax6 = 1.0
        irx3 = 1.0

        # We should be able to set the initial state now
        initial_states[num_variables * index] = shh
        initial_states[num_variables * index + 1] = shh_rec
        initial_states[num_variables * index+ 2] = shh_bound
        initial_states[num_variables * index + 3] = nkx22
        initial_states[num_variables * index + 4] = olig2
        initial_states[num_variables * index + 5] = pax6
        initial_states[num_variables * index + 6] = irx3

        G.nodes[i, j]['SHH'] = shh
        G.nodes[i, j]['PTCH1_SMO'] = shh_rec
        G.nodes[i, j]['SHH_bound'] = shh_bound
        G.nodes[i, j]['NKX2.2'] = nkx22
        G.nodes[i, j]['OLIG2'] = olig2
        G.nodes[i, j]['PAX6'] = pax6
        G.nodes[i, j]['IRX3'] = irx3

def set_ode_sol_at_time(G, time, all_timepoints, sol):

    timeIndex = np.where(all_timepoints == time)[0][0]  
    node_pairs = list(G.nodes())

    for node in node_pairs:

        i, j = node
        index = G.nodes[i, j]['index']

        G.nodes[i, j]['SHH'] = float(sol[timeIndex, num_variables * index])
        G.nodes[i, j]['PTCH1_SMO'] = float(sol[timeIndex, num_variables * index + 1])
        G.nodes[i, j]['SHH_bound'] = float(sol[timeIndex, num_variables * index + 2])
        G.nodes[i, j]['NKX2.2'] = float(sol[timeIndex, num_variables * index + 3])
        G.nodes[i, j]['OLIG2'] = float(sol[timeIndex, num_variables * index + 4])
        G.nodes[i, j]['PAX6'] = float(sol[timeIndex, num_variables * index + 5])
        G.nodes[i, j]['IRX3'] = float(sol[timeIndex, num_variables * index + 6])

def plot_ode_sols(G, ode_sol_matrices, fig_width, fig_height, num_rows, num_cols, v_min, v_max):

    plot_count = 0
    node_coordinates = dict(G.nodes(data='pos')) # Store as node coordinates for later
    state_names = ['SHH', 'PTCH1_SMO', 'SHH_bound',\
                   'NKX2.2', 'OLIG2', 'PAX6', 'IRX3']

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

    return output_sol

def construct_ode_sol_matrices_at_time(G, time, all_timepoints, sol):

    timeIndex = np.where(all_timepoints == time)[0][0]  
    node_pairs = list(G.nodes())

    state_matrices = {}
    state_matrices['SHH'] = np.zeros((num_cols, num_rows))
    state_matrices['PTCH1_SMO'] = np.zeros((num_cols, num_rows))
    state_matrices['SHH_bound'] = np.zeros((num_cols, num_rows))
    state_matrices['NKX2.2'] = np.zeros((num_cols, num_rows))
    state_matrices['OLIG2'] = np.zeros((num_cols, num_rows))
    state_matrices['PAX6'] = np.zeros((num_cols, num_rows))
    state_matrices['IRX3'] = np.zeros((num_cols, num_rows))

    for node in node_pairs:
        
        state_sol = np.zeros((num_cols, num_rows))
        i, j = node
        index = G.nodes[i, j]['index']

        state_matrices['SHH'][i, j] = float(sol[timeIndex, num_variables * index])
        state_matrices['PTCH1_SMO'][i, j] = float(sol[timeIndex, num_variables * index + 1])
        state_matrices['SHH_bound'][i, j] = float(sol[timeIndex, num_variables * index + 2])
        state_matrices['NKX2.2'][i, j]  = float(sol[timeIndex, num_variables * index + 3])
        state_matrices['OLIG2'][i, j]  = float(sol[timeIndex, num_variables * index + 4])
        state_matrices['PAX6'][i, j]  = float(sol[timeIndex, num_variables * index + 5])
        state_matrices['IRX3'][i, j] = float(sol[timeIndex, num_variables * index + 6])

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

# Define the node pairs
node_pairs = list(G.nodes())

for node in node_pairs:
    i, j = node
    index = G.nodes[i, j]['index']
    pos = G.nodes[i, j]['pos']

    prod_rates_shh[index] = prod_scale_shh

    # Only have a few cells 
    if pos[0] > domain_width / 18.0:
        prod_rates_shh[index] = 0.0
    
set_graph_attributes(G, prod_rates_shh, lig='SHH') # To set teh production rates


# Generate the initial state

def grn_rhs(t, Y, dY, D, model_params):

    diff_shh, bind_shh, beta_shh,\
    K_shh_nkx22, K_shh_olig2,\
    alpha_olig2, alpha_olig2,\
    alpha_pax6, alpha_irx3,\
    K_nkx22_olig2, K_nkx22_pax6, K_nkx22_irx3,\
    K_olig2_nkx22, K_olig2_pax6, K_olig2_irx3,\
    K_pax6_nkx22, K_irx3_nkx22, K_irx3_olig2,\
    beta_nkx22, beta_olig2, beta_pax6, beta_irx3 = model_params

    # Helper function to define reaction rates later
    def hill(x):
        return x / (1 + x)
    
    for index in range(num_nodes):

        prod_rate_shh = prod_rates_shh[index]

        # Get the kinetic parameters for the ligand-receptor binding
        shh = Y[num_variables * index] # SHH
        shh_rec = Y[num_variables * index + 1] # PTCH1_SMO
        shh_bound = Y[num_variables * index + 2] # Bound SHH complex
        nkx22 = Y[num_variables * index + 3] # NKX2.2 
        olig2 = Y[num_variables * index + 4] # OLIG2
        pax6 = Y[num_variables * index + 5] # PAX6
        irx3 = Y[num_variables * index + 6] # IRX3

        # We want D[index, j] * Shh[j]
        shh_flux = np.sum(np.array([D[index, n]*Y[num_variables*(n)] for n in range(num_nodes)]))

        # SHH Ligand-receptor binding
        dY[num_variables * index] = diff_shh * shh_flux\
                                     - bind_shh * shh * shh_rec\
                                     + prod_rate_shh\
                                     - beta_shh * shh # FGF8
        dY[num_variables * index + 1] = - bind_shh * shh * shh_rec # FGF receptor
        dY[num_variables * index + 2] = bind_shh * shh * shh_rec # Bound complex

        # Nkx2.2 regulation due to Olig2, Pax6, and Irx3
        dY[num_variables * index + 3] = alpha_nkx22 * hill( (1.0 + K_shh_nkx22 * shh_bound) \
                                        / ( ( (1.0 + K_olig2_nkx22 * olig2)**2.0 )\
                                        * ( (1.0 + K_pax6_nkx22 * pax6)**2.0 )\
                                        * ( (1.0 + K_irx3_nkx22 * olig2)**2.0 ) ) )\
                                        - beta_nkx22 * nkx22
        
        # Olig2 regulation due to Nkx2.2 and Irx3
        dY[num_variables * index + 4] = alpha_olig2 * hill( (1.0 + K_shh_olig2 * shh_bound) \
                                        / ( ( (1.0 + K_nkx22_olig2 * nkx22)**2.0 ) \
                                        + ( (1.0 + K_irx3_olig2 * irx3)**2.0 ) ) )\
                                        - beta_olig2 * olig2

        # Pax6 regulation due to Nkx2.2 and Olig2
        dY[num_variables * index + 5] = alpha_pax6 * hill(1.0 / ( ( (1.0 + K_nkx22_pax6 * nkx22)**2.0 ) * ( (1.0 + K_olig2_pax6 * olig2)**2.0) ) )\
                                        - beta_pax6 * pax6

        # Irx3 regulation due to Nkx2.2 and Olig2
        dY[num_variables * index + 6] = alpha_irx3 * hill(1.0 / ( ( (1.0 + K_nkx22_irx3 * nkx22)**2.0 ) * ( (1.0 + K_olig2_irx3 * olig2)**2.0 ) ) ) \
                                        - beta_irx3 * irx3

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

    model_params = np.array([diff_shh, bind_shh, beta_shh,\
                            K_shh_nkx22, K_shh_olig2,\
                            alpha_olig2, alpha_olig2,\
                            alpha_pax6, alpha_irx3,\
                            K_nkx22_olig2, K_nkx22_pax6, K_nkx22_irx3,\
                            K_olig2_nkx22, K_olig2_pax6, K_olig2_irx3,\
                            K_pax6_nkx22, K_irx3_nkx22, K_irx3_olig2,\
                            beta_nkx22, beta_olig2, beta_pax6, beta_irx3])

    args_obs = np.array((D.ctypes.data, D.shape[0], D.shape[1], model_params.ctypes.data, model_params.shape[0]),dtype=args_dtype)

    funcptr = rhs_cfunc.address
    ode_sol, success = lsoda(funcptr, initial_states, t_eval, data=args_obs)

    # Save the model parameters
    ode_params = {'seed':seed,
                    'num_rows':num_rows,
                    'num_cols':num_cols,
                    'num_variables':num_variables,
                    'diff_shh':diff_shh,
                    'prod_scale_shh':prod_scale_shh,
                    'bind_shh':bind_shh,
                    'beta_shh':beta_shh,
                    'K_shh_nkx22': K_shh_nkx22,
                    'K_shh_olig2': K_shh_olig2,
                    'alpha_nkx22': alpha_olig2,
                    'alpha_olig2': alpha_olig2,
                    'alpha_pax6': alpha_pax6, 
                    'alpha_irx3': alpha_irx3,
                    'K_nkx22_olig2': K_nkx22_olig2,
                    'K_nkx22_pax6': K_nkx22_pax6,
                    'K_nkx22_irx3': K_nkx22_irx3,
                    'K_olig2_nkx22': K_olig2_nkx22, 
                    'K_olig2_pax6': K_olig2_pax6,
                    'K_olig2_irx3': K_olig2_irx3,
                    'K_pax6_nkx22': K_pax6_nkx22,
                    'K_irx3_nkx22': K_irx3_nkx22,
                    'K_irx3_olig2': K_irx3_olig2,
                    'beta_nkx22': beta_nkx22,
                    'beta_olig2': beta_olig2,
                    'beta_pax6': beta_pax6,
                    'beta_irx3': beta_irx3,
                    'bws_ligand': bws_ligand}
    
    f = open('%s/SHH_NEURAL_TUBE/meta_data_' % simulation_path + str(seed) + '.pkl',"wb")
    pickle.dump(ode_params, f)
    f.close()

    # Save the ODE sol
    ode_sol_output = construct_ode_sol(G, t_eval, ode_sol)

    # Add the ODE sol to G
    set_ode_sol_at_time(G, end_time, t_eval, ode_sol)

    state_matrices = construct_ode_sol_matrices_at_time(G, end_time, t_eval, ode_sol)

    # Plot this shit
    
    np.savetxt(simulation_path + "/SHH_NEURAL_TUBE/shh_neural_tube_grn_network_ode_sol_" + str(seed) + ".csv",
                ode_sol_output,
                delimiter=',',
                header="t,x,y,Shh,Ptch1,Shh_bound,Nkx2-2,Olig2,Pax6,Irx3",
                comments="")
    
    # Perturb the "causal" parameters (direct causal)
    bind_shh_pert = bind_shh * knockout_scale
    K_shh_nkx22_pert = K_shh_nkx22 * knockout_scale
    K_shh_olig2_pert = K_shh_olig2 * knockout_scale

    K_nkx22_olig2_pert = K_nkx22_olig2 * knockout_scale
    K_nkx22_pax6_pert = K_nkx22_pax6 * knockout_scale
    K_nkx22_irx3_pert = K_nkx22_irx3 * knockout_scale
    K_olig2_nkx22_pert = K_olig2_nkx22 * knockout_scale
    K_olig2_pax6_pert = K_olig2_pax6 * knockout_scale
    K_olig2_irx3_pert = K_olig2_irx3 * knockout_scale
    K_pax6_nkx22_pert = K_pax6_nkx22 * knockout_scale
    K_irx3_nkx22_pert = K_irx3_nkx22 * knockout_scale
    K_irx3_olig2_pert = K_irx3_olig2 * knockout_scale

    initial_states = np.zeros(num_variables*num_nodes)

    set_ode_parameters(G, initial_states, domain_height, domain_width, bws_ligand)


    model_params = np.array([diff_shh, bind_shh_pert, beta_shh,\
                            K_shh_nkx22_pert, K_shh_olig2_pert,\
                            alpha_olig2, alpha_olig2,\
                            alpha_pax6, alpha_irx3,\
                            K_nkx22_olig2_pert, K_nkx22_pax6_pert, K_nkx22_irx3_pert,\
                            K_olig2_nkx22_pert, K_olig2_pax6_pert, K_olig2_irx3_pert,\
                            K_pax6_nkx22_pert, K_irx3_nkx22_pert, K_irx3_olig2_pert,\
                            beta_nkx22, beta_olig2, beta_pax6, beta_irx3])

    args_obs = np.array((D.ctypes.data, D.shape[0], D.shape[1], model_params.ctypes.data, model_params.shape[0]),dtype=args_dtype)

    funcptr = rhs_cfunc.address
    ode_sol, success = lsoda(funcptr, initial_states, t_eval, data=args_obs)

    # Save the model parameters
    ode_params = {'seed':seed,
                    'num_rows':num_rows,
                    'num_cols':num_cols,
                    'num_variables':num_variables,
                    'diff_shh':diff_shh,
                    'prod_scale_shh':prod_scale_shh,
                    'bind_shh':bind_shh_pert,
                    'beta_shh':beta_shh,
                    'K_shh_nkx22': K_shh_nkx22_pert,
                    'K_shh_olig2': K_shh_olig2_pert,
                    'alpha_nkx22': alpha_olig2,
                    'alpha_olig2': alpha_olig2,
                    'alpha_pax6': alpha_pax6, 
                    'alpha_irx3': alpha_irx3,
                    'K_nkx22_olig2': K_nkx22_olig2_pert,
                    'K_nkx22_pax6': K_nkx22_pax6_pert,
                    'K_nkx22_irx3': K_nkx22_irx3_pert,
                    'K_olig2_nkx22': K_olig2_nkx22_pert, 
                    'K_olig2_pax6': K_olig2_pax6_pert,
                    'K_olig2_irx3': K_olig2_irx3_pert,
                    'K_pax6_nkx22': K_pax6_nkx22_pert,
                    'K_irx3_nkx22': K_irx3_nkx22_pert,
                    'K_irx3_olig2': K_irx3_olig2_pert,
                    'beta_nkx22': beta_nkx22,
                    'beta_olig2': beta_olig2,
                    'beta_pax6': beta_pax6,
                    'beta_irx3': beta_irx3,
                    'bws_ligand': bws_ligand}
    
    f = open('%s/SHH_NEURAL_TUBE/meta_data_' % simulation_path + "pert_" + str(seed) + '.pkl',"wb")
    pickle.dump(ode_params, f)
    f.close()

    # Save the ODE sol
    ode_sol_output = construct_ode_sol(G, t_eval, ode_sol)

    # Add the ODE sol to G
    set_ode_sol_at_time(G, end_time, t_eval, ode_sol)

    state_matrices = construct_ode_sol_matrices_at_time(G, end_time, t_eval, ode_sol)

    # Plot this shit
    # plot_ode_sols(G, state_matrices, 9, 9, 3, 3, 0.0001, 1.0)
    
    np.savetxt(simulation_path + "/SHH_NEURAL_TUBE/shh_neural_tube_grn_network_ode_sol_pert_" + str(seed) + ".csv",
               ode_sol_output,
                delimiter=',',
                header="t,x,y,Shh,Ptch1,Shh_bound,Nkx2-2,Olig2,Pax6,Irx3",
                comments="")