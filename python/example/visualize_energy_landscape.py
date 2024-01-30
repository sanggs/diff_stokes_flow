import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
from importlib import import_module
import scipy.optimize
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import json

from py_diff_stokes_flow.common.common import print_info, print_ok, print_error, print_warning, ndarray
from py_diff_stokes_flow.common.grad_check import check_gradients
from py_diff_stokes_flow.common.display import export_gif

# Update this dictionary if you would like to add new demos.
all_demo_names = {
    # ID: (module name, class name).
    'amplifier': ('amplifier_env_2d', 'AmplifierEnv2d'),
    'piecewise_linear_single': ('piecewise_linear_env_2d', 'PiecewiseLinearEnv2d'),
    'piecewise_linear': ('piecewise_linear_env_2d', 'PiecewiseLinearEnv2d'),
    'piecewise_linear_io': ('piecewise_linear_inlet_outlet_env_2d', 'PiecewiseLinearInletOutletEnv2d'),
    'piecewise_linear_sphere': ('piecewise_linear_sphere_env_2d', 'PiecewiseLinearSphereEnv2d'),
    'flow_averager': ('flow_averager_env_3d', 'FlowAveragerEnv3d'),
    'superposition_gate': ('superposition_gate_env_3d', 'SuperpositionGateEnv3d'),
    'funnel': ('funnel_env_3d', 'FunnelEnv3d'),
    'fluidic_twister': ('fluidic_twister_env_3d', 'FluidicTwisterEnv3d'),
    'fluidic_switch': ('fluidic_switch_env_3d', 'FluidicSwitchEnv3d'),
}

# def write_ndarray(arr, fname, sep = '\n'):
#     with open(fname, 'wb') as f:
#         for v in arr:
#             f.write(v)
#             f.write(os.linesep.encode("utf-8"))

def export_invariables(info_dict, folder = 'data/'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fname = os.path.join(folder, 'info.json')
    with open(fname, 'w') as f:
        json.dump(info_dict, f)

def export_data(vfs, den, vns = None, frame = 1, folder = 'data/'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    vfs = np.array(vfs)
    shape = np.shape(vfs)
    shape = shape[::-1]
    np_vfs = np.zeros(shape)
    for i in range(shape[0]):
        np_vfs[i, :, :] = vfs[:, :, i]

    vfsfname = os.path.join(folder, str(frame) + '_vFS.npy')
    dfname = os.path.join(folder, str(frame) + '_rho.npy')
    np.save(vfsfname, np_vfs)
    
    np_den = np.array(den)
    shape = (shape[1]-1, shape[2]-1)
    np_den = np.reshape(np_den, shape)
    np.save(dfname, np_den)

    if (vns is not None):
        vnsfname = os.path.join(folder, str(frame) + '_vNS.npy')
        np.save(vnsfname, np.array(vns))

def prepare_invariant_data(env):
    dim = 2
    domain_size = [int(cn) for cn in env._cell_nums]
    dx = 1.
    nu = env._cell_options['nu']
    E = env._cell_options['E']
    dir_nodes = []
    dir_velocities = []
    num_dir_nodes = int(len(env._node_boundary_info)/dim)
    for n in range(num_dir_nodes):
        vel = []
        node = []
        for i in range(dim):
            dir_info = env._node_boundary_info[2 * n + i]
            nd = list(dir_info[0])[i]
            node.append(nd)
            vd = dir_info[1]
            vel.append(vd)
        dir_nodes.append(node)
        dir_velocities.append(vel)
    info_dict = {}
    info_dict['domain_size'] = domain_size
    info_dict['dx'] = dx
    info_dict['E'] = E
    info_dict['nu'] = nu
    info_dict['dir_nodes'] = dir_nodes
    info_dict['dir_velocities'] = dir_velocities
    print(info_dict)
    return info_dict
    
if __name__ == '__main__':
    # Input check.
    # if len(sys.argv) != 2:
    #     print_error('Usage: python run_demo.py [demo_name]')
    #     sys.exit(0)
    demo_names = ['piecewise_linear_single']
    for demo_name in demo_names:
        assert demo_name in all_demo_names

        # Hyperparameters which are loaded from the config file.
        config_file_name = 'config/{}.json'.format(demo_name)
        config = {}
        with open(config_file_name, 'r') as f:
            config = json.load(f)
        assert(config is not None)

        cell_dim = config['domain']
        E = config['youngs_modulus']
        pr = config['poissons_ratio']
        vol_tol = config['volume_tolerance']
        inlets = config['inlets']
        inlet_velocities = config['inlet_velocities']
        if (demo_name == 'piecewise_linear_io'):
            outlets = config['outlets']
        num_pieces_per_curve = config['num_pieces_per_curve']
        bounds_for_each_piece = config['bounds_for_each_piece']
        seed = int(config['seed'])
        sample_num = int(config['sample_num'])
        num_noise_samples = int(config['num_noise_samples'])
        solver = config['solver']
        spp = int(config['spp'])
        fps = int(config['fps'])

        # Load class.
        module_name, env_name = all_demo_names[demo_name]
        Env = getattr(import_module('py_diff_stokes_flow.env.{}'.format(module_name)), env_name)
        if(demo_name == 'piecewise_linear_io'):
            env = Env(seed, demo_name, cell_dim, E, pr, vol_tol, inlets, outlets, inlet_velocities, num_pieces_per_curve, bounds_for_each_piece)
        else:
            env = Env(seed, demo_name, cell_dim, E, pr, vol_tol, inlets, inlet_velocities, num_pieces_per_curve, bounds_for_each_piece)

        # Global search: randomly sample initial guesses and pick the best.
        samples = []
        losses = []
        best_sample = None
        best_loss = np.inf
        
        fs = []
        ns = []

        evals = []

        x = env.sample()
        
        env._interface_boundary_type = 'free-slip'
        info_fs = env.solve(x, False, False, { 'solver': solver }, add_noise = False, move_cp = False, info_dict = None) 
        u_single = info_fs[0]['velocity_field']
        u_single = np.copy(u_single).ravel()
        energy = info_fs[0]["scene"].ComputeComplianceEnergy(u_single)
        # evals.append([0, energy])
        # fs.append(info_fs[0])

        
        '''
        for i in range(len(num_pieces_per_curve)):
            info_dict = {}
            info_dict["pidx"] = i

            for j in range(num_pieces_per_curve[i]-1):
                info_dict["cidx"] = j + 1
                for k in range(num_noise_samples):
                    info_dict["noise"] = np.random.rand(2) * 2
                    
                    env._interface_boundary_type = 'free-slip'
                    info_fs = env.solve(x, False, False, { 'solver': solver }, add_noise = False, move_cp = True, info_dict = info_dict) 
                    u_single = info_fs[0]['velocity_field']
                    u_single = np.copy(u_single).ravel()
                    en = info_fs[0]["scene"].ComputeComplianceEnergy(u_single)
                    # print(info_dict["pos"][0], info_dict["pos"][1], en)
                    evals.append([info_dict["pos"][0], info_dict["pos"][1], en])
                    # env._interface_boundary_type = 'no-slip'
                    # info_ns = env.solve(x, False, False, { 'solver': solver }, add_noise = True)
                    # export_data(info_fs[0]['velocity_field'], info_fs[0]['density'], None, (num_noise_samples + 1) * i + j, demo_name)
                
                    # fs.append(info_fs[0])
                    # ns.append(info_ns[0])
        '''

        s = np.linspace(-1., 1., num = num_noise_samples)
        for j in range(1):
            evals = []
            info_dict = {}
            info_dict["noise"] = []
            for i in range(len(num_pieces_per_curve)):
                ni = np.random.rand(num_pieces_per_curve[i] + 1)
                info_dict["noise"].append(np.copy(ni))
            
            for k in range(num_noise_samples):
                info_dict["scale"] = s[k]
                env._interface_boundary_type = 'no-slip'
                info_fs = env.solve(x, False, False, { 'solver': solver }, add_noise = False, move_cp = True, info_dict = info_dict) 
                u_single = info_fs[0]['forward_result_single']
                u_single = np.copy(u_single).ravel()
                en = info_fs[0]["scene"].ComputeComplianceEnergy(u_single)
                # print(info_dict["pos"][0], info_dict["pos"][1], en)
                evals.append([s[k], en])
                # env._interface_boundary_type = 'no-slip'
                # info_ns = env.solve(x, False, False, { 'solver': solver }, add_noise = True)
                # export_data(info_fs[0]['velocity_field'], info_fs[0]['density'], None, (num_noise_samples + 1) * i + j, demo_name)
            
                fs.append(info_fs[0])
                # ns.append(info_ns[0])

            with open('ns_energy_vals_{}_{}.txt'.format(j, num_pieces_per_curve[0]), 'w') as f:
                for entry in evals:
                    f.write("{} {}\n".format(entry[0], entry[1]))

        env._render_vel_2d(fs, 'viz_ns_linesearch_sample')
        # env._render_vel_2d(ns, 'viz_ns')
    
    

    
