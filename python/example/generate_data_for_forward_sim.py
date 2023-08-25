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

from py_diff_stokes_flow.common.common import print_info, print_ok, print_error, print_warning, ndarray
from py_diff_stokes_flow.common.grad_check import check_gradients
from py_diff_stokes_flow.common.display import export_gif

# Update this dictionary if you would like to add new demos.
all_demo_names = {
    # ID: (module name, class name).
    'amplifier': ('amplifier_env_2d', 'AmplifierEnv2d'),
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

def export_data(vfs, den, vns, frame = 1, folder = 'data/'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    vfsfname = os.path.join(folder, str(frame) + '_vFS.npy')
    dfname = os.path.join(folder, str(frame) + '_rho.npy')
    vnsfname = os.path.join(folder, str(frame) + '_vNS.npy')
    np.save(vfsfname, np.array(vfs))
    np.save(vnsfname, np.array(vns))
    np.save(dfname, np.array(den))

if __name__ == '__main__':
    # Input check.
    if len(sys.argv) != 2:
        print_error('Usage: python run_demo.py [demo_name]')
        sys.exit(0)
    demo_name = sys.argv[1]
    assert demo_name in all_demo_names

    # Hyperparameters which are loaded from the config file.
    config_file_name = 'config/{}.txt'.format(demo_name)
    config = {}
    with open(config_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, val = line.strip().split(':')
            key = key.strip()
            val = val.strip()
            config[key] = val
    seed = int(config['seed'])
    sample_num = int(config['sample_num'])
    solver = config['solver']
    rel_tol = float(config['rel_tol'])
    max_iter = int(config['max_iter'])
    enable_grad_check = config['enable_grad_check'] == 'True'
    spp = int(config['spp'])
    fps = int(config['fps'])

    # Load class.
    module_name, env_name = all_demo_names[demo_name]
    Env = getattr(import_module('py_diff_stokes_flow.env.{}'.format(module_name)), env_name)
    env = Env(seed, demo_name)

    # Global search: randomly sample initial guesses and pick the best.
    samples = []
    losses = []
    best_sample = None
    best_loss = np.inf
    print_info('Randomly sampling initial guesses...')
    
    fs = []
    ns = []
    for i in tqdm(range(sample_num)):
        x = env.sample()
        env._interface_boundary_type = 'free-slip'
        _, info_fs = env.solve(x, False, { 'solver': solver }) 
        env._interface_boundary_type = 'no-slip'
        _, info_ns = env.solve(x, False, { 'solver': solver })
        export_data(info_fs[0]['velocity_field'], info_fs[0]['density'], info_ns[0]['velocity_field'], i, 'amplifier')
        
        fs.append(info_fs[0])
        ns.append(info_ns[0])
    
    env._render_vel_2d(fs, 'viz_fs')
    env._render_vel_2d(ns, 'viz_ns')

    
