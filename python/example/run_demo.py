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

from py_diff_stokes_flow.common.common import ndarray, create_folder, print_warning
from py_diff_stokes_flow.common.common import print_info, print_ok, print_error, print_warning, ndarray
from py_diff_stokes_flow.common.grad_check import check_gradients
from py_diff_stokes_flow.common.display import export_gif

# Update this dictionary if you would like to add new demos.
all_demo_names = {
    # ID: (module name, class name).
    'amplifier': ('amplifier_env_2d', 'AmplifierEnv2d'),
    'lattice_amplifier': ('lattice_amplifier_env_2d', 'LatticeAmplifierEnv2d'),
    'flow_averager': ('flow_averager_env_3d', 'FlowAveragerEnv3d'),
    'superposition_gate': ('superposition_gate_env_3d', 'SuperpositionGateEnv3d'),
    'funnel': ('funnel_env_3d', 'FunnelEnv3d'),
    'fluidic_twister': ('fluidic_twister_env_3d', 'FluidicTwisterEnv3d'),
    'fluidic_switch': ('fluidic_switch_env_3d', 'FluidicSwitchEnv3d'),
}

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

    (cx, cy) = env._cell_nums
    (lx, ly) = env._lattice_cell_nums
    unique_id = '_{}_{}_{}_{}'.format(cx, cy, lx, ly)
    demo_name = demo_name + unique_id

    if demo_name is not None:
        folder = Path(demo_name)
        create_folder(demo_name, exist_ok=True)

    # Global search: randomly sample initial guesses and pick the best.
    samples = []
    losses = []
    best_sample = None
    best_loss = np.inf
    print_info('Randomly sampling initial guesses...')
    for _ in tqdm(range(sample_num)):
        x = env.sample()
        params_and_J = env._variables_to_shape_params(x)
        (param, _) = params_and_J
        env._embed_control_points_in_lattice(param)
        # params_and_J = env._lattice_to_shape_params(env._lattice_nodes)
        loss, _ = env.solve(params_and_J, False, { 'solver': solver }) 
        losses.append(loss)
        samples.append(ndarray(x).copy())
        if loss < best_loss:
            best_loss = loss
            best_sample = np.copy(x)
    unit_loss = np.mean(losses)
    pickle.dump((losses, samples, unit_loss, best_sample), open('{}/sample.data'.format(demo_name), 'wb'))
    # Load from file.
    losses, _, unit_loss, best_sample = pickle.load(open('{}/sample.data'.format(demo_name), 'rb'))
    print_info('Randomly sampled {:d} initial guesses.'.format(sample_num))
    print_info('Loss (min, max, mean): ({:4f}, {:4f}, {:4f}).'.format(
        np.min(losses), np.max(losses), np.mean(losses)
    ))
    print_info('Normalized loss (min, max, mean): ({:4f}, {:4f}, {:4f}).'.format(
        np.min(losses) / unit_loss, np.max(losses) / unit_loss, 1
    ))

    # Local optimization: run L-BFGS from best_sample.
    x_init = np.copy(best_sample)
    param_and_J_init = env._variables_to_shape_params(x)
    env._embed_control_points_in_lattice(param_and_J_init[0])
    lattice_init = np.copy(env._lattice_nodes)
    lattice_bound = np.ones_like(lattice_init) * 1.0
    lattice_bound = np.reshape(lattice_bound, (env._lattice_cell_nums[0]+1, env._lattice_cell_nums[1]+1, 2))
    # for i in range(1):
    #     lattice_bound[0, :, i] = 0
    #     lattice_bound[-1, :, i] = 0
    #     lattice_bound[:, 0, i] = 0
    #     lattice_bound[:, -1, i] = 0
    lattice_bound = np.reshape(lattice_bound, (-1, 1))
    
    bounds = scipy.optimize.Bounds(lattice_init - lattice_bound, lattice_init + lattice_bound)
    def loss_and_grad(x):
        t_begin = time.time()
        params_and_J = env._lattice_to_shape_params(x)
        loss, grad, _ = env.solve(params_and_J, True, { 'solver': solver })
        # Normalize loss and grad.
        loss /= unit_loss
        grad /= unit_loss
        t_end = time.time()
        print('loss: {:3.6e}, |grad|: {:3.6e}, time: {:3.6f}s'.format(loss, np.linalg.norm(grad), t_end - t_begin))
        return loss, grad

    if enable_grad_check:
        print_info('Checking gradients...')
        # Sanity check gradients.
        success = check_gradients(loss_and_grad, x_init)
        if success:
            print_ok('Gradient check succeeded.')
        else:
            print_error('Gradient check failed.')
            sys.exit(0)

    # File index + 1 = len(opt_history).
    loss, grad = loss_and_grad(lattice_init)
    opt_history = [((lattice_init.copy(), env._embedding_weights().copy()), loss, grad.copy())]
    pickle.dump(opt_history, open('{}/{:04d}.data'.format(demo_name, 0), 'wb'))
    def callback(x):
        loss, grad = loss_and_grad(x)
        ew = env._embedding_weights()
        global opt_history
        cnt = len(opt_history)
        print_info('Summary of iteration {:4d}'.format(cnt))
        opt_history.append(((x.copy(), np.copy(ew)), loss, grad.copy()))
        print_info('loss: {:3.6e}, |grad|: {:3.6e}, |x|: {:3.6e}, |p|: {:3.6e}'.format(
            loss, np.linalg.norm(grad), np.linalg.norm(x), np.linalg.norm(x)))
        # Save data to the folder.
        pickle.dump(opt_history, open('{}/{:04d}.data'.format(demo_name, cnt), 'wb'))

    for iter in range(max_iter):
        results = scipy.optimize.minimize(loss_and_grad, lattice_init.copy(), method='L-BFGS-B', jac=True, bounds=bounds,
            callback=callback, options={ 'ftol': rel_tol, 'maxiter': 10})
        if not results.success:
            print_warning('Local optimization fails to reach the optimal condition and will return the last solution.')
        print_info('Data saved to {}/{:04d}.data.'.format(demo_name, len(opt_history) - 1))
        params_and_J = env._lattice_to_shape_params(opt_history[-1][0][0])
        env._embed_control_points_in_lattice(params_and_J[0])

    # Load results from demo_name.
    cnt = 0
    while True:
        data_file_name = '{}/{:04d}.data'.format(demo_name, cnt)
        if not os.path.exists(data_file_name):
            cnt -= 1
            break
        cnt += 1
    data_file_name = '{}/{:04d}.data'.format(demo_name, cnt)
    print_info('Loading data from {}.'.format(data_file_name))
    opt_history = pickle.load(open(data_file_name, 'rb'))

    # Plot the optimization progress.
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)

    fig = plt.figure(figsize=(18, 12))
    ax_loss = fig.add_subplot(121)
    ax_grad = fig.add_subplot(122)

    ax_loss.set_position((0.12, 0.2, 0.33, 0.6))
    iterations = np.arange(len(opt_history))
    ax_loss.plot(iterations, [l for _, l, _ in opt_history], color='tab:red')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_yscale('log')
    ax_loss.grid(True, which='both')

    ax_grad.set_position((0.55, 0.2, 0.33, 0.6))
    ax_grad.plot(iterations, [np.linalg.norm(g) + np.finfo(np.float).eps for _, _, g in opt_history],
        color='tab:green')
    ax_grad.set_xlabel('Iteration')
    ax_grad.set_ylabel('|Gradient|')
    ax_grad.set_yscale('log')
    ax_grad.grid(True, which='both')

    plt.show()
    fig.savefig('{}_progress.pdf'.format(demo_name))

    # Render the results.
    print_info('Rendering optimization history in {}/'.format(demo_name))
    # 000k.png renders opt_history[k], which is also the last element in 000k.data.
    cnt = len(opt_history)
    for k in range(cnt):
        xk, _, _ = opt_history[k]
        
        x = env._lattice_and_weights_to_shape_params(xk[0], xk[1])
        env.render([x, xk[0]], '{:04d}.png'.format(k), { 'solver': solver, 'spp': spp })
        print_info('{}/mode_[0-9]*/{:04d}.png is ready.'.format(demo_name, k))
    # env.render(opt_history[-1][0], '{:04d}.png'.format((cnt - 1) * fps), { 'solver': solver, 'spp': spp })
    # print_info('{}/mode_[0-9]*/{:04d}.png is ready.'.format(demo_name, (cnt - 1) * fps))

    # Get mode number.
    mode_num = 0
    while True:
        mode_folder = Path(demo_name) / 'mode_{:04d}'.format(mode_num)
        if not mode_folder.exists():
            break

        export_gif(mode_folder, '{}_{:04d}.gif'.format(demo_name, mode_num), fps=fps)
        print_info('Video {}_{:04d}.gif is ready.'.format(demo_name, mode_num))

        mode_num += 1