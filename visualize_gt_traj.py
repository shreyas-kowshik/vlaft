'''
Load a trajectory from files and visualize it by stepping in the environment
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if not os.path.exists("/scratch/skowshik"):
    os.makedirs("/scratch/skowshik", exist_ok=True)

# LIBERO ENV SETUP #
sys.path.append('/home/skowshik/vla/codebase/envs/LIBERO')
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}
####################

import torch  # only for loading the init_state .pt
import numpy as np
from scipy.spatial.transform import Rotation as R
import h5py
from PIL import Image

import types
m = types.ModuleType("robosuite.macros_private")
m.FILE_LOGGING_LEVEL = None                # turn off file logging
# If your robosuite expects a path var too, set both:
m.FILE_LOGGING_PATH = os.path.expanduser("/scratch/skowshik/robosuite/robosuite.log")
os.makedirs(os.path.dirname(m.FILE_LOGGING_PATH), exist_ok=True)
sys.modules["robosuite.macros_private"] = m
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["PYOPENGL_PLATFORM"] = "egl" # EGL to run on GPU
os.environ['MUJOCO_GL'] = 'egl' # EGL to run on GPU
##### Imports Done #####

def quaternion_to_euler(q):
    rot = R.from_quat(q)
    return rot.as_euler('xyz', degrees=False)

def save_rgbs_to_gif(rgbs, path):
    # rgbs = [Image.fromarray(rgb) for rgb in rgbs]
    rgbs[0].save(path, save_all=True, append_images=rgbs[1:], duration=100, loop=0)

def state_from_obs(obs):
    state_pos = obs["robot0_eef_pos"]
    state_ori = quaternion_to_euler(obs["robot0_eef_quat"])
    st = np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']]).astype(np.float32)  # (8,)
    return st

def setup_env(task_id, task_name):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(task_id)
    assert task_name == task.name, "Task name is not correct"
    libero_path = "/home/skowshik/vla/codebase/envs/LIBERO"
    task_bddl_file = os.path.join(f"{libero_path}/libero/libero/bddl_files",
                                    task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 224,
        "camera_widths": 224,
        "render_gpu_device_id": 0,
    }
    env = OffScreenRenderEnv(**env_args)
    env.task_id = task_id
    env.task_name = task_name
    env.task_suite_name = "libero_10"
    env.reset()
    env.seed(0)

    return env

def visualize_gt_traj(demo_data, env, gif_path):
    # Visualize the trajectory
    actions = demo_data['actions']
    rgbs = []
    for (t, action) in enumerate(actions):
        # Get current image for rendering #
        img_agentview_rgb = Image.fromarray(demo_data["obs"]["agentview_rgb"][t])
        rgbs.append(img_agentview_rgb)

        # Set sim state
        cur_sim_state = demo_data['states'][()][t]
        env.set_state(cur_sim_state)

        # CAN COMPARE MODEL ACTION WITH GT ACTION HERE #
        ################################################

        env.sim.forward()

    # Save the images #
    save_rgbs_to_gif(rgbs, gif_path)

def visualize_gt_traj_with_actions(demo_data, env, gif_path):
    # Visualize the trajectory
    actions = demo_data['actions']
    rgbs = []

    # Set initial state #
    cur_sim_state = demo_data['states'][()][0]
    env.set_state(cur_sim_state)
    env.sim.forward()

    for (t, action) in enumerate(actions):
        # Take gt action #
        obs, reward, done, info = env.step(actions[()][t])

        # Get current image for rendering #
        img_agentview_rgb = Image.fromarray(obs["agentview_image"])
        rgbs.append(img_agentview_rgb)

    # Save the images #
    save_rgbs_to_gif(rgbs, gif_path)

# Load a gt trajectory
BASE_PATH = "/data/user_data/skowshik/datasets/libero_pro/libero_10"
TASK_NAME = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
TASK_ID = 9
DEMO_ID = 0
traj_file = h5py.File(os.path.join(BASE_PATH, TASK_NAME + "_demo.hdf5"), "r")
demo_data = traj_file['data']['demo_{}'.format(DEMO_ID)]
env = setup_env(TASK_ID, TASK_NAME)

# visualize_gt_traj(demo_data, env, "gt_set_state_sim_rgb.gif")
visualize_gt_traj_with_actions(demo_data, env, "gt_step_sim_gt_actions_rgb.gif")
# Close the environment #
env.close()