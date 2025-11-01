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

from models.bc_simple import generate_attention_mask

# Other imports #
from scipy.spatial.transform import Rotation as R
import tqdm
import numpy as np
from collections import deque
import jax
import jax.numpy as jnp
from PIL import Image
import clip
import torch

# Utility functions #
def quaternion_to_euler(q):
    rot = R.from_quat(q)
    return rot.as_euler('xyz', degrees=False)

# Model wrapper #
class JAXModelWrapper:
    def __init__(self, model_dict, libero_cfg={}):
        self.model_dict = model_dict
        self.model_def = model_dict["model_def"]
        self.history_len = libero_cfg.get("history_len", 10)
        self.action_pred_steps = libero_cfg.get("action_pred_steps", 3)  # CRITICAL: Required for attention mask
        self.libero_cfg = libero_cfg

        self.gripper_width = libero_cfg.get("gripper_width", False)
        # Initialize gripper state (will be updated during execution)
        self.gripper_state = np.array([-1.0], dtype=np.float32)  # Start with closed gripper
        # queues
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.cnt = 0

        # jitted inference
        @jax.jit
        def _infer(params, batch_stats, images, states, actions_zero, text_tokens, attention_mask):
            variables = {"params": params, "batch_stats": batch_stats}
            # train=False ⇒ deterministic; no mutable collections
            return self.model_def.apply(
                variables,
                images, states, actions_zero,
                text_tokens, attention_mask,
                train=False
            )
        self._infer = _infer

    def reset(self):
        self.img_queue.clear()
        self.gripper_queue.clear()
        self.state_queue.clear()
        self.text_queue.clear()
        self.cnt = 0
        # Reset gripper state to closed
        self.gripper_state = np.array([-1.0], dtype=np.float32)
    
    def _process_image(self, img_obs):
        img = Image.fromarray(img_obs)
        img = img.resize((self.libero_cfg["libero_img_size"], self.libero_cfg["libero_img_size"]))
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Return (H, W, C) format
        return img_array

    
    def step(self, obs, text_tokens):
        img = self._process_image(obs["agentview_image"])
        grip = self._process_image(obs["robot0_eye_in_hand_image"])

        # proprio: pos(3) + euler(3) + gripper(1 or qpos(1))
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])
        if not self.gripper_width:
            st = np.concatenate([state_pos, state_ori, self.gripper_state]).astype(np.float32)  # (7,)
        else:
            st = np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']]).astype(np.float32)  # (8,)
        st_t = jnp.asarray(st)[None, None, :]  # (1,1,7/8)

        # push queues
        # Images are (H, W, C), need to add batch and time dims: (H, W, C) -> (1, 1, H, W, C)
        img_with_dims = jnp.asarray(img)[None, None, ...]  # (1, 1, H, W, C)
        grip_with_dims = jnp.asarray(grip)[None, None, ...]  # (1, 1, H, W, C)
        self.img_queue.append(img_with_dims)
        self.gripper_queue.append(grip_with_dims)
        self.state_queue.append(st_t)
        if len(self.text_queue) == 0:
            tt = jnp.asarray(text_tokens)  # Should be (1, 77) or (77,)
            if tt.ndim == 1:
                tt = tt[None, :]  # Make it (1, 77)
            # Add time dimension: (1, 77) -> (1, 1, 77)
            tt = tt[:, None, :]  # (1, 1, 77)
            for _ in range(self.history_len):
                self.text_queue.append(tt)

        # build history (pad by repeating last)
        # Concatenate along time dimension (axis=1)
        image_primary = jnp.concatenate(list(self.img_queue), axis=1)    # (1, T, H, W, C)
        image_wrist   = jnp.concatenate(list(self.gripper_queue), axis=1)# (1, T, H, W, C)
        state_hist    = jnp.concatenate(list(self.state_queue), axis=1)  # (1, T, S)
        text_hist     = jnp.concatenate(list(self.text_queue), axis=1)    # (1, T, 77)

        Tcur = int(image_primary.shape[1])
        if Tcur < self.history_len:
            need = self.history_len - Tcur
            image_primary = jnp.concatenate([image_primary,
                                             jnp.repeat(image_primary[:, -1:], repeats=need, axis=1)], axis=1)
            image_wrist   = jnp.concatenate([image_wrist,
                                             jnp.repeat(image_wrist[:, -1:], repeats=need, axis=1)], axis=1)
            state_hist    = jnp.concatenate([state_hist,
                                             jnp.repeat(state_hist[:, -1:], repeats=need, axis=1)], axis=1)

        # combine two views to (B=1, Ni=2, T, H, W, C) - model expects channels last!
        images = jnp.stack([image_primary, image_wrist], axis=1)
        B, Ni, T, H, W, C = images.shape  # Fixed: Model expects (B, Ni, T, H, W, C) not (B, Ni, T, C, H, W)

        # attention mask (L,L)
        attn = generate_attention_mask(T, Ni + 1 + 1, self.action_pred_steps)
        attn = jnp.asarray(attn, dtype=bool)

        # zero action context (same shape used in training API) - use T (actual sequence length)
        actions_zero = jnp.zeros((1, T, 7), dtype=jnp.float32)

        # model expects text tokens as ints with shape (B, T, 77)
        # In eval_libero.py, text_hist[:, 0, :] removes time dim, but model code shows it expects (B*T, 77)
        # which means reshape(B*T, 77), so we should keep the time dimension and reshape later
        text_hist = text_hist.astype(jnp.int32)  # (1, T, 77)
        # Keep history dimension - model will reshape to (B*T, 77) internally

        # forward (jitted)
        action_pred_arm, action_pred_grip = self._infer(
            self.model_dict["params"], self.model_dict["batch_stats"],
            images, state_hist, actions_zero, text_hist, attn
        )  # shapes: (1,T,k,6) and (1,T,k,1)

        # choose the last available time index (same heuristic as SEER)
        sel = Tcur - 1 if Tcur < self.history_len else -1
        arm = action_pred_arm[:, sel]          # (1,k,6)
        grip = action_pred_grip[:, sel]        # (1,k,1)

        # pick the first-step prediction for execution
        arm0  = jnp.asarray(arm[:, 0])         # (1,6)
        grip0 = jnp.asarray(grip[:, 0])        # (1,1) - raw logits from model
        
        # Apply sigmoid to get probabilities (model outputs logits, training uses sigmoid_binary_cross_entropy)
        grip0 = jax.nn.sigmoid(grip0)  # (1,1) in [0,1]

        # convert gripper to {-1, +1}
        grip_bin = jnp.where(grip0 > 0.5, 1.0, -1.0)

        action = jnp.concatenate([arm0, grip_bin], axis=-1)   # (1,7)
        action_np = np.array(action[0], dtype=np.float32)

        # track gripper state for next step
        self.gripper_state = np.array([action_np[-1]], dtype=np.float32)

        info_dict = {}
        info_dict["rgb"] = img
        return action_np, info_dict

# Evaluation functions #
def evaluate_libero_task(task, env, obs, model, libero_cfg={}):
    steps = 0
    success = 0
    model.reset()
    goal = task.language

    # Tokenize text once #
    with torch.no_grad():
        # clip.tokenize expects a list of strings
        uniq_tokens = clip.tokenize([goal], truncate=True)   # (1, 77)
        # Move to CPU explicitly to avoid GPU memory accumulation
        uniq_tokens = uniq_tokens.cpu()
    goal_tokens = uniq_tokens.numpy()  # (1, 77)
    
    rgbs = []

    max_eval_steps = libero_cfg.get("libero_eval_max_steps", 400)
    while steps < max_eval_steps:
        action, info_dict = model.step(obs, goal_tokens)
        steps += 1
        rgbs.append(info_dict["rgb"])
        obs, reward, done, info = env.step(action)
        if done:
            success = 1
            break
    env.close()
    return success, rgbs

def eval_libero10(model_dict, libero_path, task_name=None, num_eval_episodes=20, task_num=10, libero_cfg={}):
    """
    Evaluate the model on the Libero 10 task.
    Args:
        model_dict: Dictionary containing the model definition and variables that form model state like params, and batch_stats
        libero_path: Path to the Libero simulator
    """
    # Setup libero env #
    # Minimal single-process version (keeps original structure)
    TASK_NAME = task_name
    task_num = 10 # Libero 10

    benchmark_dict = benchmark.get_benchmark_dict()
    num_sequences = num_eval_episodes * task_num
    eval_sequences = list(range(num_sequences))
    task_suite = benchmark_dict["libero_10"]()

    # Create a model wrapper for evaluation #
    model = JAXModelWrapper(model_dict, libero_cfg=libero_cfg)
    
    results = []
    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes
        task = task_suite.get_task(task_id)
        task_name = task.name

        if TASK_NAME is not None:
            if TASK_NAME != task_name.lower():
                print(f"Skipping task {task_name} because it is not the target task {TASK_NAME}")
                continue
        
        task_bddl_file = os.path.join(f"{libero_path}/libero/libero/bddl_files",
                                      task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": libero_cfg["libero_img_size"],
            "camera_widths": libero_cfg["libero_img_size"],
            "render_gpu_device_id": 0,
        }
        env = OffScreenRenderEnv(**env_args)
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = "libero_10"
        env.reset()
        env.seed(0)

        # set initial state
        init_states_path = os.path.join(
            f"{libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
        )
        import torch  # only for loading the init_state .pt
        init_states = torch.load(init_states_path, weights_only=False) # Version mismatch change to include `weights_only=False`
        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(5):
            env.step(np.zeros(7, dtype=np.float32))

        result, rgbs = evaluate_libero_task(task, env, obs, model, libero_cfg=libero_cfg)
        results.append(result)
        print("results so far:", results)

        # Make 'gifs' directory
        # os.makedirs("gifs", exist_ok=True)
        # save_rgbs_to_gif(rgbs, f"gifs/rgbs_{task_name}_{eval_id}.gif")

    # print aggregate
    # if TASK_NAME is None:
    #     print_and_save([(r, i) for i, r in enumerate(results)], task_suite)

# if __name__ == "__main__":
#     main()