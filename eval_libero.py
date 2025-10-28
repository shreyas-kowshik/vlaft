# eval_libero_jax.py  (minimal JAX port of SEER's eval_libero.py)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/home/skowshik/vla/codebase/envs/LIBERO')

import types
m = types.ModuleType("robosuite.macros_private")
m.FILE_LOGGING_LEVEL = None                # turn off file logging
# If your robosuite expects a path var too, set both:
m.FILE_LOGGING_PATH = os.path.expanduser("~/.cache/robosuite/robosuite.log")
os.makedirs(os.path.dirname(m.FILE_LOGGING_PATH), exist_ok=True)
sys.modules["robosuite.macros_private"] = m

# ----------------- runtime env (same as SEER) -----------------
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['MUJOCO_GL'] = 'egl'

# ----------------- CONFIG: point to your JAX checkpoint -----------------
MODEL_PATH = "/data/user_data/skowshik/datasets/libero_pro/checkpoints/jit_20251028_024223/epoch_7/"  # <- set this

from pathlib import Path
import copy
import numpy as np
from collections import deque
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from PIL import Image
from utils.argument_utils import get_parser

# SEER preprocessing (reuse)
from utils.data_utils import preprocess_image, preprocess_text_calvin

# LIBERO env
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ----------------- JAX / Flax imports -----------------
import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
import optax

import torch
import clip

# Your model (same one used in training)
from models.bc_simple import BCSimple, generate_attention_mask, GPTConfig

# ----------------- tiny TrainState (mirrors training) -----------------
nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: callable = nonpytree_field()
    model_def: any = nonpytree_field()
    params: any = None
    batch_stats: any = None
    tx: any = nonpytree_field(default=None)
    opt_state: any = None
    rng: any = None

# ----------------- utils -----------------
def quaternion_to_euler(q):
    rot = R.from_quat(q)
    return rot.as_euler('xyz', degrees=False)

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

# ----------------- JAX model wrapper -----------------
class JAXModelWrapper:
    """
    Drop-in-ish replacement for SEER's torch ModelWrapper.
    Keeps queues and padding logic the same, but runs a jitted Flax model.
    """
    def __init__(
        self,
        model_def: BCSimple,
        state: TrainState,
        tokenizer,                # kept for reuse of preprocess_text_calvin
        image_processor,
        history_len=10,
        libero_eval_max_steps=600,
        action_pred_steps=3,
        gripper_width=False,
        use_ensembling=False,
        ensembling_temp=0.01,
    ):
        self.model_def = model_def
        self.state = state
        self.history_len = history_len
        self.action_pred_steps = action_pred_steps
        self.gripper_width = gripper_width
        self.libero_eval_max_steps = libero_eval_max_steps
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp

        # preprocessing fns from SEER
        self.text_process_fn  = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)

        # queues
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.cnt = 0
        self._reset_all_time_actions()

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

    def _reset_all_time_actions(self):
        # only used if ensembling; keep on host (numpy) then convert when needed
        if self.use_ensembling:
            self.all_time_actions = np.zeros(
                (self.libero_eval_max_steps, self.libero_eval_max_steps + self.action_pred_steps, 7),
                dtype=np.float32
            )
        self.gripper_state = np.array([-1.0], dtype=np.float32)

    def reset(self):
        self.img_queue.clear()
        self.gripper_queue.clear()
        self.state_queue.clear()
        self.text_queue.clear()
        self._reset_all_time_actions()
        self.cnt += 1

    def step(self, obs, goal, timestep):
        # --- preprocess images to (1,1,C,H,W) tensors (SEER style), then to JAX ---
        img = Image.fromarray(obs["agentview_image"])

        img_t = self.image_process_fn([img]).unsqueeze(1)     # (1,1,C,H,W)

        grip = Image.fromarray(obs["robot0_eye_in_hand_image"])
        grip_t = self.image_process_fn([grip]).unsqueeze(1)   # (1,1,C,H,W)

        # text -> tokens (1,1,77); convert to jnp.int32 later
        text_toks_t = self.text_process_fn([goal]).unsqueeze(1)

        # proprio: pos(3) + euler(3) + gripper(1 or qpos(1))
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])
        if not self.gripper_width:
            st = np.concatenate([state_pos, state_ori, self.gripper_state]).astype(np.float32)  # (7,)
        else:
            st = np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']]).astype(np.float32)  # (8,)
        st_t = jnp.asarray(st)[None, None, :]  # (1,1,7/8)

        # push queues
        self.img_queue.append(jnp.asarray(np.array(img_t)))
        self.gripper_queue.append(jnp.asarray(np.array(grip_t)))
        self.state_queue.append(st_t)
        if len(self.text_queue) == 0:
            tt = jnp.asarray(np.array(text_toks_t))
            for _ in range(self.history_len):
                self.text_queue.append(tt)

        # build history (pad by repeating last)
        image_primary = jnp.concatenate(list(self.img_queue), axis=1)    # (1,T,C,H,W)
        image_wrist   = jnp.concatenate(list(self.gripper_queue), axis=1)# (1,T,C,H,W)
        state_hist    = jnp.concatenate(list(self.state_queue), axis=1)  # (1,T,S)
        text_hist     = jnp.concatenate(list(self.text_queue), axis=1)   # (1,T,77)

        Tcur = int(image_primary.shape[1])
        if Tcur < self.history_len:
            need = self.history_len - Tcur
            image_primary = jnp.concatenate([image_primary,
                                             jnp.repeat(image_primary[:, -1:], repeats=need, axis=1)], axis=1)
            image_wrist   = jnp.concatenate([image_wrist,
                                             jnp.repeat(image_wrist[:, -1:], repeats=need, axis=1)], axis=1)
            state_hist    = jnp.concatenate([state_hist,
                                             jnp.repeat(state_hist[:, -1:], repeats=need, axis=1)], axis=1)

        # combine two views to (B=1, Ni=2, T, C, H, W)
        images = jnp.stack([image_primary, image_wrist], axis=1)
        B, Ni, T, C, H, W = images.shape

        # attention mask (L,L)
        attn = generate_attention_mask(T, Ni + 1 + 1, self.action_pred_steps)
        attn = jnp.asarray(attn, dtype=bool)

        # zero action context (same shape used in training API)
        actions_zero = jnp.zeros((1, self.history_len, 7), dtype=jnp.float32)

        # model expects text tokens as ints
        text_hist = text_hist.astype(jnp.int32)
        text_hist = text_hist[:, 0, :] # No history dimension here

        # forward (jitted)
        action_pred_arm, action_pred_grip = self._infer(
            self.state.params, self.state.batch_stats,
            images, state_hist, actions_zero, text_hist, attn
        )  # shapes: (1,T,k,6) and (1,T,k,1)

        # choose the last available time index (same heuristic as SEER)
        sel = Tcur - 1 if Tcur < self.history_len else -1
        arm = action_pred_arm[:, sel]          # (1,k,6)
        grip = action_pred_grip[:, sel]        # (1,k,1)

        # pick the first-step prediction for execution
        arm0  = jnp.asarray(arm[:, 0])         # (1,6)
        grip0 = jnp.asarray(grip[:, 0])        # (1,1) in [0,1] due to sigmoid

        # convert gripper to {-1, +1}
        grip_bin = jnp.where(grip0 > 0.5, 1.0, -1.0)

        action = jnp.concatenate([arm0, grip_bin], axis=-1)   # (1,7)
        action_np = np.array(action[0], dtype=np.float32)

        # track gripper state for next step
        self.gripper_state = np.array([action_np[-1]], dtype=np.float32)

        info_dict = {}
        info_dict["rgb"] = img
        return action_np, info_dict

def save_rgbs_to_gif(rgbs, path):
    # rgbs = [Image.fromarray(rgb) for rgb in rgbs]
    rgbs[0].save(path, save_all=True, append_images=rgbs[1:], duration=100, loop=0)

# ----------------- eval helpers (kept close to SEER) -----------------
def evaluate_libero_task(task, env, obs, args, model: JAXModelWrapper):
    steps = 0
    success = 0
    model.reset()
    goal = task.language
    rgbs = []
    while steps < args.libero_eval_max_steps:
        action, info_dict = model.step(obs, goal, steps)
        steps += 1
        rgbs.append(info_dict["rgb"])
        obs, reward, done, info = env.step(action)
        if done:
            success = 1
            break
    env.close()
    return success, rgbs

def evaluate_policy_ddp(args, model: JAXModelWrapper):
    # Minimal single-process version (keeps original structure)
    # TASK_NAME = None
    TASK_NAME = "kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = 1
    device_id = 0
    results = []
    if "libero" in args.finetune_type:
        if args.finetune_type == "libero_10":
            global num_eval_episodes, task_num
            num_eval_episodes = 2
            task_num = 10
            NUM_SEQUENCES = num_eval_episodes * task_num
            eval_sequences = list(range(NUM_SEQUENCES))
            eval_sequences = tqdm(eval_sequences)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes
        task = task_suite.get_task(task_id)
        task_name = task.name
        # breakpoint()

        if TASK_NAME is not None:
            if TASK_NAME != task_name.lower():
                print(f"Skipping task {task_name} because it is not the target task {TASK_NAME}")
                continue
        
        task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files",
                                      task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.libero_img_size,
            "camera_widths": args.libero_img_size,
            "render_gpu_device_id": device_id
        }
        env = OffScreenRenderEnv(**env_args)
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = args.finetune_type
        env.reset()
        env.seed(args.seed)

        # set initial state
        init_states_path = os.path.join(
            f"{args.libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
        )
        import torch  # only for loading the init_state .pt
        init_states = torch.load(init_states_path)
        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(5):
            env.step(np.zeros(7, dtype=np.float32))

        result, rgbs = evaluate_libero_task(task, env, obs, args, model)
        results.append(result)
        print("results so far:", results)

        # Make 'gifs' directory
        os.makedirs("gifs", exist_ok=True)
        save_rgbs_to_gif(rgbs, f"gifs/rgbs_{task_name}_{eval_id}.gif")

    # print aggregate
    if TASK_NAME is None:
        print_and_save([(r, i) for i, r in enumerate(results)], task_suite)

def print_and_save(result_list, task_suite):
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        this_result_list = np.array(this_result_list)
        avg_success = np.mean(this_result_list, axis=0)[0]
        task = task_suite.get_task(j)
        task_name = task.name
        print(f"Success rates for task {j} {task_name}: {avg_success * 100:.1f}%")

# ----------------- entry: build model, load checkpoint, wrap -----------------
def build_jax_model_and_state(args, image_size, seq_len, action_pred_steps, gripper_width=False):
    """
    Rebuild the Flax model the same way as in training and restore the saved TrainState.
    """
    hidden_dim = 768
    num_layers = 6
    num_heads = 8
    Ni = 2
    action_dim = 7 + (1 if gripper_width else 0)  # 7 by default; keep 7 if gripper_width=False
    state_dim = 8 if gripper_width else 7

    gpt_conf = GPTConfig(
        block_size=seq_len * (Ni + 1 + 1 + 3),
        num_layers=num_layers, num_heads=num_heads,
        num_embeds=hidden_dim, use_bias=True, dtype=None,
    )

    model_def = BCSimple(
        sequence_length=seq_len,
        input_image_size=image_size,
        action_pred_steps=action_pred_steps,
        transformer_layers=num_layers,
        hidden_dim=hidden_dim,
        transformer_heads=num_heads,
        gripper_width=gripper_width,
        num_images=Ni,
        action_dim=action_dim,
        state_dim=state_dim,
        config=gpt_conf,
    )

    # shape-only init with zeros
    H = W = image_size
    key = jax.random.PRNGKey(0)
    dummy_images = jnp.zeros((1, Ni, seq_len, 3, H, W), dtype=jnp.float32)
    dummy_states = jnp.zeros((1, seq_len, state_dim), dtype=jnp.float32)
    dummy_actions = jnp.zeros((1, seq_len, action_dim), dtype=jnp.float32)
    dummy_tokens = jnp.zeros((1, 77), dtype=jnp.int32)
    dummy_mask = jnp.array(generate_attention_mask(seq_len, Ni + 1 + 1, action_pred_steps), dtype=bool)

    variables = model_def.init({'params': key, 'dropout': key},
                               dummy_images, dummy_states, dummy_actions,
                               dummy_tokens, dummy_mask, train=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    tx = optax.adam(1e-4)
    template = TrainState(
        0, apply_fn=model_def.apply, model_def=model_def,
        params=params, batch_stats=batch_stats, tx=tx,
        opt_state=tx.init(params), rng=None,
    )

    # Restore (works whether MODEL_PATH is a dir or a specific file prefix)
    restored = checkpoints.restore_checkpoint(MODEL_PATH, target=template)
    return model_def, restored

def eval_one_epoch_libero_jax(args, image_processor, tokenizer):
    hist_len = args.sequence_length
    model_def, state = build_jax_model_and_state(
        args,
        image_size=args.libero_img_size,
        seq_len=hist_len,
        action_pred_steps=args.action_pred_steps,
        gripper_width=args.gripper_width,
    )
    wrapped = JAXModelWrapper(
        model_def=model_def,
        state=state,
        tokenizer=tokenizer,
        image_processor=image_processor,
        history_len=hist_len,
        libero_eval_max_steps=args.libero_eval_max_steps,
        action_pred_steps=args.action_pred_steps,
        gripper_width=args.gripper_width,
        use_ensembling=getattr(args, "eval_libero_ensembling", False),
        ensembling_temp=getattr(args, "ensembling_temp", 0.01),
    )
    evaluate_policy_ddp(args, wrapped)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, image_processor = clip.load("ViT-B/32", device=device)

    parser = get_parser(is_eval=True)
    args = parser.parse_args()
    # image_processor = functools.partial(preprocess_image, image_processor=image_processor)
    eval_one_epoch_libero_jax(args, image_processor, clip)