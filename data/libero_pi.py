# pip install -U lerobot torch torchvision
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

try:
    # v0.3+ (older) locations
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    # newer package structure
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _infer_keymap(sample: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """
    Infer canonical keys for images, state, and action from a LeRobot sample.
    Handles both v2 ('image', 'wrist_image', 'state', 'actions') and
    newer 'observation.*' / 'action' naming.
    """
    keys = list(sample.keys())

    # candidate patterns
    image_like = []
    # LIBERO v2 provides 'image' and 'wrist_image' (H,W,C), but LeRobot returns as CHW tensors
    for k in keys:
        kl = k.lower()
        if any(s in kl for s in ["image", "images", "rgb"]):
            # exclude non-visual aux like 'timestamp', 'frame_index'
            if sample[k].ndim >= 3:  # [C,H,W] or [T,C,H,W]
                image_like.append(k)
    # pick a stable order: primary first, wrist second if present
    image_like = sorted(
        image_like,
        key=lambda x: (0 if "wrist" not in x.lower() else 1, x)
    )

    # state key
    state_key = None
    for cand in ["observation.state", "state", "env_state.state", "proprio", "observation.proprio"]:
        if cand in sample:
            state_key = cand
            break
    if state_key is None:
        # last resort: any 1D/low-D float vector not action/timestamp
        for k in keys:
            if "action" in k or "time" in k or "index" in k:
                continue
            t = sample[k]
            if t.dtype.is_floating_point and t.ndim == 1 and 3 <= t.numel() <= 32:
                state_key = k
                break

    # action key
    action_key = "action" if "action" in sample else ("actions" if "actions" in sample else None)
    if action_key is None:
        for k in keys:
            if "action" in k:
                action_key = k
                break

    if not image_like:
        raise KeyError(f"Could not find image keys in sample: {keys}")
    if state_key is None:
        raise KeyError(f"Could not infer state key from sample: {keys}")
    if action_key is None:
        raise KeyError(f"Could not infer action key from sample: {keys}")

    return {"images": image_like, "state": state_key, "action": action_key}


def _get_fps(ds: LeRobotDataset, fallback: float = 10.0) -> float:
    # v3 docs: ds.meta.fps; many v2 datasets also expose fps in meta/info.json
    fps = None
    meta = getattr(ds, "meta", None)
    if meta is not None:
        fps = getattr(meta, "fps", None)
    if fps is None:
        fps = getattr(ds, "fps", None)
    return float(fps) if fps is not None else float(fallback)


def load_libero_lerobot(
    repo_id: str = "physical-intelligence/libero",
    frame_stack: int = 7,           # past frames to include for images/state (includes t=0)
    action_horizon: int = 3,        # future action chunk length (t+0..t+H-1)
    fps_override: Optional[float] = None,
    which_images: Optional[List[str]] = None,  # e.g., ["image"] or ["image","wrist_image"]
    episodes: Optional[List[int]] = None,      # subset of episodes if you want
):
    """
    Returns:
      ds: LeRobotDataset configured with delta_timestamps so indexing yields stacks
      keymap: {"images": [...], "state": <key>, "action": <key>}
      fps: the FPS used to compute timestamps
    """
    # 1) Load once to introspect keys & fps
    tmp = LeRobotDataset(repo_id, episodes=episodes)
    sample = tmp[0]  # random frame is fine; keys are stable across frames
    keymap = _infer_keymap(sample)
    fps = fps_override or _get_fps(tmp, fallback=10.0)

    # choose which image streams to keep
    img_keys = keymap["images"]
    if which_images is not None:
        # filter while keeping order
        img_keys = [k for k in img_keys if any(sel in k for sel in which_images)]

    # 2) Build delta_timestamps (seconds) for stacked observations & action horizon
    # past frames: [-((N-1)/fps), ..., -1/fps, 0.0]
    num_hist = max(1, int(frame_stack))
    past = [-(num_hist - 1 - i) / fps for i in range(num_hist)]

    delta = {}
    for k in img_keys:
        delta[k] = past[:]        # [T, C, H, W] on access
    delta[keymap["state"]] = past[:]

    # actions: current and next (action_horizon-1) steps
    if action_horizon > 0:
        future = [(i) / fps for i in range(action_horizon)]
        delta[keymap["action"]] = future

    # 3) Re-instantiate with delta_timestamps so every __getitem__ returns stacked tensors
    ds = LeRobotDataset(
        repo_id,
        delta_timestamps=delta,
        episodes=episodes,
    )
    return ds, {"images": img_keys, "state": keymap["state"], "action": keymap["action"]}, fps


# ---- Convenience: quick demo / DataLoader wiring ----

def make_dataloader(
    repo_id: str = "physical-intelligence/libero",
    batch_size: int = 16,
    frame_stack: int = 7,
    action_horizon: int = 3,
    num_workers: int = 8,
):
    ds, keys, fps = load_libero_lerobot(
        repo_id=repo_id,
        frame_stack=frame_stack,
        action_horizon=action_horizon,
    )
    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # LeRobot returns tensors already; default_collate usually works,
        # but some users prefer explicit stacking to keep control:
        out = {}
        for k in batch[0].keys():
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=_collate
    )
    return loader, keys, fps
