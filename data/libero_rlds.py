# libero_rlds.py
import os
import json
from pathlib import Path
from typing import Dict, Any, Iterable

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
from omegaconf import DictConfig
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R
import rlds

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def repeat_pad_to_len(struct, target_len: int):
    """Right-pad along axis 0 by repeating the last step."""
    def _pad_leaf(x):
        x = tf.convert_to_tensor(x)
        # scalars: leave as-is
        return tf.cond(
            tf.rank(x) == 0,
            lambda: x,
            lambda: _pad_seq(x),
        )

    def _pad_seq(x):
        cur_len = tf.shape(x)[0]
        pad = target_len - cur_len

        def do_pad():
            last = x[-1:]                # (1, ...)
            pad_vals = tf.repeat(last, pad, axis=0)
            return tf.concat([x, pad_vals], axis=0)

        return tf.cond(pad > 0, do_pad, lambda: x)

    return tf.nest.map_structure(_pad_leaf, struct)


def episode_to_windows_with_prefix(episode: dict, window_size: int):
    """
    Produce windows:
      [0,0,0], [0,1,1], [0,1,2], [1,2,3], ..., tail padded
    even when the episode is shorter than window_size.
    """
    steps = episode[rlds.STEPS] if rlds.STEPS in episode else episode["steps"]

    # 1) build prefix windows in python: lengths 1..(K-1)
    prefix_ds = None
    assert window_size > 1, "Window size must be greater than 1"
    for i in range(window_size - 1):
        # dataset with i+1 real steps → single elem → pad
        p = (
            steps
            .take(i + 1)
            .batch(i + 1, drop_remainder=False)
            .map(lambda win, ws=window_size: repeat_pad_to_len(win, ws))
        )
        prefix_ds = p if prefix_ds is None else prefix_ds.concatenate(p)

    # 2) normal sliding windows over the real steps
    main_ds = (
        steps
        .window(window_size, shift=1, drop_remainder=False)      # gives dict-of-datasets
        .flat_map(                                              # <-- FLATTEN, don’t map
            lambda w: (
                tf.data.Dataset
                .zip(w)                                         # dict-of-datasets → dataset-of-dicts
                .batch(window_size, drop_remainder=False)       # (T<=K, ...)
                .map(lambda win, ws=window_size:                # pad tail windows
                     repeat_pad_to_len(win, ws))
            )
        )
    )

    assert prefix_ds is not None, "Prefix dataset is None"
    return prefix_ds.concatenate(main_ds)

class LiberoRldsConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,  # only keyword
        root_dir: str,
        info_path: str,
        image_primary_size: int = 224,
        image_wrist_size: int = 224,
        gripper_width: bool = False,
        **kwargs,
    ):
        super().__init__(version=tfds.core.Version("0.1.3"), **kwargs)
        self.root_dir = root_dir
        self.info_path = info_path
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.gripper_width = gripper_width


class LiberoRlds(tfds.core.GeneratorBasedBuilder):
    """Minimal RLDS-style LIBERO dataset (episodes -> steps).

    Args:
        root_dir: Path to the root directory of the dataset.
        image_primary_size: Size of the primary image.
        image_wrist_size: Size of the wrist image.
    """

    BUILDER_CONFIGS = [
        LiberoRldsConfig(
            name="local_libero",
            description="Local LIBERO episodes in RLDS layout",
            root_dir="/ABS/PATH/TO/LIBERO",  # change in main.py
            info_path="./data_info/libero_10_converted_kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.json",
            image_primary_size=200,
            image_wrist_size=84,
            gripper_width=False,
        )
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        # RLDS says: top level = episode, inside = steps[...]  :contentReference[oaicite:1]{index=1}
        return tfds.core.DatasetInfo(
            builder=self,
            description="LIBERO trajectories exported as RLDS episodes.",
            features=tfds.features.FeaturesDict(
                {
                    "episode_id": tfds.features.Text(),
                    "steps": tfds.features.Dataset(
                        {
                            "is_first": tf.bool,
                            "is_last": tf.bool,
                            "is_terminal": tf.bool,
                            "observation": tfds.features.FeaturesDict(
                                {
                                    # will be height x width x 3 uint8
                                    "rgb_static": tfds.features.Image(shape=(None, None, 3)),
                                    "rgb_gripper": tfds.features.Image(shape=(None, None, 3)),
                                    # 15 and 24 taken from your DictConfig
                                    "robot_obs": tfds.features.Tensor(shape=(15,), dtype=tf.float32),
                                    "scene_obs": tfds.features.Tensor(shape=(24,), dtype=tf.float32),
                                    "language": tfds.features.Text(),
                                }
                            ),
                            # 7-dim rel action is common in LIBERO conversions
                            "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                            "reward": tf.float32,
                            "discount": tf.float32,
                            "step_id": tf.float32,
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # nothing to download; we just read from local disk
        root = Path(self.builder_config.root_dir).expanduser()
        # this is your "./data_info/<dataset>.json" list of [episode_id, num_steps]
        meta_path = Path(self.builder_config.info_path).expanduser()
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected {meta_path} to exist")

        with meta_path.open("r") as f:
            episode_info = json.load(f)  # list of [episode_id, num_steps]

        return {
            "train": self._generate_examples(root, episode_info),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_image(self, path: Path, size: int) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if size is not None and size > 0:
            img = img.resize((size, size))
        return np.array(img, dtype=np.uint8)

    def _load_other_h5(self, path: Path) -> h5py.File:
        return h5py.File(path, "r")

    def _load_robot_obs(self, f) -> np.ndarray:
        """
        Reproduce the PyTorch-side logic:

            robot_obs[:6] = tcp_pose[:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            robot_obs[3:6] = euler.as_euler("xyz", degrees=False)
            robot_obs[-1] = gripper_state
            robot_obs[7:14] = proprio
            if gripper_width:
                robot_obs[-2:] = gripper_position
        """
        robot_obs = np.zeros(15, dtype=np.float32)

        # 1) position + euler-ish orientation
        tcp_pose = f["observation"]["tcp_pose"][:6]
        robot_obs[:6] = tcp_pose

        # 2) normalize the orientation part via scipy rotation
        #    (build rotation from xyz-radians, ask it back as same order) :contentReference[oaicite:1]{index=1}
        rot = R.from_euler("xyz", robot_obs[3:6], degrees=False)
        robot_obs[3:6] = rot.as_euler("xyz", degrees=False).astype(np.float32)

        # 3) write proprio (7..13)
        robot_obs[7:14] = f["observation"]["proprio"][()].astype(np.float32)

        # 4) gripper state goes to the very last slot
        robot_obs[-1] = np.float32(f["observation"]["gripper_state"][()])

        # 5) optional gripper width: overwrite last 2 slots
        if self.builder_config.gripper_width:
            grip_pos = f["observation"]["gripper_position"][()].astype(np.float32)
            # put in the last two locations
            robot_obs[-2:] = grip_pos

        return robot_obs

    def _load_action(self, f: h5py.File) -> np.ndarray:
        act = np.array(f["action"][...], dtype=np.float32)
        # pad / trim to 7 to have static shape
        if act.shape[0] < 7:
            act = np.pad(act, (0, 7 - act.shape[0]))
        elif act.shape[0] > 7:
            act = act[:7]
        return act

    def _generate_examples(
        self,
        root: Path,
        episode_info: Iterable[Any],
    ):
        """
        episode_info: list like [["000000", 123], ["000001", 87], ...]
        Directory layout assumed:
        root/
          episodes/<episode_id>/steps/0000/image_primary.jpg
                                        /image_wrist.jpg
                                        /other.h5
        """
        for epi, (episode_id, num_steps) in enumerate(episode_info):
            steps = []
            for step_id in range(num_steps):
                sid = f"{step_id:04d}"
                step_dir = root / "episodes" / episode_id / "steps" / sid
                other_f = self._load_other_h5(step_dir / "other.h5")

                rgb_static = self._load_image(
                    step_dir / "image_primary.jpg",
                    self.builder_config.image_primary_size,
                )
                rgb_gripper = self._load_image(
                    step_dir / "image_wrist.jpg",
                    self.builder_config.image_wrist_size,
                )

                step = {
                    "is_first": step_id == 0,
                    "is_last": step_id == (num_steps - 1),
                    "is_terminal": step_id == (num_steps - 1),
                    "observation": {
                        "rgb_static": rgb_static,
                        "rgb_gripper": rgb_gripper,
                        "robot_obs": self._load_robot_obs(other_f),
                        # you had zeros in your PyTorch code for scene_obs
                        "scene_obs": np.zeros((24,), dtype=np.float32),
                        "language": other_f["language_instruction"][()].decode("utf-8"),
                    },
                    "action": self._load_action(other_f),
                    "reward": np.float32(0.0),
                    "discount": np.float32(1.0),
                    "step_id": np.float32(step_id),
                }
                steps.append(step)
                other_f.close()

            yield episode_id, {
                "episode_id": episode_id,
                "steps": steps,
            }
