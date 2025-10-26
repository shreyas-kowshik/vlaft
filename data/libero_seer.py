import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time

from utils.argument_utils import get_parser

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union

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

def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}

def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}

def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class BaseLiberoDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "libero",
        small_size: int = 0,
        gripper_width: bool = False,
        load_libero_file: str = "h5", 
        **kwargs: Any,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        assert os.path.exists(f"./data_info/{self.dataset_info}.json")
        with open(f"./data_info/{self.dataset_info}.json", 'r') as f:
        # assert os.path.exists(f"./data_info/libero_10_debug.json")
        # with open(f"./data_info/libero_10_debug.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)
        # breakpoint()

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        self.gripper_width = gripper_width
        self.load_libero_file = load_libero_file

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)

        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """

        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )

        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            str_step_id = str(step_id).zfill(4)
            if self.load_libero_file == "h5":
                other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/other.h5"
                other_file = h5py.File(other_path)
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, str_step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, str_step_id)
            data_dict["rel_actions"] = self.load_action(other_file)
            data_dict["robot_obs"] = self.load_robot_obs(other_file)
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, str_step_id)
            episodes.append(data_dict)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.load_language_instruction(other_file, self.language_mode)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        image_primary = np.array(Image.open(image_primary_path).convert("RGB"))

        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        image_wrist = np.array(Image.open(image_wrist_path).convert("RGB"))

        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_file, language_mode="language_instruction"):
        if self.load_libero_file == "h5":
            language_instruction = other_file[language_mode][()].decode('utf-8')
        elif self.load_libero_file == "npz":
            language_instruction = other_file[language_mode].tobytes().decode('utf-8')
        else:
            raise NotImplementedError

        return language_instruction
        
    def load_action(self, other_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        if self.load_libero_file == "h5":
            action = other_file["action"][()]
        elif self.load_libero_file == "npz":
            action = other_file["action"]
        else:
            raise NotImplementedError
        
        return action

    def load_robot_obs(self, other_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        if self.load_libero_file == "h5":
            robot_obs[:6] = other_file['observation']['tcp_pose'][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file['observation']['gripper_state'][()]
            robot_obs[7:14] = other_file['observation']['proprio'][()]
            if self.gripper_width:
                robot_obs[-2:] = other_file['observation']['gripper_position'][()]
        elif self.load_libero_file == "npz":
            robot_obs[:6] = other_file["observation_tcp_pose"][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file["observation_gripper_state"]
            robot_obs[7:14] = other_file["observation_proprio"]
            if self.gripper_width:
                robot_obs[-2:] = other_file["observation_gripper_position"]
        else:
            raise NotImplementedError      

        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)

        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DiskLiberoDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        gripper_width: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseLiberoDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                gripper_width=gripper_width,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]
    
    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        episode_id = [s["episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)

        if self.act_step != 1:
        
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)

            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_libero_pretrain_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = ["libero_90_converted"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    libero_dataset = DiskLiberoDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info='libero_90_converted',
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    # breakpoint()
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * 1 # args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        num_replicas=1,
        # rank=args.rank,
        rank=0,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)

def get_libero_finetune_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = ["libero_90_converted"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    # breakpoint()

    libero_dataset = DiskLiberoDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info='libero_90_converted',
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * 1 # * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        # num_replicas=args.world_size,
        num_replicas=1,
        # rank=args.rank,
        rank=0,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Minimal setup to build the data loader
    clip_model, image_processor = clip.load("ViT-B/32", device=device)
    parser = get_parser(is_eval=False)
    args = parser.parse_args()
    # print(args)

    dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0, floor=False)
    for data in dataset.dataloader:
        breakpoint()
    breakpoint()