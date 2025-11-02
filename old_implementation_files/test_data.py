# from libero_dataset import load_libero_lerobot, make_dataloader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("physical-intelligence/libero", "/data/user_data/skowshik/datasets/libero_converted")

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("physical-intelligence/libero")

# BASE_PATH = "/data/user_data/skowshik/datasets/libero_converted"

# # 1) Get a dataset with 7-frame history and 3-step action chunk
# ds, keys, fps = load_libero_lerobot(
#     repo_id="physical-intelligence/libero",
#     frame_stack=7,        # past 7 frames: ~0.6–0.7 s if fps≈10
#     action_horizon=3,     # predict 3 future actions
# )

# print("FPS:", fps)
# print("Image streams:", keys["images"])
# print("State key:", keys["state"], " Action key:", keys["action"])

# sample = ds[12345]
# # Shapes:
# #   sample[keys["images"][0]] -> [T, C, H, W]
# #   sample[keys["state"]]     -> [T, D_state]
# #   sample[keys["action"]]    -> [H, D_action]
# print({k: tuple(v.shape) for k, v in sample.items() if "image" in k or k in (keys["state"], keys["action"])})

# # 2) With a DataLoader:
# loader, keys, fps = make_dataloader("physical-intelligence/libero", batch_size=32)
# batch = next(iter(loader))
# #   batch[keys["images"][0]] -> [B, T, C, H, W]
# #   batch[keys["state"]]     -> [B, T, D_state]
# #   batch[keys["action"]]    -> [B, H, D_action]
