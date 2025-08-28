# import torch

# Load the checkpoint file
# checkpoint = torch.load(r"E:\dc\new.ckpt", map_location='cpu')

# # Print the keys in the checkpoint
# print("Keys in checkpoint:")
# for key in checkpoint.keys():
#     print(f"  {key}")

# # Print detailed information about each key
# print("\nDetailed contents:")
# for key, value in checkpoint.items():
#     if isinstance(value, torch.Tensor):
#         print(f"{key}: Tensor of shape {value.shape}, dtype {value.dtype}")
#     elif isinstance(value, dict):
#         print(f"{key}: Dictionary with {len(value)} items")
#         for subkey in list(value.keys())[:5]:  # Show first 5 subkeys
#             print(f"    {subkey}")
#         if len(value) > 5:
#             print(f"    ... and {len(value) - 5} more items")
#     else:
#         print(f"{key}: {type(value)} - {value}")
import deepchem as dc
from deepchem.molnet import load_clintox
tasks, datasets, _ = load_clintox(reload=False)
_, valid_dataset, test = datasets
model = dc.models.MultitaskClassifier(n_tasks=2,
                                           n_features=1024,
                                           layer_sizes=[1000],
                                           dropouts=0.2,
                                           learning_rate=0.0001,
                                           device="cpu",
                                           batch_size=16)
model.fit(test, nb_epoch=3)
# state_dict = model.model.state_dict()
state_dict = model._pytorch_optimizer.state_dict()
# Print the keys in the state_dict
# print("Keys in state_dict:", len(state_dict))
# for key in state_dict.keys():
#     print(f"  {key}")
# # Print detailed information about each key in state_dict
# print("\nDetailed contents of state_dict:")
# for key, value in state_dict.items():
#     if 'optimizer' in key.lower():
#         if isinstance(value, torch.Tensor):
#             print(f"{key}: Tensor of shape {value.shape}, dtype {value.dtype}")
#         else:
#             print(f"{key}: {type(value)} - {value}")
print(state_dict)