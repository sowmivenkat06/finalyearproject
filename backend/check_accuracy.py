import torch

checkpoint = torch.load(
    "checkpoints/checkpoint_epoch_7.pth",
    map_location="cpu"
)

print("Checkpoint keys:")
print(checkpoint.keys())

print("\nTraining History:")
for x in checkpoint["training_history"]:
    print(x)