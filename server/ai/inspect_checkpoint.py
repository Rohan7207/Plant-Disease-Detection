
import torch

checkpoint_path = "server/ai/mobilenetv2_phase2.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

print("Checkpoint contents:")
for key, value in checkpoint.items():
    print(f"\n{key}: {type(value).__name__}")

print("\nNumber of classes:", len(checkpoint["classes"]))

print("\nFirst 10 classes:")
for class_name in checkpoint["classes"][:10]:
    print(class_name)

state = checkpoint["model_state_dict"]

print("\nNumber of saved parameter tensors:", len(state))

total_parameters = sum(
    tensor.numel() for tensor in state.values()
)

print("Total saved parameters:", total_parameters)

print("\nFirst 10 saved parameter names:")
for name in list(state.keys())[:10]:
    print(name, state[name].shape)
