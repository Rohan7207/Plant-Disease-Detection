
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activations
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, inputs, output):
        self.activations = output.detach()

        print(
            "Activation:",
            "shape =", tuple(output.shape),
            "min =", round(self.activations.min().item(), 6),
            "max =", round(self.activations.max().item(), 6),
            "mean =", round(self.activations.mean().item(), 6),
            "positive values =", int((output > 0).sum().item())
        )

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

        print(
            "Gradient:",
            "min =", round(self.gradients.min().item(), 6),
            "max =", round(self.gradients.max().item(), 6),
            "mean =", round(self.gradients.mean().item(), 6),
            "nonzero =", int((self.gradients != 0).sum().item())
        )


    def generate(self, input_tensor, target_class):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        score = output[:, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        print(
            "Raw CAM:",
            "min =", round(cam.min().item(), 6),
            "max =", round(cam.max().item(), 6),
            "mean =", round(cam.mean().item(), 6),
            "positive pixels =", int((cam > 0).sum().item())
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)

        return cam.cpu().numpy()


    def generate_layercam(self, input_tensor, target_class):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        score = output[:, target_class]
        score.backward()

        # Use positive gradients at each spatial location.
        positive_gradients = F.relu(self.gradients)

        weighted_activations = positive_gradients * self.activations

        # Diagnostic: retain only positive channel contributions
        positive_contributions = F.relu(weighted_activations)

        cam = positive_contributions.sum(dim=1, keepdim=True)

        print(
            "LayerCAM positive contribution sum:",
            "min =", round(cam.min().item(), 6),
            "max =", round(cam.max().item(), 6),
            "positive positions =", int((cam > 0).sum().item())
        )

        # Upscale to model input resolution.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize to [0, 1].
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)

        return cam.detach().cpu().numpy()


    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
