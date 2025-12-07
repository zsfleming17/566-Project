# Generates Grad-CAM heatmaps showing what the model focuses on
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Import from training script
from train_baseline import make_model, MEAN, STD

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # backprop from predicted class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        
        # weight activation maps by gradient importance
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), class_idx, output.softmax(dim=1)[0, class_idx].item()


def load_and_preprocess(image_path, size=224):
    """Load image and return both tensor and original image."""
    img = Image.open(image_path).convert("RGB")
    
    # Transform for model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = transform(img).unsqueeze(0)
    
    # non-normalized version for display
    display_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
    ])
    display_img = display_transform(img)
    
    return tensor, display_img


def overlay_cam(img, cam, alpha=0.5):
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    img_array = np.array(img) / 255.0
    blended = (1 - alpha) * img_array + alpha * heatmap
    return (blended * 255).astype(np.uint8)

def generate_gradcam_panel(image_paths, model_path, output_path, labels=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layer = model.features[-1] # last conv layer
    gradcam = GradCAM(model, target_layer)
    cls_names = {0: "clear", 1: "obstructed"}
    
    n_images = len(image_paths)
    fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, img_path in enumerate(image_paths):
        tensor, display_img = load_and_preprocess(img_path)
        tensor = tensor.to(device)
        
        cam, pred_idx, confidence = gradcam.generate(tensor)
        overlay = overlay_cam(display_img, cam)
        
        axes[0, i].imshow(display_img)
        axes[0, i].axis("off")
        if labels:
            axes[0, i].set_title(f"True: {labels[i]}", fontsize=10)
        
        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Pred: {cls_names[pred_idx]} ({confidence:.0%})", fontsize=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def generate_single_gradcam(image_path, model_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = make_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    cls_names = {0: "clear", 1: "obstructed"}
    
    tensor, display_img = load_and_preprocess(image_path)
    tensor = tensor.to(device)
    
    cam, pred_idx, confidence = gradcam.generate(tensor)
    overlay = overlay_cam(display_img, cam)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {cls_names[pred_idx]} ({confidence:.0%})")
    axes[1].axis("off")
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    df = pd.read_csv("splits/random_test.csv")
    clear_samples = df[df["label"] == "clear"].head(3)["path"].tolist()
    obst_samples = df[df["label"] == "obstructed"].head(3)["path"].tolist()
    
    # clear panel
    generate_gradcam_panel(
        clear_samples,
        "best_random.pt",
        "figures/gradcam_clear.png",
        labels=["clear"] * 3
    )
    
    # obstructed panel
    generate_gradcam_panel(
        obst_samples,
        "best_random.pt",
        "figures/gradcam_obstructed.png",
        labels=["obstructed"] * 3
    )
    
    # mixed panel
    all_samples = clear_samples + obst_samples
    all_labels = ["clear"] * 3 + ["obstructed"] * 3
    generate_gradcam_panel(
        all_samples,
        "best_random.pt",
        "figures/gradcam_panel.png",
        labels=all_labels
    )
    
    print("\nDone! Check the figures/ folder.")