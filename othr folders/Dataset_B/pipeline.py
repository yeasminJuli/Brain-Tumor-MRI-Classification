import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import random

# =========================
# CONFIGURATION
# =========================

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

DATASET_ROOT = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\preprocessed_testt"
TARGET_CLASS = 'glioma'
NUM_IMAGES = 10

MODEL_TYPE = 'inception'          # 'inception' or 'efficientnet'
MODEL_PATH = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\traing_outputs_inceptionV3\best_model.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# IMAGE PREPROCESSING
# =========================

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Correct for InceptionV3
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# MODEL LOADER
# =========================

def load_inception(model_path, device):
    # Keep aux_logits=True to match training
    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    # Ignore AuxLogits weights to avoid mismatch
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if "AuxLogits" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def load_efficientnet(model_path, device):
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =========================
# IMAGE LOADING
# =========================

def load_n_images(root_dir, class_name, num_images):
    class_dir = os.path.join(root_dir, class_name)
    images = os.listdir(class_dir)
    selected = random.sample(images, min(num_images, len(images)))
    paths = [os.path.join(class_dir, img) for img in selected]
    return paths

# =========================
# PREDICTION FUNCTION
# =========================

def predict_images(model, image_paths, device):
    results = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)  # AuxLogits ignored in eval mode
            probs = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        results.append({
            'image': os.path.basename(path),
            'predicted_class': CLASS_NAMES[pred_idx.item()],
            'confidence': conf.item()
        })
    return results

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    if MODEL_TYPE == 'inception':
        model = load_inception(MODEL_PATH, DEVICE)
    elif MODEL_TYPE == 'efficientnet':
        model = load_efficientnet(MODEL_PATH, DEVICE)
    else:
        raise ValueError("MODEL_TYPE must be 'inception' or 'efficientnet'")

    image_paths = load_n_images(DATASET_ROOT, TARGET_CLASS, NUM_IMAGES)
    predictions = predict_images(model, image_paths, DEVICE)

    print("\n===================================")
    print(f"Images taken from folder: {TARGET_CLASS}")
    print("===================================\n")
    for p in predictions:
        print(f"{p['image']}  -->  Predicted: {p['predicted_class']}  |  Confidence: {p['confidence']:.4f}")
