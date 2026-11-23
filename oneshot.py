import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

def load_images(folder_path):
    images = []
    labels = []
    for idx, class_name in enumerate(sorted(os.listdir(folder_path))):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
                labels.append(idx)
    return images, labels

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("L")
    return transform(image)

@torch.no_grad()
def run_oneshot(model, run_folder, transform):
    training_path = os.path.join(run_folder, "training")
    test_path = os.path.join(run_folder, "test")

    labels_path = Path(run_folder) / "class_labels.txt"
    if labels_path.is_file():
        pairs = [line.split() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError(f"Invalid one-shot class labels: {labels_path}")
        support_names = sorted({Path(pair[1]).name for pair in pairs})
        if len({Path(pair[0]).name for pair in pairs}) != len(pairs):
            raise ValueError(f"Duplicate one-shot query filenames: {labels_path}")
        support_images = [str(Path(training_path) / name) for name in support_names]
        support_labels = support_names
        query_images = [str(Path(test_path) / Path(pair[0]).name) for pair in pairs]
        query_labels = [Path(pair[1]).name for pair in pairs]
        for image in support_images + query_images:
            if not Path(image).is_file():
                raise FileNotFoundError(f"One-shot image referenced by {labels_path} is missing: {image}")
    else:
        support_images, support_labels = load_images(training_path)
        query_images, query_labels = load_images(test_path)

    if not support_images or not query_images:
        raise ValueError(f"One-shot run has no support or query images: {run_folder}")

    support_tensors = [preprocess_image(img, transform) for img in support_images]
    query_tensors = [preprocess_image(img, transform) for img in query_images]

    support_tensors = torch.stack(support_tensors)
    query_tensors = torch.stack(query_tensors)

    device = next(model.parameters()).device
    support_tensors = support_tensors.to(device)
    query_tensors = query_tensors.to(device)
    encoder = getattr(model, 'sub_forward', model)
    support_embeddings = encoder(support_tensors)
    query_embeddings = encoder(query_tensors)
    nearest_support = torch.cdist(query_embeddings, support_embeddings).argmin(dim=1).cpu().tolist()
    predicted_labels = [support_labels[index] for index in nearest_support]
    correct = sum(predicted == actual for predicted, actual in zip(predicted_labels, query_labels))
    total = len(query_labels)

    accuracy = correct / total
    return accuracy
