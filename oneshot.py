import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

def run_oneshot(model, run_folder, transform):
    training_path = os.path.join(run_folder, "training")
    test_path = os.path.join(run_folder, "test")

    support_images, support_labels = load_images(training_path)
    query_images, query_labels = load_images(test_path)

    support_tensors = [preprocess_image(img, transform) for img in support_images]
    query_tensors = [preprocess_image(img, transform) for img in query_images]

    support_tensors = torch.stack(support_tensors)
    query_tensors = torch.stack(query_tensors)

    correct = 0
    total = 0

    with torch.no_grad():
        for query_tensor, query_label in zip(query_tensors, query_labels):
            query_tensor = query_tensor.unsqueeze(0)  # add batch-dim

            distances = []
            for support_tensor in support_tensors:
                support_tensor = support_tensor.unsqueeze(0)
                distance = model(query_tensor, support_tensor)
                distances.append(distance.item())

            predicted_label = support_labels[torch.argmin(torch.tensor(distances))]

            if predicted_label == query_label:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy
