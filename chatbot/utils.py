import torch
from torchvision import models, transforms
from PIL import Image

# Load model once (for performance)
model = models.resnet50(pretrained=True)
model.eval()

# ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import requests
labels = requests.get(LABELS_URL).text.strip().split("\n")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
    _, index = output[0].max(0)
    return labels[index]
