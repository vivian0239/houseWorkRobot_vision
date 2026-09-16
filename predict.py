import torch

from PIL import Image

from torchvision import transforms

from model import SimpleClassifier


CLASSES = [
    "blanket",
    "bowl",
    "clothes",
    "lego",
    "plate",
    "remote",
    "shoe",
    "sock",
    "tissue",
    "toy"
]


model = SimpleClassifier(
    num_classes=10
)

model.load_state_dict(
    torch.load(
        "robot_classifier.pth",
        map_location="cpu"
    )
)

model.eval()

img = Image.open(
    "test.jpg"
).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = transform(img).unsqueeze(0)

with torch.no_grad():

    pred = model(img)

    idx = pred.argmax(1).item()

print(
    CLASSES[idx]
)
