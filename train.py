import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model import SimpleClassifier


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

model = SimpleClassifier(
    num_classes=len(dataset.classes)
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(20):

    total_loss = 0

    for imgs,labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        pred = model(imgs)

        loss = loss_fn(
            pred,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        epoch,
        total_loss
    )

torch.save(
    model.state_dict(),
    "robot_classifier.pth"
)
