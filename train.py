import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import SimpleCNN
from ph_utils import ph_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm_attack(model, images, labels, epsilon=0.2):
    images.requires_grad = True
    loss = F.cross_entropy(model(images), labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed, 0, 1)

def train(model, loader, optimizer, epoch, lambda_ph=0.1):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        adv_images = fgsm_attack(model, images, labels)

        output = model(images)
        ce_loss = F.cross_entropy(output, labels)

        ph_loss = torch.mean(torch.stack([
            ph_distance(images[i], adv_images[i])
            for i in range(images.size(0))
        ]))

        loss = ce_loss + lambda_ph * ph_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

# Dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# Model
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1, 4):
    train(model, train_loader, optimizer, epoch)

# Save model
torch.save(model.state_dict(), "ph_robust_cnn.pth")
print("Model saved as ph_robust_cnn.pth")
