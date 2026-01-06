import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gudhi as gd
from torchvision import datasets, transforms
from model import SimpleCNN
from ph_utils import extract_ph_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm_attack(model, images, labels, epsilon=0.2):
    images.requires_grad = True
    loss = F.cross_entropy(model(images), labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed, 0, 1)

def test(model, loader, attack=False):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if attack:
                images = fgsm_attack(model, images, labels)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# Load data
transform = transforms.ToTensor()
test_data = datasets.MNIST('./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("ph_robust_cnn.pth", map_location=device))
model.eval()

# Accuracy
clean_acc = test(model, test_loader)
adv_acc = test(model, test_loader, attack=True)

print(f"Clean Accuracy: {clean_acc:.2f}%")
print(f"Adversarial Accuracy: {adv_acc:.2f}%")

# Persistence Diagram
sample_img, _ = test_data[0]
diag = extract_ph_features(sample_img)
gd.plot_persistence_diagram(diag)
plt.title("Persistence Diagram (H1)")
plt.show()
