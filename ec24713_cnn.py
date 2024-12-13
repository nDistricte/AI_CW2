import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


train_set = torchvision.datasets.FashionMNIST(root = "FashionMNIST/", train=True,
download=True,
transform=transforms.ToTensor())

validation_set = torchvision.datasets.FashionMNIST(root = "FashionMNIST/", train=False,
download=True,
transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)

# Fix the seed to be able to get the same randomness across runs and
# hence reproducible outcomes
torch.manual_seed(0)

# Define the CNN model

class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1) # 1 input channel, 32 output channels, 5x5 kernel, input size 28x28
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12x12x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1) # 8x8x64
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 4x4x64
        self.fc1 = nn.Linear(4*4*64, 1024) # 1024 output features
        self.fc2 = nn.Linear(1024, 256) # 256 output features
        self.output = nn.Linear(256, 10) # 10 output features

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(-1, 4*4*64)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.output(x)

        return x
model = myCNN()
EPOCHS = 30
LR = 0.1

def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(module.weight)

model.apply(init_weights)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_model = None

# Training and validation loop
for epoch in range(EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train_samples = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = output.argmax(dim=1)
        correct_train += (preds == target).sum().item()
        total_train_samples += target.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train_samples

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val_samples = 0

    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)
            loss = criterion(output, target)

            total_val_loss += loss.item()
            preds = output.argmax(dim=1)
            correct_val += (preds == target).sum().item()
            total_val_samples += target.size(0)

    avg_val_loss = total_val_loss / len(validation_loader)
    val_accuracy = correct_val / total_val_samples

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
        f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}"
    )
train_accuracy = correct_train / total_train_samples
final_val_accuracy = correct_val / total_val_samples
print(f"Final Training Accuracy: {train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
torch.save(model.state_dict(), "ec24713_best_cnn.pth")
print("Model saved as cnn_best.pth")
    

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.savefig("accuracy.png")
plt.show()


# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.savefig("loss.png")
plt.show()

