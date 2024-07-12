import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Assuming you have a dataset and a model defined
# Replace these with your actual dataset and model
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=100, n_features=10):
        # Initialize your data here
        self.data = torch.randn(n_samples, n_features)
        self.labels = torch.randint(0, 2, (n_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    def __init__(self, n_features=10):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(n_features, 2)

    def forward(self, x):
        return self.fc(x)


n_samples = 10_000
n_features = 10

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Initialize dataset, dataloader, model, loss function, and optimizer
dataset = SimpleDataset(n_samples, n_features)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = SimpleModel(n_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}"
            )
            running_loss = 0.0

print("Finished Training")
