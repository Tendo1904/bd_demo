import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os


# Простая CNN
class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network (CNN) for image classification.

    This class defines a convolutional neural network structure that consists of multiple layers,
    including convolutional layers, activation functions, pooling layers, flattening, and fully
    connected layers. It is designed to process image data and perform classification tasks.

    Methods:
        __init__: Initializes the SimpleCNN model.
        forward: Computes the forward pass through the neural network.

    Attributes:
        None

    The __init__ method constructs the various layers of the CNN, setting up the architecture
    necessary for processing image inputs. The forward method takes input data and passes it
    through the network to produce the final output.
    """

    def __init__(self):
        """
        Initialize a SimpleCNN model.

            This method creates a convolutional neural network (CNN) comprised of multiple layers,
            including convolutional layers, activation functions, pooling layers, flattening, and
            fully connected layers. The network is designed for image classification.

            Parameters:
                None

            Returns:
                None
        """
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """

        Computes the forward pass through the neural network.

        This method takes an input and passes it through the neural network to obtain the output.

        Args:
            x: The input data to be processed by the network.

        Returns:
            The output from the neural network after the input has been processed.
        """
        return self.net(x)


if __name__ == "__main__":
    # Параметры
    epochs = 5
    batch_size = 64
    lr = 0.001
    save_path = "mnist_model.pth"

    # Трансформации и датасеты
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()
    # Dummy input для визуализации графа
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, dummy_input)

    # Обучение
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)

    # Сохраняем модель
    torch.save(model.state_dict(), save_path)
    torch.save(model, save_path[:-1])
    print("Model saved!")

    writer.close()
