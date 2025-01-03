import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class FFNet(nn.Module):
    """
    Feedforward neural network, virtual base class
    """

    def __init__(self):
        super(FFNet, self).__init__()
        self.dnn_model = self.build_model()

    def build_model(self):
        """build specific model"""
        raise NotImplementedError

    def forward(self, x):
        """feed data forward and return result"""
        raise NotImplementedError

    def train_model(self, trainloader, testloader, loss_fn, optimizer, num_epochs):
        """Train a model."""

        def train_epoch(model, dataloader, loss_fn, optimizer):
            """Train a single epoch"""
            num_data = len(dataloader.dataset)
            # Set the model to training mode
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{num_data:>5d}]")

        def test_epoch(model, dataloader, loss_fn):
            """Test a single epoch"""
            num_data = len(dataloader.dataset)
            num_batches = len(dataloader)
            # Set the model to evaluate mode
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in dataloader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= num_data
            print(
                f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
            )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_epoch(self, trainloader, loss_fn, optimizer)
            test_epoch(self, testloader, loss_fn)
        print("Done!")

    def make_predict(self, images):
        """predict labels for images"""
        self.eval()
        with torch.no_grad():
            outputs = self(images)
            _, predicted = torch.max(outputs.data, 1)  # (max, max_indices)
            return predicted

    def evaluate_model(self, testloader, n):
        """evaluation: return accuracy"""
        correct = 0
        for inputs, labels in testloader:
            pred = self.make_predict(inputs)
            correct += (pred == labels).sum()
        return 100 * correct / n

    def predict_one(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            predicted = outputs[0].argmax(0)
            return predicted


class MLP(FFNet):
    def __init__(self):
        super(MLP, self).__init__()

    def build_model(self):
        """build specific model"""
        # Define a simple MLP for MNIST classification
        # Input: 28x28 = 784 -> Hidden: 512 -> Hidden: 256 -> Output: 10
        model = nn.Sequential(
            nn.Flatten(),  # Flatten 28x28 to 784
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        return model

    def forward(self, x):
        """feed data forward and return result"""
        return self.dnn_model(x)


if __name__ == "__main__":
    # load train and test set
    trainset = torchvision.datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.MNIST(
        "../data", train=False, download=True, transform=transforms.ToTensor()
    )
    print(f"Train #: {len(trainset)}; Test #: {len(testset)}")
    # data iterator
    dataiter = iter(torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False))
    images, labels = next(dataiter)
    print(f"Labels: {labels}; Batch shape: {images.size()}")

    # construct model
    model = MLP()
    print(model)

    # data loader: easier iteration
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # train the model
    loss_fn = nn.CrossEntropyLoss()  # cross entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # SGD
    NUM_EPOCH = 5
    model.train_model(trainloader, testloader, loss_fn, optimizer, NUM_EPOCH)

    # check prediction accuracy
    print(f"Labels    : {labels}")
    print(f"Prediction: {model.make_predict(images)}")
    print(f"Accuracy: {model.evaluate_model(testloader, len(testset)):.2f}")

    # application: predict hand-written digit
    from PIL import Image

    image = Image.open("number6c.png")
    image = transforms.ToTensor()(image).unsqueeze(0)
    print(f"loaded image shape: {image.size()}")
    print(f'Predicted: "{model.predict_one(image)}", Actual: "{6}"')
