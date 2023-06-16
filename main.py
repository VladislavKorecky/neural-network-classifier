import torch as t
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from matplotlib.pyplot import plot, show

from network import NeuralNetwork


# ------------------
#       CONFIG
# ------------------
EPOCHS = 7
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# -----------------
#       SETUP
# -----------------
dataset = MNIST("./", train=True, transform=ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

net = NeuralNetwork()
optimizer = Adam(net.parameters(), lr=LEARNING_RATE)

accuracy_history = []


def train() -> None:
    """
    Do one training pass (one epoch) on the dataset.
    """

    for data_batch, label_batch in dataloader:
        data_batch = data_batch.reshape((-1, 784))

        prediction = net.forward(data_batch)

        # turn the network output (a vector) into a single number representing the chosen class
        classes = t.argmax(prediction, dim=1)

        # get the difference between the labels and the chosen classes
        diff = label_batch != classes

        # calculate the number of differences
        diff_count = diff.sum().item()

        # use the diff count to calculate accuracy
        accuracy = 100 - (diff_count / BATCH_SIZE * 100)
        accuracy_history.append(accuracy)

        print(accuracy)

        # calculate the loss and update the parameters
        loss = cross_entropy(prediction, label_batch)
        loss.backward()
        optimizer.step()


# ------------------
#      TRAINING
# ------------------
for _ in range(EPOCHS):
    train()

# save the AI's parameters
t.save(net.state_dict(), "model.txt")

# show the accuracy on a graph
plot(accuracy_history)
show()
