from os.path import isfile
from random import choice

import torch as t
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from matplotlib.pyplot import imshow, show

from network import NeuralNetwork


# -----------------
#       SETUP
# -----------------
# check that the AI was trained
if not isfile("model.txt"):
    print("ERROR: File \"model.txt\" not found. You have to train the AI before testing.")
    exit(0)

dataset = MNIST("./", train=False, transform=ToTensor(), download=True)

# load the trained AI
net = NeuralNetwork()
net.load_state_dict(t.load("model.txt"))

# turn on the testing mode
net.eval()


# -----------------
#      TESTING
# -----------------
while True:
    # pick a random sample from the dataset
    random_sample = choice(dataset)

    # separate the image from the label
    image, label = random_sample

    # get rid of the extra dimension (1 x 28 x 28 -> 28 x 28)
    display_image = image.reshape(28, 28)

    # reshape the image so that it can be fed to the network
    image_input = image.reshape(784)

    # get the AI's prediction
    prediction = net.predict(image_input)
    predicted_class = t.argmax(prediction)

    # display the guess and the image itself
    print(f"Label: {label}, Guess: {predicted_class}")
    imshow(display_image, cmap="gray")
    show()
