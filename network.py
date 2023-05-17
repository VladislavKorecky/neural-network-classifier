from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.nn.functional import softmax


class NeuralNetwork(Module):
    """
    Neural network classifier.
    """

    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Linear(784, 300),
            LeakyReLU(),

            Linear(300, 300),
            LeakyReLU(),

            Linear(300, 300),
            LeakyReLU(),

            Linear(300, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Make a feed forward pass through the network.

        Args:
            x (Tensor): Neural network's input.

        Returns:
            Tensor: Neural network's output.
        """

        return self.layers.forward(x)

    def predict(self, x: Tensor) -> Tensor:
        """
        Make a feed forward pass and apply softmax to the output.

        Args:
            x (Tensor): Neural network's input.

        Returns:
            Tensor: Neural network's output.
        """

        return softmax(self.forward(x), dim=1)
