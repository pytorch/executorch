# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import logging
import os.path
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from executorch import exir
from executorch.examples.models import model_base
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CifarNet(model_base.EagerModelBase):

    def __init__(self, batch_size: int = 1, pth_file: str | None = None):
        self.batch_size = batch_size
        self.pth_file = pth_file or os.path.join(
            os.path.dirname(__file__), "cifar_net.pth"
        )

    def get_eager_model(self) -> torch.nn.Module:
        return get_model(self.batch_size, state_dict_file=self.pth_file)

    def get_example_inputs(self) -> Tuple[torch.Tensor]:
        tl = get_test_loader()
        ds, _ = tl.dataset[
            0
        ]  # Dataset returns the data and the class. We need just the data.
        return (ds.unsqueeze(0),)

    def get_calibration_inputs(
        self, batch_size: int = 1
    ) -> Iterator[Tuple[torch.Tensor]]:
        tl = get_test_loader(batch_size)

        def _get_first(a, _):
            return (a,)

        return itertools.starmap(_get_first, iter(tl))


class CifarNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.fc = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv1(x)
        x = self.pool1(x)

        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv2(x)
        x = self.pool1(x)

        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv3(x)
        x = self.pool2(x)

        # The output of the previous MaxPool has shape [batch, 64, 4, 4] ([batch, 4, 4, 64] in Neutron IR). When running
        #  inference of the `FullyConnected`, Neutron IR will automatically collapse the channels and spatial dimensions and
        #  work with a tensor of shape [batch, 1024].
        # PyTorch will combine the C and H with `batch`, and leave the last dimension (W). This will result in a tensor of
        #  shape [batch * 256, 4]. This cannot be multiplied with the weight matrix of shape [1024, 10].
        x = torch.reshape(x, (-1, 1024))
        x = self.fc(x)
        x = self.softmax(x)

        return x


def get_train_loader(batch_size: int = 1):
    """Get loader for training data."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader


def get_test_loader(batch_size: int = 1):
    """Get loader for testing data."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return test_loader


def get_model(
    batch_size: int = 1,
    state_dict_file: str | None = None,
    train: bool = False,
    num_epochs: int = 1,
) -> nn.Module:
    """Create the CifarNet model.

    :param batch_size: Batch size to use during training.
    :param state_dict_file: `.pth` file. If provided and the file exists, weights will be loaded from it. Also after
                             training, the weights will be stored in the file.
    :param train: Boolean indicating whether to train the model.
    :param num_epochs: Number of epochs to use during training.
    :return: The loaded/trained CifarNet model.
    """
    cifar_net = CifarNetModel()

    if state_dict_file is not None and os.path.isfile(state_dict_file):
        # Load the pre-trained weights.
        logger.info(f"Using pre-trained weights from `{state_dict_file}`.")
        cifar_net.load_state_dict(torch.load(state_dict_file, weights_only=True))

    if train:
        cifar_net = train_cifarnet_model(
            cifar_net=cifar_net, batch_size=batch_size, num_epochs=num_epochs
        )

        if state_dict_file is not None:
            logger.info(f"Saving the trained weights in `{state_dict_file}`.")
            torch.save(cifar_net.state_dict(), state_dict_file)

    return cifar_net


def get_cifarnet_calibration_data(num_images: int = 100) -> tuple[torch.Tensor]:
    """Return a tuple containing 1 tensor (for the 1 model input) and the tensor will have shape
    [`num_images`, 3, 32, 32].
    """
    loader = iter(get_train_loader(1))  # The train loader shuffles the images.
    images = [image for image, _ in itertools.islice(loader, num_images)]
    tensor = torch.vstack(images)
    return (tensor,)


def train_cifarnet_model(
    cifar_net: nn.Module | torch.fx.GraphModule,
    batch_size: int = 1,
    num_epochs: int = 1,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cifar_net.parameters(), lr=0.0001, momentum=0.6)
    train_loader = get_train_loader(batch_size)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cifar_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    logger.info("Finished training.")
    return cifar_net


def test_cifarnet_model(cifar_net: nn.Module, batch_size: int = 1) -> float:
    """Test the CifarNet model on the CifarNet10 testing dataset and return the accuracy.

    This function may at some point in the future be integrated into the `CifarNet` class.

    :param cifar_net: The model to test with the CifarNet10 testing dataset.
    :return: The accuracy of the model (between 0 and 1).
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in get_test_loader(batch_size):
            images, labels = data
            outputs = cifar_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels).item()

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pte-file",
        required=False,
        help="Name of a `.pte` file to save the trained model in.",
    )
    parser.add_argument(
        "--pth-file",
        required=False,
        type=str,
        help="Name of a `.pth` file to save the trained weights in. If it already exists, the model "
        "will be initialized with these weights.",
    )
    parser.add_argument(
        "--train", required=False, action="store_true", help="Train the model."
    )
    parser.add_argument(
        "--test", required=False, action="store_true", help="Test the trained model."
    )
    parser.add_argument("-b", "--batch-size", required=False, type=int, default=1)
    parser.add_argument("-e", "--num-epochs", required=False, type=int, default=1)
    args = parser.parse_args()

    cifar_net = get_model(
        state_dict_file=args.pth_file,
        train=args.train,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    if args.test:
        logger.info("Running tests.")
        accuracy = test_cifarnet_model(cifar_net, args.batch_size)
        logger.info(f"Accuracy of the network on the 10000 test images: {accuracy}")

    if args.pte_file is not None:
        tracing_inputs = (torch.rand(args.batch_size, 3, 32, 32),)
        aten_dialect_program = torch.export.export(cifar_net, tracing_inputs)
        edge_dialect_program: exir.EdgeProgramManager = exir.to_edge(
            aten_dialect_program
        )
        executorch_program = edge_dialect_program.to_executorch()

        with open(args.pte_file, "wb") as file:
            logger.info(f"Saving the trained model as `{args.pte_file}`.")
            file.write(executorch_program.buffer)
