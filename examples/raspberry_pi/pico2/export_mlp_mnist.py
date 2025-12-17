#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir import EdgeCompileConfig, to_edge

from torch.export import export


# Constants
INPUT_SIZE = 784  # 28*28 for MNIST
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 10
IMAGE_SIZE = 28


class TinyMLPMNIST(nn.Module):
    """A small MLP for MNIST digit classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.fc3 = nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        """Forward pass through the network."""
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_balanced_model():
    """
    Create a balanced MLP model for MNIST digit classification.

    The model is pre-initialized with specific weights to recognize
    digits 0, 1, 4, and 7 through hand-crafted feature detectors.

    Returns:
        torch.nn.Module: A TinyMLPMNIST model with balanced weights
    """
    model = TinyMLPMNIST()

    with torch.no_grad():
        # Zero everything first
        for param in model.parameters():
            param.fill_(0.0)

        # Feature 0: Vertical lines (for 1, 4, 7)
        for row in range(IMAGE_SIZE):
            # Middle column
            model.fc1.weight[0, row * IMAGE_SIZE + 14] = 2.0

        # Feature 1: Top horizontal (for 7, 4)
        model.fc1.weight[1, 0:84] = 2.0  # Top 3 rows

        # Feature 2: Bottom horizontal (for 1, 4)
        model.fc1.weight[2, 25 * IMAGE_SIZE :] = 2.0  # Bottom 3 rows

        # Feature 3: STRONGER Oval detector for 0
        # Top and bottom curves
        model.fc1.weight[3, 1 * IMAGE_SIZE + 8 : 1 * IMAGE_SIZE + 20] = 2.0
        model.fc1.weight[3, 26 * IMAGE_SIZE + 8 : 26 * IMAGE_SIZE + 20] = 2.0
        # Left and right sides
        for row in range(4, 24):
            model.fc1.weight[3, row * IMAGE_SIZE + 7] = 2.0  # Left
            model.fc1.weight[3, row * IMAGE_SIZE + 20] = 2.0  # Right
        # Anti-middle (hollow center)
        for row in range(10, 18):
            model.fc1.weight[3, row * IMAGE_SIZE + 14] = -1.5

        # Feature 4: Middle horizontal (for 4) - make it STRONGER
        model.fc1.weight[4, 13 * IMAGE_SIZE : 15 * IMAGE_SIZE] = 3.0

        # Second layer: More decisive detection
        # Digit 0 detector: STRONG oval requirement
        model.fc2.weight[0, 3] = 5.0  # Strong oval requirement
        model.fc2.weight[0, 0] = -2.0  # Anti-vertical
        model.fc2.weight[0, 4] = -3.0  # Anti-middle horizontal

        # Digit 1 detector: vertical + bottom - others
        model.fc2.weight[1, 0] = 3.0  # Vertical
        model.fc2.weight[1, 2] = 2.0  # Bottom
        model.fc2.weight[1, 1] = -1.0  # Anti-top
        model.fc2.weight[1, 3] = -2.0  # Anti-oval

        # Digit 4 detector: REQUIRES middle horizontal
        model.fc2.weight[2, 0] = 2.0  # Vertical
        model.fc2.weight[2, 1] = 1.0  # Top
        model.fc2.weight[2, 4] = 4.0  # STRONG middle requirement
        model.fc2.weight[2, 3] = -2.0  # Anti-oval

        # Digit 7 detector: top + some vertical - bottom
        model.fc2.weight[3, 1] = 3.0  # Top
        model.fc2.weight[3, 0] = 1.0  # Some vertical
        model.fc2.weight[3, 2] = -2.0  # Anti-bottom

        # Output layer
        model.fc3.weight[0, 0] = 5.0  # Digit 0
        model.fc3.weight[1, 1] = 5.0  # Digit 1
        model.fc3.weight[4, 2] = 5.0  # Digit 4
        model.fc3.weight[7, 3] = 5.0  # Digit 7

        # Bias against other digits
        for digit in [2, 3, 5, 6, 8, 9]:
            model.fc3.bias[digit] = -3.0

    return model


def test_comprehensive(model):
    """
    Test model with clear digit patterns.

    Args:
        model: The PyTorch model to test
    """

    # Create clearer test patterns
    def create_digit_1():
        """Create a test pattern for digit 1."""
        digit = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        # Thick vertical line in middle
        digit[0, 2:26, 13:16] = 1.0  # Thick vertical line
        # Top part (like handwritten 1)
        digit[0, 2:5, 11:14] = 1.0
        # Bottom base
        digit[0, 24:27, 10:19] = 1.0
        return digit

    def create_digit_7():
        """Create a test pattern for digit 7."""
        digit = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        # Top horizontal line
        digit[0, 1:4, 3:26] = 1.0
        # Diagonal line
        for i in range(23):
            row = 4 + i
            col = 23 - i
            if 0 <= row < IMAGE_SIZE and 0 <= col < IMAGE_SIZE:
                digit[0, row, col - 1 : col + 2] = 1.0  # Thick diagonal
        return digit

    def create_digit_0():
        """Create a test pattern for digit 0."""
        digit = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        # Oval shape
        for row in range(3, 25):
            for col in range(8, 20):
                condition1 = ((row - 14) ** 2 / 11**2 + (col - 14) ** 2 / 6**2) <= 1
                condition2 = ((row - 14) ** 2 / 8**2 + (col - 14) ** 2 / 3**2) > 1
                if condition1 and condition2:
                    digit[0, row, col] = 1.0
        return digit

    patterns = {
        "Digit 1": create_digit_1(),
        "Digit 7": create_digit_7(),
        "Digit 0": create_digit_0(),
    }

    print("🧪 Testing with clear patterns:")
    model.eval()
    with torch.no_grad():
        for name, pattern in patterns.items():
            output = model(pattern)
            pred = output.argmax().item()
            confidence = F.softmax(output, dim=1)[0, pred].item()
            print(f"   {name} → predicted: {pred} (confidence: {confidence:.3f})")

            # Show top 3 predictions
            top3 = output.topk(3, dim=1)
            predictions = [
                (top3.indices[0, i].item(), top3.values[0, i].item()) for i in range(3)
            ]
            print(f"      Top 3: {predictions}")


def main():
    """Main function to create, test, and export the model."""
    print("🔥 Creating balanced MLP MNIST model...")

    model = create_balanced_model()

    # Test the model
    test_comprehensive(model)

    # Export
    example_input = torch.randn(1, IMAGE_SIZE, IMAGE_SIZE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model parameters: {param_count:,}")

    print("📦 Exporting...")
    with torch.no_grad():
        exported_program = export(model, (example_input,))

    print("⚙️ Converting to ExecuTorch...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_manager = to_edge(exported_program, compile_config=edge_config)
    et_program = edge_manager.to_executorch()

    # Save with error handling
    filename = "balanced_tiny_mlp_mnist.pte"
    print(f"💾 Saving {filename}...")
    try:
        with open(filename, "wb") as f:
            f.write(et_program.buffer)

        model_size_kb = len(et_program.buffer) / 1024
        print("✅ Export complete!")
        print(f"📁 Model size: {model_size_kb:.1f} KB")
    except IOError as e:
        print(f"❌ Failed to save model: {e}")


if __name__ == "__main__":
    main()
