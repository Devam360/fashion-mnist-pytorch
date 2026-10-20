"""A small convolutional neural network for 28x28 grayscale images."""
import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(1, 32), block(32, 64))  # 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
