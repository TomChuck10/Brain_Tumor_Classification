#Model bazowy (Baseline CNN) — punkt odniesienia dla transfer learningu

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    #Prosta CNN: 4 bloki Conv-BN-ReLU-Pool + klasyfikator. Zwraca logity (bez softmaxu)

    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stały rozmiar mapy cech niezależnie od rozmiaru wejścia.
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        flattened_size = 256 * 14 * 14

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),  # logity — softmax robi CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Sanity check: czy model przepuszcza batch i zwraca kształt (N, 4).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    model = BaselineCNN(num_classes=4).to(device)
    print("\n── Struktura modelu BaselineCNN ──")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLiczba parametrów: {total_params:,} (uczonych: {trainable:,})")

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    print(f"\nKształt wejścia : {tuple(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Kształt wyjścia : {tuple(output.shape)}  (oczekiwane: (2, 4))")
    assert output.shape == (2, 4), "BŁĄD: wyjście nie ma kształtu (2, 4)!"
    print("\n✓ Test zaliczony — model zwraca poprawny kształt (2, 4).")
