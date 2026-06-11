"""Transfer learning: ResNet50 (ImageNet) z zamrożonym backbone i nową głowicą FC."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_resnet50(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """ResNet50 z wagami ImageNet; backbone zamrożony, uczy się tylko nowa warstwa fc."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    # Zamrożony backbone — uczymy tylko nową głowicę, nie psując cech z ImageNet.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Nowa warstwa (requires_grad=True) dopasowana do num_classes. Bez softmaxu.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Sanity check: pobranie wag (raz) + kształt wyjścia (N, 4).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    model = build_resnet50(num_classes=4, freeze_backbone=True).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParametry całkowite : {total_params:,}")
    print(f"Parametry uczone    : {trainable_params:,}")
    print(f"  → uczymy tylko ~{100 * trainable_params / total_params:.2f}% sieci (sama głowica FC).")

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    print(f"\nKształt wejścia : {tuple(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Kształt wyjścia : {tuple(output.shape)}  (oczekiwane: (2, 4))")
    assert output.shape == (2, 4), "BŁĄD: wyjście nie ma kształtu (2, 4)!"
    print("\n✓ Test zaliczony — ResNet50 zwraca poprawny kształt (2, 4).")
