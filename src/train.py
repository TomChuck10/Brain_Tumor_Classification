"""Trening ResNet50 (transfer learning): train → val → checkpoint → finalny test."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset_setup import prepare_dataloaders
from model_transfer import build_resnet50


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()         # gradienty się kumulują — zerujemy co batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Walidacja lub test — model.eval() + bez gradientów."""
    model.eval()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


def plot_curves(history: dict, save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Trening")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Walidacja")
    ax1.set_title("Krzywa straty (Loss)")
    ax1.set_xlabel("Epoka"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Trening")
    ax2.plot(epochs, history["val_acc"],   "s-", label="Walidacja")
    ax2.set_title("Krzywa dokładności (Accuracy)")
    ax2.set_xlabel("Epoka"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Krzywe uczenia — ResNet50 (Transfer Learning)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  Wykresy zapisane: {save_path}")


def run_training(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
    num_workers: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print("  [UWAGA] Trening na CPU będzie BARDZO wolny!")

    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(
        image_size=image_size, batch_size=batch_size, num_workers=num_workers,
    )
    num_classes = len(class_names)
    print(f"Klasy ({num_classes}): {class_names}")
    print(f"Próbki: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model = build_resnet50(num_classes=num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Tylko parametry uczone — backbone jest zamrożony (requires_grad=False).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Gdy val_loss nie spada przez 2 epoki, tnie lr 10× — wygładza skoki.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_resnet50.pth"
    best_val_acc = 0.0

    print("\n" + "=" * 70)
    print(f"START TRENINGU — {num_epochs} epok, lr={learning_rate}, batch={batch_size}")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoka {epoch:2d}/{num_epochs} | "
            f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val: loss={val_loss:.4f} acc={val_acc:.4f} | lr={current_lr:.1e}",
            end="",
        )

        # Checkpoint po val_acc; zapisujemy state_dict (przenośne).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("  ← najlepszy dotąd, zapisano ✓")
        else:
            print()

    print("=" * 70)
    print(f"Trening zakończony. Najlepsze val_accuracy = {best_val_acc:.4f}")
    print(f"Najlepszy model zapisany w: {best_model_path}")

    plot_curves(history, REPORTS_DIR / "training_curves_resnet50.png")

    # Finalna, uczciwa ocena: najlepsze wagi na zbiorze testowym (wcześniej nieużywanym).
    print("\n" + "=" * 70)
    print("FINALNA OCENA NA ZBIORZE TESTOWYM (Kaggle 'Testing')")
    print("=" * 70)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_training(
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        image_size=224,
        num_workers=0,   # 0 = bezpieczne dla Windows
    )
