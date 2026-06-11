#Wczytywanie danych MRI i budowa DataLoaderów (train / val / test)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN   = PROJECT_ROOT / "data" / "raw" / "Training"
DATA_TEST    = PROJECT_ROOT / "data" / "raw" / "Testing"

# ResNet50 wymaga wejścia znormalizowanego dokładnie statystykami ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


def prepare_dataloaders(
    train_dir: Path = DATA_TRAIN,
    test_dir:  Path = DATA_TEST,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    # Train/val wydzielone z folderu Training, test = osobny folder Testing (nietknięty)
    train_tf, eval_tf = build_transforms(image_size)

    # Osobny ImageFolder bez augmentacji, żeby walidacja nie była augmentowana.
    train_full_aug  = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    train_full_eval = datasets.ImageFolder(root=str(train_dir), transform=eval_tf)

    n_total = len(train_full_aug)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    # Stały seed → powtarzalny podział.
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_subset = random_split(
        train_full_aug, [n_train, n_val], generator=generator
    )

    train_subset = train_split
    val_subset = Subset(train_full_eval, val_split_subset.indices)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_tf)

    use_gpu = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_gpu)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_gpu)

    return train_loader, val_loader, test_loader, train_full_aug.classes


def print_summary(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    class_names:  list[str],
    batch_size: int,
    image_size: int,
) -> None:
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)

    print("\n" + "=" * 55)
    print("       PODSUMOWANIE ZBIORU DANYCH (3 zbiory)")
    print("=" * 55)
    print(f"  Klasy ({len(class_names)} szt.): {class_names}")
    print("-" * 55)
    print(f"  {'Zbiór':<20} {'Liczba próbek':>15} {'Batchy':>10}")
    print("-" * 55)
    print(f"  {'Trening (85%)':<20} {n_train:>15} {len(train_loader):>10}")
    print(f"  {'Walidacja (15%)':<20} {n_val:>15} {len(val_loader):>10}")
    print(f"  {'Test (Kaggle)':<20} {n_test:>15} {len(test_loader):>10}")
    print("-" * 55)
    print(f"  {'ŁĄCZNIE':<20} {n_train + n_val + n_test:>15}")
    print("=" * 55)
    print(f"  Rozmiar obrazu   : {image_size} × {image_size} px")
    print(f"  Rozmiar batcha   : {batch_size}")
    print(f"  Urządzenie GPU   : {'TAK ✓  ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NIE (CPU)'}")

    sample_imgs, _ = next(iter(train_loader))
    print(f"  Shape batcha     : {tuple(sample_imgs.shape)}  → (batch, kanały, H, W)")
    print("=" * 55 + "\n")


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    # Odwraca Normalize() — bez tego obraz na wykresie ma przekłamane kolory
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C) dla matplotlib


def show_sample_grid(loader: DataLoader, class_names: list[str], n_images: int = 6) -> None:
    images, labels = next(iter(loader))

    n_images = min(n_images, len(images))
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle("Przykładowe obrazy MRI z batcha treningowego",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    for i in range(n_images):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.imshow(_denormalize(images[i]))
        ax.set_title(f"Klasa: {class_names[labels[i].item()]}", fontsize=12, pad=6)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "sample_grid.png", dpi=120, bbox_inches="tight")
    print("  Siatka zapisana jako sample_grid.png")
    plt.show()


def main() -> None:
    IMAGE_SIZE  = 224
    BATCH_SIZE  = 32
    NUM_WORKERS = 0   # 0 = bezpieczne dla windows (później można podbić do 4!!!)

    for path, name in [(DATA_TRAIN, "Training"), (DATA_TEST, "Testing")]:
        if not path.exists():
            print(f"[BŁĄD] Folder '{name}' nie istnieje pod ścieżką:\n  {path}")
            sys.exit(1)

    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(
        DATA_TRAIN, DATA_TEST,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )
    print_summary(train_loader, val_loader, test_loader, class_names, BATCH_SIZE, IMAGE_SIZE)
    show_sample_grid(train_loader, class_names, n_images=6)


if __name__ == "__main__":
    main()
