import torch

print("--- TEST ŚRODOWISKA ---")
print(f"Wersja PyTorch: {torch.__version__}")
print(f"Czy CUDA jest dostępna? {'TAK' if torch.cuda.is_available() else 'NIE'}")

if torch.cuda.is_available():
    print(f"Wykryta karta: {torch.cuda.get_device_name(0)}")
    print(f"Liczba dostępnych GPU: {torch.cuda.device_count()}")
else:
    print("BŁĄD: PyTorch nie widzi Twojej karty RTX. Sprawdź wersję sterowników NVIDIA.")