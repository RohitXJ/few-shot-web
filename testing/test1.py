import torch

def test_model_export(model_path):
    print(f"Loading model from: {model_path}")
    config = torch.load(model_path, map_location='cpu')

    print("\n✅ Model Configuration Summary:\n" + "-"*40)

    for key, value in config.items():
        print(f"\n🔹 {key}: ", end='')

        if isinstance(value, torch.Tensor):
            print(f"Tensor with shape {tuple(value.shape)}")
        elif isinstance(value, dict):
            print("Dictionary")
            for k, v in value.items():
                print(f"    • {k} → {v}")
        elif isinstance(value, list):
            print(f"List of {len(value)} items: {value}")
        else:
            print(f"{value} ({type(value).__name__})")

    print("\n✅ All items loaded and printed successfully.")

if __name__ == "__main__":
    test_model_export("export/fewshot_model.pt")
