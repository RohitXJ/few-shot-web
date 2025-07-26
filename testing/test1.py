import torch

def test_model_export(model_path):
    print(f"Loading model from: {model_path}")
    config = torch.load(model_path, map_location='cpu')

    for key, value in config.items():
        if torch.is_tensor(value):
            print(f"{key}: Tensor with shape {tuple(value.shape)}")
        else:
            print(f"{key}: {value}")

# Run
test_model_export("export/fewshot_model.pt")
