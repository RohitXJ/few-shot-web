# testing/test1.py

import torch
import torchvision.transforms as T
import pprint

# Trust the transforms used in the model export
torch.serialization.add_safe_globals({
    'torchvision.transforms.transforms.Compose': T.Compose,
    'torchvision.transforms.transforms.Resize': T.Resize,
    'torchvision.transforms.transforms.ToTensor': T.ToTensor
})

def test_model_export(model_path):
    print(f"\nLoading model from: {model_path}")
    
    config = torch.load(model_path, map_location='cpu', weights_only=False)

    print("\n--- Exported Model Content ---")

    for key, value in config.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor shape = {tuple(value.shape)}")
        else:
            print(f"{key}:")
            pprint.pprint(value)

if __name__ == "__main__":
    test_model_export("export/fewshot_model.pt")
