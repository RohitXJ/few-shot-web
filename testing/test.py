import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Custom dataset to load images from a folder
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        self.image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # sort numerically
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name  # return name for reference

# Main testing function
def run_tests(prototypes, encoder):
    test_dir = r"testing/dataset"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load images
    test_dataset = ImageFolderDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()

    with torch.no_grad():
        for img, name in test_loader:
            img = img.to(device)
            embedding = encoder(img)  # shape could be [1, D] or [1, D, 1, 1]
            
            if len(embedding.shape) > 2:
                embedding = torch.flatten(embedding, start_dim=1)  # shape: [1, D]

            dists = torch.cdist(embedding, prototypes.to(device))  # [1, n_way]
            pred = torch.argmin(dists, dim=1).item()  # scalar index
            all_preds.append((name[0], pred))


    # Print results
    print("Predictions:")
    for fname, pred_class in all_preds:
        print(f"{fname} → Predicted class index: {pred_class}")
