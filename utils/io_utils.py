from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import os
from PIL import Image

def img_check(path_to_check):
    detected = set()
    for cls in os.listdir(path_to_check):
        cls_path = os.path.join(path_to_check,cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path,img_name)
            try:
                with Image.open(img_path) as img:
                    detected.add(img.mode)
            except:
                raise ValueError(f"Corrupted or unreadable image: {img_path}")
    if len(detected)>1:
        raise ValueError(f"Mixed image formats detected: {detected}. Please upload consistent image types (e.g., all RGB or all grayscale).")
    
    return detected.pop()

def get_transform(image_mode):
    if image_mode == 'L':  # Grayscale
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  # Assume RGB or compatible
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
            
def get_img(img_path):
    image_format = img_check(img_path)
    transform = get_transform(image_format)
    data = datasets.ImageFolder(root=img_path,transform=transform)
    return data,image_format
#Test
#est_path = r"data/query"
#print(get_img(test_path))