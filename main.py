import json
import os
import torch
from torch.utils.data import DataLoader
from utils import io_utils,fs_utils,model_utils,export
from testing.test import run_tests

config = {
    "n_way": 2,
    "k_shot": 4,
    "support_dir": "data/support/",
    "query_dir": "data/query/",
    "backbone": "resnet34",
    "lables": ["PCB","Macbook"]
}

def run_fewshot_pipeline(config):
    support_data, img_format,transforms = io_utils.get_img(img_path=config["support_dir"],ret_transform = True)
    query_data,_ = io_utils.get_img(img_path=config["query_dir"])

    encoder = model_utils.get_encoder(backbone_name= config["backbone"],image_format=img_format)

    support_loader = DataLoader(support_data, batch_size=len(support_data))
    query_loader = DataLoader(query_data, batch_size=len(query_data))

    support_embeddings, support_labels = fs_utils.get_embeddings(support_loader, encoder)
    query_embeddings, query_labels = fs_utils.get_embeddings(query_loader, encoder)
    
    prototypes = fs_utils.compute_prototypes(support_embeddings, support_labels)
    preds_labels, true_labels = fs_utils.compute_distances_and_predict(query_embeddings, query_labels, prototypes)

    correct = (preds_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total * 100

    print("Predicted:", preds_labels.tolist())
    print("Actual   :", true_labels.tolist())
    print(f"Accuracy : {accuracy:.2f}%")
    run_tests(prototypes,encoder)

    class_labels = support_data.classes

    config_out = {
        "labels": class_labels,
        "label_map": {i: label for i, label in enumerate(class_labels)},
        "backbone": config["backbone"],
        "image_format": img_format,
        "transforms": str(transforms),  # directly store the transform object
        "prototypes": prototypes.cpu()
    }
    exp_path = export.export_model(config_out)
    return {
        "export_path": exp_path,
        "accuracy": accuracy,
        "predicted_labels": preds_labels.tolist(),
        "true_labels": true_labels.tolist(),
        "labels": class_labels,
        "n_way": config["n_way"],
        "k_shot": config["k_shot"]
    }

#Test
print(run_fewshot_pipeline(config))