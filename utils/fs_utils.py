import torch
from collections import defaultdict

def get_embeddings(dataloader, encoder):
    """Returns stacked embeddings and labels from a DataLoader"""
    encoder.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))  # single batch
        embeddings = encoder(images).squeeze()   # [N, D]
    return embeddings, labels

def compute_prototypes(embeddings, labels):
    """
    Computes class-wise prototype vectors (mean embeddings).
    
    Args:
        embeddings: Tensor of shape [N, D]
        labels: Tensor of shape [N]

    Returns:
        prototypes: Tensor of shape [n_way, D]
    """
    prototypes = []
    unique_classes = torch.unique(labels)

    for cls in unique_classes:
        class_embeddings = embeddings[labels == cls]   # [k_shot, D]
        class_mean = class_embeddings.mean(dim=0)       # [D]
        prototypes.append(class_mean)

    return torch.stack(prototypes)

def compute_distances_and_predict(query_embeddings, query_labels, prototypes):
    """
    Predicts labels for queries based on distances to prototypes.
    
    Args:
        query_embeddings: Tensor of shape [Q, D]
        query_labels: Tensor of shape [Q]
        prototypes: Tensor of shape [n_way, D]

    Returns:
        preds: Tensor of shape [Q] — predicted class indices (0 to n_way-1)
        labels: Tensor of shape [Q] — actual class indices (0 to n_way-1)
    """
    # Compute pairwise distances [Q, n_way]
    dists = torch.cdist(query_embeddings, prototypes)
    print(dists.shape)

    # Predicted class is index of nearest prototype
    preds = torch.argmin(dists,dim=1)  # [Q]
    print(preds.shape,preds)
    return preds, query_labels
