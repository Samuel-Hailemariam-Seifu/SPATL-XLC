
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_random_pruning(model, prune_ratio=0.2, logger=None):
    """
    Apply random unstructured pruning to all Conv2d and Linear layers in the model.
    
    Args:
        model (torch.nn.Module): Local model to prune.
        prune_ratio (float): Fraction of weights to prune randomly (e.g., 0.2 = 20%).
        logger (logging.Logger): Optional logger to log pruning info.
    
    Returns:
        pruned_model (torch.nn.Module): Pruned version of the input model.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name="weight", amount=prune_ratio)
            if logger:
                logger.info(f"[Random Pruning] Applied to {name}: {prune_ratio * 100:.1f}%")

    return model

def get_model_sparsity(model):
    total, zero = 0, 0
    for param in model.parameters():
        total += param.numel()
        zero += torch.sum(param == 0).item()
    return zero / total

sparsity = get_model_sparsity(net)
logger.info(f"[Random Pruning] Model sparsity: {sparsity:.2%}")


import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_magnitude_pruning(model, prune_ratio=0.2, logger=None):
    """
    Apply unstructured magnitude-based pruning to all Conv2d and Linear layers.
    
    Args:
        model (torch.nn.Module): Local model to prune.
        prune_ratio (float): Fraction of weights to prune (e.g., 0.2 = 20%).
        logger (logging.Logger): Optional logger.
    
    Returns:
        pruned_model (torch.nn.Module): Pruned model with low-magnitude weights removed.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            if logger:
                logger.info(f"[Magnitude Pruning] Applied to {name}: {prune_ratio * 100:.1f}%")

    return model

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def apply_grad_x_input_pruning(model, dataloader, device, prune_ratio=0.2, logger=None):
    """
    Apply Gradient × Input-based pruning to Conv2d and Linear layers.
    
    Args:
        model (torch.nn.Module): Model to prune.
        dataloader (DataLoader): One mini-batch (or a few) to estimate saliency.
        device (torch.device): 'cuda' or 'cpu'
        prune_ratio (float): Fraction of weights to prune.
        logger (Logger): Optional logging.
    
    Returns:
        model (torch.nn.Module): Pruned model.
    """
    model.eval()
    model.to(device)

    # Accumulate saliency scores for each layer
    saliency_scores = {}

    # Use only 1 batch for saliency estimation
    inputs, labels = next(iter(dataloader))
    inputs, labels = inputs.to(device), labels.to(device)

    # Enable gradient tracking
    inputs.requires_grad = True

    # Forward pass
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    # Collect saliency = |grad × input| for each weight
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.grad is None:
                continue
            score = (module.weight.grad * module.weight.data).abs()
            saliency_scores[module] = score

    # Prune based on saliency threshold (layer-wise)
    for module, score in saliency_scores.items():
        threshold = torch.quantile(score.view(-1), prune_ratio)
        mask = (score > threshold).float()
        prune.custom_from_mask(module, name="weight", mask=mask)
        if logger:
            logger.info(f"[Grad x Input Pruning] Pruned {prune_ratio*100:.1f}% of weights in {module.__class__.__name__}")

    return model

import torch
import shap
import torch.nn as nn
from torch.nn.utils import prune

def apply_classwise_shap_pruning(model, dataloader, device, prune_ratio=0.2, num_classes=10, logger=None):
    """
    Apply class-wise SHAP-based pruning using GradientExplainer.

    Args:
        model (torch.nn.Module): The local model.
        dataloader (DataLoader): Local data loader (should contain class diversity if possible).
        device (torch.device): Computation device.
        prune_ratio (float): Total proportion of weights to prune per layer.
        num_classes (int): Number of total classes in the task (e.g., 10 for CIFAR-10).
        logger (Logger): Optional logger.

    Returns:
        model (torch.nn.Module): Pruned model.
    """
    import shap
    from shap.explainers import GradientExplainer

    model.eval()
    model.to(device)

    # Collect a batch of data per class
    class_inputs = {}
    for x, y in dataloader:
        for i in range(len(y)):
            cls = int(y[i].item())
            if cls not in class_inputs:
                class_inputs[cls] = []
            if len(class_inputs[cls]) < 5:  # max 5 samples per class
                class_inputs[cls].append(x[i].unsqueeze(0))
        if all(len(v) >= 5 for v in class_inputs.values()):
            break

    # Combine classwise examples
    combined_inputs = torch.cat([torch.cat(class_inputs[c], dim=0) for c in sorted(class_inputs.keys())], dim=0).to(device)

    # Setup SHAP explainer
    background = combined_inputs[:10]
    explainer = shap.GradientExplainer((model, model.encoder), background)

    # Run SHAP
    shap_values = explainer.shap_values(combined_inputs)
    # shap_values shape: [num_classes][N, C, H, W] or [num_classes][N, D] for MLP

    # Aggregate per-layer saliency
    saliency_scores = {}
    model.zero_grad()

    for cls_idx, cls_shap in enumerate(shap_values):
        flat = torch.tensor(cls_shap).abs().mean(dim=0)  # Average across samples
        if flat.dim() > 1:
            flat = flat.mean(dim=tuple(range(1, flat.dim())))
        saliency_scores[cls_idx] = flat

    # Combine all classes’ importance (could weight this by local class distribution)
    overall_score = sum(saliency_scores.values()) / len(saliency_scores)

    # Prune each Conv2d or Linear layer using this saliency
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            flat_score = overall_score.flatten()
            threshold = torch.quantile(flat_score, prune_ratio)
            score = weight.abs()  # fallback if SHAP score fails for layer shape
            mask = (score > threshold).float()
            prune.custom_from_mask(module, name='weight', mask=mask)
            if logger:
                logger.info(f"[Classwise SHAP] Pruned {prune_ratio*100:.1f}% of weights in {name}")

    return model

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn.functional as F

def compute_entropy(tensor, eps=1e-8):
    """
    Compute entropy of a tensor along the batch dimension.
    """
    probs = F.softmax(tensor, dim=1)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean().item()

def apply_entropy_pruning(model, dataloader, device, prune_ratio=0.2, logger=None):
    """
    Prune neurons/filters with lowest output entropy (least informative).
    
    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Local data.
        device (torch.device): Device to use.
        prune_ratio (float): Proportion to prune.
        logger (Logger): Optional logging.
    
    Returns:
        model: Pruned model.
    """
    model.eval()
    model.to(device)

    entropies = {}

    # Capture activations
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                entropies[name] = compute_entropy(output.detach())
        return hook

    # Register hooks on conv and linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # Run one batch
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    with torch.no_grad():
        _ = model(inputs)

    for hook in hooks:
        hook.remove()

    # Sort layers by entropy
    sorted_layers = sorted(entropies.items(), key=lambda x: x[1])

    # Prune low-entropy layers (or parts of them)
    for name, entropy in sorted_layers:
        module = dict(model.named_modules())[name]
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Prune based on weight magnitude within low-entropy layers
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            if logger:
                logger.info(f"[Entropy Pruning] {name}: entropy={entropy:.4f}, pruned {prune_ratio*100:.1f}%")

    return model

