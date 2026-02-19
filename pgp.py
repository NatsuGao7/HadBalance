import torch
from typing import List

def _dot_product(g1: List[torch.Tensor], g2: List[torch.Tensor]) -> torch.Tensor:
    """Helper function: Compute the dot product of two lists of gradients."""
    s = torch.tensor(0.0, device=g1[0].device)
    for a, b in zip(g1, g2):
        s = s + (a * b).sum()
    return s

def _squared_norm(g: List[torch.Tensor]) -> torch.Tensor:
    """Helper function: Compute the squared L2 norm of a list of gradients."""
    s = torch.tensor(0.0, device=g[0].device)
    for a in g:
        s = s + (a * a).sum()
    return s

def slack_agp_project(
    grads_per_task: List[List[torch.Tensor]], 
    primary_idx: int = 0, 
    slack: float = -0.01, 
    eps: float = 1e-12
) -> List[List[torch.Tensor]]:
    """
    Adaptive Gradient Projection with Slack (Slack AGP)
    
    Purpose: Resolve gradient conflicts in Multi-Task Learning. When an auxiliary 
             task's gradient severely hinders the primary task, its conflicting 
             component is projected out. However, a slight deviation (determined 
             by slack) is allowed to preserve geometric/regularization features.

    Args:
        grads_per_task (List[List[torch.Tensor]]): A list containing gradients for all tasks.
                                                   e.g., [g_seg, g_A, g_P, g_X]
        primary_idx (int): Index of the primary task (e.g., segmentation) in the list. Defaults to 0.
        slack (float): Slack threshold. 
                       If = 0.0, it degrades to standard PCGrad (projects once the angle exceeds 90°).
                       If < 0.0 (e.g., -0.01), it allows a minimal amount of negative cosine similarity 
                       (obtuse angle conflict) to protect edge features and prevent over-projection.
        eps (float): A small epsilon value to prevent division by zero.

    Returns:
        List[List[torch.Tensor]]: A new list of gradients after resolving conflicts.
    """
    # 1. Extract the primary task gradient and its squared L2 norm
    g_primary = grads_per_task[primary_idx]
    primary_norm_sq = _squared_norm(g_primary) + eps
    
    # 2. Prepare the output gradient list (shallow copy the outer list to avoid in-place modification issues)
    projected_grads = list(grads_per_task)

    # 3. Iterate over all auxiliary tasks
    for k in range(len(projected_grads)):
        if k == primary_idx:
            continue  # The primary task does not need to project itself

        g_aux = projected_grads[k]
        dot = _dot_product(g_aux, g_primary)
        
        # 🌟 Core logic: Project only when the conflict exceeds our tolerated slack threshold
        if dot < slack * primary_norm_sq:
            # Calculate projection coefficient: (g_aux · g_primary) / ||g_primary||^2
            coef = dot / primary_norm_sq
            
            # Execute projection: subtract the projection component of the auxiliary gradient 
            # along the primary gradient direction
            projected_grads[k] = [
                a - coef * p for a, p in zip(g_aux, g_primary)
            ]

    return projected_grads
