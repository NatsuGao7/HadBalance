# =========================
# optimization.py
# =========================
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch
from scipy.optimize import least_squares


class AGBSolver:
    """
    AGB (Adaptive Gradient Balancing) upper-level solver.

    Solves for alpha >= 0 such that:
        (G^T G) alpha ~= 1 / sqrt(alpha)
    Joint direction:
        d = sum_i alpha_i g_i

    Options:
      - use_unit_gram: Build the Gram matrix on unit-normalized gradients (direction-only). 
                       STRICT default: False (uses raw Gram matrix).
      - anchor_primary: Hard-fix alpha_primary = anchor_value (NOT strict; default False).
      - normalize_weights: Force the weights to sum to 1 (NOT strict; default False).
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        use_unit_gram: bool = False,
        ridge: float = 1e-12,
        barrier_eps: float = 1e-8,
    ):
        self.n_tasks = int(n_tasks)
        self.device = device
        self.use_unit_gram = bool(use_unit_gram)
        self.ridge = float(ridge)
        self.barrier_eps = float(barrier_eps)

    @staticmethod
    def _dot_list(g1: List[torch.Tensor], g2: List[torch.Tensor]) -> torch.Tensor:
        """Helper function: Compute the dot product of two lists of gradients."""
        s = torch.tensor(0.0, device=g1[0].device)
        for a, b in zip(g1, g2):
            s = s + (a * b).sum()
        return s

    @staticmethod
    def _norm_sq_list(g: List[torch.Tensor]) -> torch.Tensor:
        """Helper function: Compute the squared L2 norm of a list of gradients."""
        s = torch.tensor(0.0, device=g[0].device)
        for a in g:
            s = s + (a * a).sum()
        return s

    @staticmethod
    def _scale_list(g: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
        """Helper function: Scale a list of gradients."""
        if scale <= 0.0:
            return [torch.zeros_like(x) for x in g]
        return [x / scale for x in g]

    def solve_weights_from_grads(
        self,
        grads_per_task: List[List[torch.Tensor]],
        primary_idx: int = 0,
        eps: float = 1e-12,
        normalize_weights: bool = False,
        anchor_primary: bool = False,
        anchor_value: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Solves for the optimal balancing weights (alpha) based on the task gradients.
        """
        assert len(grads_per_task) == self.n_tasks
        device = grads_per_task[0][0].device
        tiny = float(torch.finfo(torch.float32).tiny)

        # 1) Compute raw norms for each task gradient
        norms = np.zeros(self.n_tasks, dtype=np.float64)
        for t in range(self.n_tasks):
            n2 = self._norm_sq_list(grads_per_task[t])
            n = float(torch.sqrt(torch.clamp(n2, min=0.0)).item())
            norms[t] = 0.0 if n <= tiny else n
        norms_list = norms.tolist()

        # Identify active tasks (tasks with non-zero gradients)
        active = norms > 0.0
        active_idx = np.where(active)[0].tolist()

        # If no active tasks exist, return uniform weights and a zero Gram matrix
        if len(active_idx) == 0:
            alpha = np.ones(self.n_tasks, dtype=np.float64) / float(self.n_tasks)
            GTG = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float64)
            return alpha, GTG, norms_list

        # 2) Select gradients for building the Gram matrix
        g_used: List[List[torch.Tensor]] = []
        for t in range(self.n_tasks):
            if norms[t] <= 0.0:
                g_used.append([torch.zeros_like(x) for x in grads_per_task[t]])
            else:
                if self.use_unit_gram:
                    g_used.append(self._scale_list(grads_per_task[t], float(norms[t])))
                else:
                    g_used.append([x for x in grads_per_task[t]])

        # 3) Build the Gram matrix GTG = <g_i, g_j>
        GTG_t = torch.zeros((self.n_tasks, self.n_tasks), device=device, dtype=torch.float32)
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if norms[i] <= 0.0 or norms[j] <= 0.0:
                    GTG_t[i, j] = 0.0
                else:
                    GTG_t[i, j] = self._dot_list(g_used[i], g_used[j])

        # Apply ridge regularization to stabilize the Gram matrix
        if self.ridge > 0:
            GTG_t = GTG_t + (self.ridge * torch.eye(self.n_tasks, device=device, dtype=GTG_t.dtype))

        GTG = GTG_t.detach().cpu().numpy().astype(np.float64)

        # 4) Solve for alpha exclusively on the ACTIVE subset of tasks
        A_act = GTG[np.ix_(active_idx, active_idx)]
        m = len(active_idx)
        beps = float(self.barrier_eps)

        def _fallback_alpha_act() -> np.ndarray:
            """Fallback strategy: distribute weights proportionally to gradient norms."""
            nn = norms[active_idx]
            s = float(nn.sum() + 1e-12)
            if s <= 0:
                return np.ones(m, dtype=np.float64) / float(m)
            return (nn / s).astype(np.float64)

        # Handle the case where only one task is active
        if m == 1:
            only_idx = active_idx[0]
            if anchor_primary and only_idx == primary_idx:
                alpha_act = np.array([float(anchor_value)], dtype=np.float64)
            else:
                alpha_act = np.array([1.0], dtype=np.float64)
        else:
            primary_in_active = (primary_idx in active_idx)

            # Solve with an anchored primary weight
            if anchor_primary and primary_in_active:
                ppos = active_idx.index(primary_idx)
                unknown_pos = [j for j in range(m) if j != ppos]
                dim = len(unknown_pos)
                x0 = np.ones(dim, dtype=np.float64)

                def objfn(beta: np.ndarray) -> np.ndarray:
                    beta = np.maximum(beta, 0.0)
                    alpha_full = np.zeros(m, dtype=np.float64)
                    alpha_full[ppos] = float(anchor_value)
                    alpha_full[unknown_pos] = beta
                    return (A_act @ alpha_full) - (1.0 / np.sqrt(alpha_full + beps))

                try:
                    res = least_squares(objfn, x0, bounds=(0.0, np.inf))
                    beta_hat = np.maximum(res.x.astype(np.float64), 0.0)

                    alpha_act = np.zeros(m, dtype=np.float64)
                    alpha_act[ppos] = float(anchor_value)
                    alpha_act[unknown_pos] = beta_hat
                except Exception:
                    alpha_act = _fallback_alpha_act()
                    alpha_act[ppos] = float(anchor_value)

            # Solve without anchoring (standard optimization)
            else:
                x0 = np.ones(m, dtype=np.float64) / float(m)

                def objfn(x: np.ndarray) -> np.ndarray:
                    x = np.maximum(x, 0.0)
                    return (A_act @ x) - (1.0 / np.sqrt(x + beps))

                try:
                    res = least_squares(objfn, x0, bounds=(0.0, np.inf))
                    alpha_act = np.maximum(res.x.astype(np.float64), 0.0)
                except Exception:
                    alpha_act = _fallback_alpha_act()

        # 5) Map the solved active weights back to the full alpha array
        alpha = np.zeros(self.n_tasks, dtype=np.float64)
        for k, idx in enumerate(active_idx):
            alpha[idx] = float(alpha_act[k])

        # 6) Normalize the final alpha weights if requested
        if normalize_weights:
            s = float(alpha.sum() + 1e-12)
            alpha = alpha / s

        return alpha, GTG, norms_list

    def compute_joint_direction(
        self,
        grads_per_task: List[List[torch.Tensor]],
        alpha: np.ndarray,
        use_unit_direction: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        """
        Computes the final joint update direction by combining the task gradients 
        weighted by the solved alpha values.
        """
        assert len(grads_per_task) == self.n_tasks
        if use_unit_direction is None:
            use_unit_direction = self.use_unit_gram

        tiny = float(torch.finfo(torch.float32).tiny)

        # Recalculate norms to prepare the directional gradients
        norms = np.zeros(self.n_tasks, dtype=np.float64)
        for t in range(self.n_tasks):
            n2 = self._norm_sq_list(grads_per_task[t])
            n = float(torch.sqrt(torch.clamp(n2, min=0.0)).item())
            norms[t] = 0.0 if n <= tiny else n

        g_dir: List[List[torch.Tensor]] = []
        for t in range(self.n_tasks):
            if norms[t] <= 0.0:
                g_dir.append([torch.zeros_like(x) for x in grads_per_task[t]])
            else:
                if use_unit_direction:
                    g_dir.append(self._scale_list(grads_per_task[t], float(norms[t])))
                else:
                    g_dir.append([x for x in grads_per_task[t]])

        # Accumulate the weighted gradients
        d_list: List[torch.Tensor] = []
        n_params = len(g_dir[0])
        for pidx in range(n_params):
            acc = torch.zeros_like(g_dir[0][pidx])
            for t in range(self.n_tasks):
                w = float(alpha[t])
                if w == 0.0:
                    continue
                acc = acc + w * g_dir[t][pidx]
            d_list.append(acc)
            
        return d_list
