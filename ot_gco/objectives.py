import torch
from torch_geometric.nn import global_add_pool


def compute_diversity_loss(y: torch.Tensor, batch):
    batch_size = batch.num_graphs
    diversity = global_add_pool(y - y**2, batch.first_batch).reshape(batch_size, -1)
    return diversity


def max_independent_set_obj(y, batch) -> torch.Tensor:
    batch_size = batch.num_graphs
    obj = global_add_pool(y, batch.first_batch).reshape(batch_size, -1)
    return -obj


def max_independent_set_constraint(y, batch) -> torch.Tensor:
    batch_size = batch.num_graphs
    penalties = global_add_pool(
        y[batch.edge_index[0]] * y[batch.edge_index[1]],
        batch.first_batch[batch.edge_index[0, :]],
    ) / 2
    return penalties.reshape(batch_size, -1)


def max_cut_obj(y, batch) -> torch.Tensor:
    y = 2 * y - 1
    batch_size = batch.num_graphs
    penalties = global_add_pool(
        (1 - y[batch.edge_index[0]] * y[batch.edge_index[1]]) / 2,
        batch.first_batch[batch.edge_index[0, :]],
    ) / 2
    return -penalties.reshape(batch_size, -1)


def max_cut_constraint(y, batch) -> torch.Tensor:
    return torch.zeros((batch.num_graphs, batch.particle_num), device=y.device)


max_cut_constrain = max_cut_constraint


def select_objective(problem_type: str):
    if problem_type == "MaxCut":
        return max_cut_obj, max_cut_constraint
    if problem_type in {"MIS", "MaxClique"}:
        return max_independent_set_obj, max_independent_set_constraint
    raise ValueError(f"Unsupported problem type: {problem_type}")
