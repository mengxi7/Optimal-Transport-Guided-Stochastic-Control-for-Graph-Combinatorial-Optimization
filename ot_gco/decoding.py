import torch


def refine_mis_vector(outputs, probs, A, B):
    mask = outputs[A] + outputs[B] == 2

    prob_A = probs[A[mask], 1]
    prob_B = probs[B[mask], 1]
    change_A = prob_A < prob_B
    change_B = ~change_A

    outputs2 = torch.zeros_like(outputs, device=outputs.device)
    outputs2[A[mask][change_A]] = 1
    outputs2[B[mask][change_B]] = 1

    mask2 = outputs2[A] + outputs2[B] == 2
    prob_A = probs[A[mask2], 1]
    prob_B = probs[B[mask2], 1]
    change_A = prob_A > prob_B
    change_B = ~change_A

    outputs[A[mask2][change_A]] = 0
    outputs[B[mask2][change_B]] = 0

    mask = outputs[A] + outputs[B] == 2
    prob_A = probs[A[mask], 1]
    prob_B = probs[B[mask], 1]
    change_A = prob_A < prob_B
    change_B = ~change_A
    outputs[A[mask][change_A]] = 0
    outputs[B[mask][change_B]] = 0
    return outputs


def decode_func(par_, graph):
    ptr = graph.ptr.tolist()
    edge_index = graph.edges.long()
    par = torch.cat(
        [par_[ptr[i] : ptr[i + 1]].reshape(graph.particle_num, -1) for i in range(graph.num_graphs)],
        dim=1,
    )
    mask = torch.ones_like(par).bool()
    solution = torch.zeros_like(par)

    while mask.any():
        prob = torch.where(mask, par, -torch.inf)
        next_node = torch.argmax(prob, 1)
        indices = torch.arange(par.shape[0], device=par.device)
        flag = mask[indices, next_node]
        indices = indices[flag]
        next_node = next_node[flag]
        solution[indices, next_node] = 1
        mask[indices, next_node] = False

        nnx_masks = edge_index[0].unsqueeze(0) == next_node.unsqueeze(1)
        nnx_indices = torch.nonzero(nnx_masks).T
        mask[indices[nnx_indices[0]], edge_index[1][nnx_indices[1]]] = False

    node_counts = torch.diff(graph.ptr / graph.particle_num).int().tolist()
    solution = torch.split(solution, node_counts, dim=1)
    return torch.cat([value.reshape(-1, 1) for value in solution], dim=0)


def proj_loss(particle_T, batch, obj_fn, con_fn):
    x = particle_T
    probs = torch.cat([1 - x, x], dim=-1)

    if batch.problem == "MaxCut":
        outputs = torch.argmax(probs, dim=-1)
    else:
        outputs = decode_func(x, batch)

    obj, penalty = obj_fn(outputs.float(), batch), con_fn(outputs.float(), batch)
    score = obj + batch.penalty * penalty
    index = torch.argmin(score, dim=-1)
    obj_min = obj[torch.arange(obj.size(0), device=obj.device), index]
    penalty_min = penalty[torch.arange(penalty.size(0), device=penalty.device), index]

    return obj.mean(), penalty.mean(), obj_min.mean(), penalty_min.mean()
