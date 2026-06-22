import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.index import Index
from torch_geometric.utils import is_sparse


def get_new_batch(batch: Batch, particle_num: int):
    counts_first = torch.diff(batch.ptr)[:, None].repeat(1, particle_num).reshape(-1) // particle_num
    half_particles = max(particle_num // 2, 1)
    counts_second = torch.diff(batch.ptr)[:, None].repeat(1, half_particles).reshape(-1) // half_particles
    batch_first = torch.arange(batch.num_graphs * particle_num).repeat_interleave(counts_first)
    batch_second = torch.arange(batch.num_graphs * half_particles).repeat_interleave(counts_second)
    return batch_first, batch_second


def batch_load(args, batch):
    if not hasattr(batch, "num_graphs"):
        batch.num_graphs = 1
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=args.device)

    batch.first_batch, batch.second_batch = get_new_batch(batch, args.particle_num)
    batch.penalty = args.penalty_values
    batch.particle_num = args.particle_num
    batch.d_penalty = args.d_penalty
    batch.h = 1.0 / args.T
    batch.T = args.T
    batch.problem = args.P
    return batch.to(args.device)


def complement_edge_index(senders, receivers, num_nodes):
    existing = {
        (int(min(s, r)), int(max(s, r)))
        for s, r in zip(senders.tolist(), receivers.tolist())
        if int(s) != int(r)
    }
    comp_s, comp_r = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) not in existing:
                comp_s.extend([i, j])
                comp_r.extend([j, i])
    return torch.tensor(comp_s, dtype=torch.long), torch.tensor(comp_r, dtype=torch.long)


def _num_nodes_from_jraph(jraph_graph, nodes, senders, receivers):
    if nodes is not None:
        return nodes.size(0)
    if hasattr(jraph_graph, "n_node") and jraph_graph.n_node is not None:
        return int(np.asarray(jraph_graph.n_node).sum())
    if senders.numel() == 0 and receivers.numel() == 0:
        raise ValueError("Cannot infer num_nodes from an empty graph without nodes or n_node.")
    return int(torch.cat([senders, receivers]).max().item()) + 1


def jraph_to_pyg(jraph_graph, particle_num, problem_type):
    nodes = torch.from_numpy(np.asarray(jraph_graph.nodes)) if jraph_graph.nodes is not None else None
    raw_senders = torch.from_numpy(np.asarray(jraph_graph.senders)).long()
    raw_receivers = torch.from_numpy(np.asarray(jraph_graph.receivers)).long()
    num_nodes = _num_nodes_from_jraph(jraph_graph, nodes, raw_senders, raw_receivers)

    if problem_type == "MaxClique":
        senders, receivers = complement_edge_index(raw_senders, raw_receivers, num_nodes)
    else:
        senders, receivers = raw_senders, raw_receivers

    edge_index_base = torch.stack([senders, receivers], dim=0)
    edge_index = torch.cat(
        [edge_index_base + i * num_nodes for i in range(particle_num)],
        dim=1,
    ).long()

    x = torch.rand((particle_num, num_nodes))
    return my_Data(
        x=x.reshape(-1, 1),
        edge_index=edge_index,
        edges=edge_index_base,
        num_nodes_=num_nodes,
    )


class my_Data(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if "batch" in key and isinstance(value, Tensor):
            if isinstance(value, Index):
                return value.get_dim_size()
            return int(value.max()) + 1
        if "index" in key or key == "face":
            return self.num_nodes
        if key == "edges":
            return self.num_nodes_
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if is_sparse(value) and ("adj" in key or "edge_index" in key):
            return (0, 1)
        if "index" in key or key == "face" or key == "edges":
            return -1
        return 0


class my_Dataset(Dataset):
    def __init__(self, particle_num: int, problem_type: str, root: Path, test_num=None, transform=None):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset split does not exist: {root}")

        files = sorted(path for path in root.iterdir() if path.is_file())
        self.root = files[:test_num] if test_num is not None else files
        self.max_len = 0
        self.particle_num = particle_num
        self.problem = problem_type
        self.transform = transform

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):
        file_path = self.root[idx]
        with file_path.open("rb") as f:
            data_dict = pickle.load(f)
        data = data_dict["H_graphs"]

        if self.transform:
            data = self.transform(data, self.particle_num, self.problem)

        self.max_len = max(self.max_len, data.num_nodes)
        return data


def get_data_loaders(args, mode="train"):
    data_path = Path(args.data_path)
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"

    train_set = (
        my_Dataset(args.particle_num, args.P, train_path, transform=jraph_to_pyg)
        if mode == "train"
        else None
    )
    val_set = my_Dataset(args.particle_num, args.P, val_path, test_num=args.test_num, transform=jraph_to_pyg)
    test_set = my_Dataset(args.particle_num, args.P, test_path, transform=jraph_to_pyg)

    train_loader = (
        DataLoader(train_set, batch_size=args.batch_size, collate_fn=Batch.from_data_list, shuffle=True)
        if mode == "train"
        else None
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=Batch.from_data_list, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=Batch.from_data_list, shuffle=False)
    return train_loader, val_loader, test_loader
