import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, MLP, LayerNorm, MessagePassing, global_mean_pool
from torch_geometric.nn.encoding import PositionalEncoding


class Node_Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.particle_num = args.particle_num
        self.time_emb = PositionalEncoding(out_channels=self.emb_dim)
        self.prob_embedding = nn.Embedding(2, self.emb_dim)
        self.node_embedding = nn.Embedding(2, self.emb_dim)
        self.time_node_embedding = nn.Sequential(
            nn.Linear(2 * self.emb_dim, 2 * self.emb_dim),
            nn.LayerNorm(2 * self.emb_dim),
            nn.ReLU(),
            nn.Linear(2 * self.emb_dim, self.emb_dim),
        )

    def forward(self, particle: torch.Tensor, samples: torch.Tensor, t):
        particle = torch.cat([1 - particle, particle], dim=-1)
        prob_emb = torch.matmul(particle, self.prob_embedding.weight)

        if t is None:
            return prob_emb

        t_emb = self.time_emb(t)
        return self.time_node_embedding(torch.cat([prob_emb, t_emb], dim=1))


class Stein_Net(MessagePassing):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.emb_dim, args.emb_dim)
        self.norm = nn.LayerNorm(args.emb_dim)
        self.attn = nn.Linear(2 * args.emb_dim, 1)
        self.ac = nn.LeakyReLU()

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor):
        x = self.ac(self.norm(node_emb + self.linear(node_emb)))
        return self.propagate(edge_index=edge_index, x=x)

    def message(self, x_i, x_j):
        x_i_, x_j_ = x_i[None, ...], x_j[None, ...]
        x = torch.cat([x_i_ - x_j_, x_i_], dim=0)
        x_tag = torch.cat([x_i_, x_i_ - x_j_], dim=-1)
        x_src = torch.cat([x_i_, x_i_], dim=-1)
        attn = self.attn(torch.cat([x_tag, x_src], dim=0)).softmax(dim=0)
        return torch.sum(attn * x, dim=0)


class Conv_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.global_emb = nn.Sequential(
            nn.Linear(args.emb_dim * 2, args.emb_dim * 2),
            LayerNorm(args.emb_dim * 2),
            nn.ReLU(),
            nn.Linear(args.emb_dim * 2, args.emb_dim),
        )
        self.batch_norms = LayerNorm(args.emb_dim)
        self.conv = GINConv(
            nn.Sequential(
                nn.Linear(args.emb_dim, args.emb_dim * 2),
                LayerNorm(args.emb_dim * 2),
                nn.ReLU(),
                nn.Linear(args.emb_dim * 2, args.emb_dim),
                LayerNorm(args.emb_dim),
            ),
            train_eps=True,
        )

    def forward(self, node_emb, batch):
        graph_emb = global_mean_pool(node_emb, batch.first_batch)
        grad_emb = self.global_emb(torch.cat([node_emb, graph_emb[batch.first_batch]], dim=1))
        grad_emb = self.batch_norms(grad_emb)
        return self.conv(grad_emb, batch.edge_index)


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.node_embedding = Node_Embedding(args)
        self.conv = nn.ModuleList([Conv_Net(args) for _ in range(args.layers_num)])
        self.conv_norm = nn.ModuleList([LayerNorm(args.emb_dim) for _ in range(args.layers_num)])
        self.stein_update = nn.ModuleList(
            [GATv2Conv(args.emb_dim, args.emb_dim, heads=1) for _ in range(args.layers_num)]
        )
        self.stein_norm = nn.ModuleList([LayerNorm(args.emb_dim) for _ in range(args.layers_num)])
        self.global_norm = nn.LayerNorm(args.emb_dim)
        self.jk = MLP(
            channel_list=[args.layers_num * args.emb_dim, args.layers_num * args.emb_dim // 2, args.emb_dim],
            act=torch.nn.ReLU(),
            norm="LayerNorm",
        )
        self.out_mu = MLP(
            channel_list=[args.emb_dim, args.emb_dim // 2, 1],
            act=torch.nn.ReLU(),
            norm="LayerNorm",
        )
        self.out_sigma = MLP(
            channel_list=[args.emb_dim, args.emb_dim // 2, 1],
            act=torch.nn.ReLU(),
            norm="LayerNorm",
        )

    def forward(self, particle: torch.Tensor, batch, t):
        samples = torch.bernoulli(particle)
        node_emb = self.node_embedding(particle, samples, t)
        hiddens = []

        for idx, conv in enumerate(self.conv):
            node_emb = self.conv_norm[idx](node_emb + conv(node_emb, batch)).relu()
            hiddens.append(node_emb)

        node_emb = self.global_norm(self.jk(torch.cat(hiddens, dim=1)))
        return self.out_mu(node_emb).sigmoid()


class Noise_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.node_embedding = Node_Embedding(args)
        self.mlp = MLP(
            channel_list=[args.emb_dim, args.emb_dim // 2, 1],
            act=torch.nn.ReLU(),
            norm="LayerNorm",
        )

    def forward(self, particle: torch.Tensor, t):
        samples = torch.bernoulli(particle)
        node_emb = self.node_embedding(particle, samples, t)
        return F.tanh(self.mlp(node_emb)).reshape(-1, 1) * 2
