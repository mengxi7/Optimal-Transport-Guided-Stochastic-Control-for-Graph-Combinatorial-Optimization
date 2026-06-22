import argparse


def add_common_args(parser):
    parser.add_argument("--particle_num", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--name", type=str, default="ot_gco")
    parser.add_argument("--arrive_p", type=float, default=1.0)
    parser.add_argument("--refine_p", type=float, default=1.0)
    parser.add_argument("--data_path", type=str, default="data/RB_iid_small")
    parser.add_argument("--log_file", type=str, default="train.log")
    parser.add_argument("--penalty_milestones", type=list, default=[])
    parser.add_argument("--penalty_values", type=float, default=1.0)
    parser.add_argument("--d_penalty", type=float, default=1.0)
    parser.add_argument("--div_penalty", type=float, default=1.0)
    parser.add_argument("--regulization_penalty", type=float, default=0.0)
    parser.add_argument("--entropy_penalty", type=float, default=0.0)
    parser.add_argument("--entropy_decay", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="model_path")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "SGD", "AdamW"])
    parser.add_argument("--lr_scheduler", type=str, default="Cosine", choices=["Cosine", "Constant"])
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--layers_num", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.0, help="Initial probability for step-mask noise.")
    parser.add_argument("--noise_decay", type=float, default=5.0, help="Decay for step-mask noise.")
    parser.add_argument("--noise_value", type=float, default=0.0, help="Value assigned to masked entries.")
    parser.add_argument("--epsilon", type=float, default=0.7, help="Euler-Maruyama Gaussian noise scale.")
    parser.add_argument("--delta", type=float, default=0.01, help="Terminal time truncation.")
    parser.add_argument("--P", type=str, default="MaxCut", choices=["MaxCut", "MIS", "MaxClique"])
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_train_args():
    parser = argparse.ArgumentParser(
        description="Train the OT-guided stochastic-control solver for graph combinatorial optimization."
    )
    add_common_args(parser)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--miniepoch", type=int, default=5)
    return parser.parse_args()


def parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OT-guided stochastic-control solver for graph combinatorial optimization."
    )
    add_common_args(parser)
    parser.set_defaults(mode="test", batch_size=1, test_num=128, T=30, optimizer="Adam")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a model state_dict checkpoint.")
    args = parser.parse_args()
    if not args.checkpoint:
        args.checkpoint = args.checkpoints
    if not args.checkpoint:
        parser.error("--checkpoint is required for evaluation.")
    return args
