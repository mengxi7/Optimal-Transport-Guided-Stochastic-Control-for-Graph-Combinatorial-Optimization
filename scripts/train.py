import os
import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ot_gco.cli import parse_train_args
from ot_gco.training import train


def main():
    args = parse_train_args()
    torch.manual_seed(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train(args)


if __name__ == "__main__":
    main()
