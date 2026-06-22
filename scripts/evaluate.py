import logging
import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ot_gco.cli import parse_eval_args
from ot_gco.data import get_data_loaders
from ot_gco.evaluation import evaluation
from ot_gco.objectives import select_objective
from ot_gco.sampler import VSampler


def main():
    args = parse_eval_args()
    torch.manual_seed(args.seed)
    _, val_loader, _ = get_data_loaders(args, mode="test")

    model = VSampler(args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if Path(args.log_file).exists():
        Path(args.log_file).open("w").close()
    logger.addHandler(logging.FileHandler(args.log_file))

    obj_fn, con_fn = select_objective(args.P)
    evaluation(args, model, val_loader, 0, None, logger, obj_fn, con_fn)


if __name__ == "__main__":
    main()
