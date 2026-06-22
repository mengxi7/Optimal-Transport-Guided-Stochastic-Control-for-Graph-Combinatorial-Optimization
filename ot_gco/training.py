import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR
from torch_geometric.utils import scatter

from .data import batch_load, get_data_loaders
from .decoding import proj_loss
from .objectives import compute_diversity_loss, select_objective
from .sampler import VSampler, effective_epsilon, inference


def get_optimizer(model, loader_len, args):
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    if args.lr_scheduler == "Cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * loader_len)
    elif args.lr_scheduler == "Constant":
        lr_scheduler = ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Invalid lr_scheduler: {args.lr_scheduler}")

    return optimizer, lr_scheduler


def loss_fn(out_o: torch.Tensor, out_f: torch.Tensor, data: tuple[torch.Tensor, float], batch, T, obj_fn, con_fn, args):
    particle, t = data

    denom = max(1.0 - t, args.delta)
    epsilon = max(effective_epsilon(args), 1e-8)
    regu = scatter((out_f - particle) ** 2, batch.first_batch).mean() / (4.0 * epsilon * denom)
    obj = obj_fn(out_f, batch).mean()
    con = con_fn(out_f, batch).mean()
    div = compute_diversity_loss(out_f, batch).mean()
    loss_1 = obj + batch.penalty * con + batch.d_penalty * div + regu

    drift = (out_f - particle) / denom
    norm = scatter(drift * drift, batch.first_batch).mean() * 0.5
    cos = torch.tensor([0.0], device=out_f.device)
    final_loss = torch.tensor([0.0], device=out_f.device)

    obj_mean, con_mean, obj_min, con_min = proj_loss(out_f, batch, obj_fn, con_fn)
    return loss_1, norm, cos, final_loss, obj, con, div, regu, obj_min, con_min, obj_mean, con_mean


def param_write(path: Path, args):
    write_file_path = path / "param.txt"
    with write_file_path.open("w") as f:
        for args_name, value in vars(args).items():
            f.write(f"{args_name}: {value}\n")


def train_batch(buffer, batch, model: VSampler, optimizer, lr_scheduler, mini_epoch, obj_fn, con_fn, logger, args):
    obj_mins_ = []
    info = {}
    f_min, f_mean, f = 1e10, 1e10, 1e10
    loss_total, norm_total, cos_total, final_loss_total = 0, 0, 0, 0

    for _ in range(mini_epoch):
        for data in buffer:
            out_f, _ = model.forward(data, batch)
            out_o = model.optimize(data, batch)
            loss_1, norm, cos, final_loss, obj, con, div, regu, obj_min, con_min, obj_mean, con_mean = loss_fn(
                out_o,
                out_f,
                data,
                batch,
                args.T,
                obj_fn,
                con_fn,
                args,
            )

            optimizer.zero_grad()
            loss_1.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            obj_mins_.append(obj_min.cpu().item())
            if f_min > obj_min + batch.penalty * con_min:
                f_min = obj_min + batch.penalty * con_min
                info["obj_min"] = round(obj_min.cpu().item(), 2)
                info["con_min"] = round(con_min.cpu().item(), 2)

            if f_mean > obj_mean + batch.penalty * con_mean:
                f_mean = obj_mean + batch.penalty * con_mean
                info["obj_mean"] = round(obj_mean.cpu().item(), 2)
                info["con_mean"] = round(con_mean.cpu().item(), 2)

            if f > obj + batch.penalty * con + batch.d_penalty * div:
                f = obj + batch.penalty * con + batch.d_penalty * div
                info["obj"] = round(obj.cpu().item(), 2)
                info["con"] = round(con.cpu().item(), 2)
                info["div"] = round(div.cpu().item(), 2)

            loss_total += loss_1.cpu().item()
            cos_total += cos.cpu().item()
            norm_total += norm.cpu().item()
            final_loss_total += final_loss.cpu().item()

        logger.info(f"obj_mins_:{obj_mins_}")

    lr_scheduler.step()

    denom = max(len(buffer), 1) * max(mini_epoch, 1)
    info["loss"] = round(loss_total / denom, 2)
    info["cos_total"] = round(cos_total / denom, 2)
    info["norm_total"] = round(norm_total / denom, 2)
    info["lr"] = round(lr_scheduler.get_last_lr()[0], 5)
    info["final_loss"] = round(final_loss_total / max(mini_epoch, 1), 2)
    return info


def train(args):
    train_loader, _, _ = get_data_loaders(args)
    full_params = {k: v for k, v in vars(args).items()}
    param_str = json.dumps(full_params, sort_keys=True)
    hash_str = hashlib.md5(param_str.encode()).hexdigest()[:6]

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if Path(args.log_file).exists():
        Path(args.log_file).open("w").close()
    logger.addHandler(logging.FileHandler(args.log_file))

    model = VSampler(args).to(args.device)
    optimizer, lr_scheduler = get_optimizer(model, len(train_loader), args)

    save_path = Path(args.save_path) / f"{args.name}/"
    save_path.mkdir(parents=True, exist_ok=True)
    save_epoch_path = save_path / date.today().strftime("%Y-%m-%d") / hash_str
    save_epoch_path.mkdir(parents=True, exist_ok=True)

    if args.checkpoints:
        model.load_state_dict(torch.load(Path(args.checkpoints), map_location=args.device))
        logger.info(f"Load model from {args.checkpoints}")

    obj_fn, con_fn = select_objective(args.P)
    param_write(save_epoch_path, args)

    for epoch in range(args.epochs):
        logger.info(f"---------------epoch{epoch}---------------")
        torch.save(model.state_dict(), save_epoch_path / f"epoch{epoch}.pth")

        for batch in train_loader:
            batch = batch_load(args, batch)
            buffer, _ = inference(args, model, batch)
            info = train_batch(
                buffer,
                batch,
                model,
                optimizer,
                lr_scheduler,
                args.miniepoch,
                obj_fn,
                con_fn,
                logger,
                args,
            )
            logger.info(", ".join(f"{k}:{v}" for k, v in info.items()))
