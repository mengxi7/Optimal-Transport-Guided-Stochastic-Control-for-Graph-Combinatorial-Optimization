import time

import torch

from .data import batch_load
from .decoding import proj_loss
from .sampler import VSampler, inference


@torch.no_grad()
def evaluation(args, model: VSampler, data_loader, best_score, save_path, logger, obj_fn, con_fn):
    obj_proj_min_total, obj_proj_total = 0, 0
    num_graph, num_batches, cost_time = 0, 0, 0

    for batch in data_loader:
        batch = batch_load(args, batch)
        start_time = time.time()
        _, outputs = inference(args, model, batch)
        end_time = time.time()

        obj_projs, obj_proj_mins = [], []
        for particle in outputs:
            obj_proj, _, obj_proj_min, _ = proj_loss(particle, batch, obj_fn, con_fn)
            obj_projs.append(obj_proj.cpu().item())
            obj_proj_mins.append(obj_proj_min.cpu().item())

        logger.info(f"obj_projs:{obj_proj_mins}")
        logger.info(f"best_obj_proj:{min(obj_proj_mins)}")
        obj_proj_total += min(obj_projs)
        obj_proj_min_total += min(obj_proj_mins)
        num_graph += batch.num_graphs
        num_batches += 1
        cost_time += end_time - start_time
        torch.cuda.empty_cache()

    logger.info(
        f"obj: {obj_proj_total / num_batches :.2f}, "
        f"obj_min: {obj_proj_min_total / num_batches :.2f}, "
        f"cost_time:{cost_time / num_graph :.2f}, cost_time_t:{cost_time}"
    )

    if obj_proj_min_total / num_graph < best_score and save_path:
        torch.save(model.state_dict(), save_path / "best.pth")
        best_score = obj_proj_min_total / num_graph

    return best_score
