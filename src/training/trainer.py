import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from .losses import TrajectoryReconstructionLoss


def optimize_batch(model, batch, device, optimizer, params, forward_losses, kl_weight):
    optimizer.zero_grad(set_to_none=True)
    _, losses = forward_losses(model, batch, "train", device, kl_weight)
    losses["total"].backward()
    if any(p.grad is None for p in params):
        raise RuntimeError("Missing gradient in a trainable parameter")
    torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"), error_if_nonfinite=True)
    if any(p.grad is not None for p in model.siglip_encoder.parameters()):
        raise RuntimeError("Frozen SigLIP received gradients")
    optimizer.step()
    return {key: value.detach() for key, value in losses.items()}


class EpochTrainer:
    def __init__(self, model, device, lr, kl_weight, forward_losses):
        self.model = model
        self.device = device
        self.kl_weight = kl_weight
        self.forward_losses = forward_losses
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.reconstruction = TrajectoryReconstructionLoss()

    def clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    def train_epoch(self, loader, log_every):
        start = self.clock()
        totals = {"reconstruction": 0.0, "kl": 0.0, "total": 0.0}
        samples = 0
        for step, batch in enumerate(loader, start=1):
            losses = optimize_batch(
                self.model, batch, self.device, self.optimizer, self.params,
                self.forward_losses, self.kl_weight,
            )
            size = batch["future_motion"].shape[0]
            samples += size
            for name, value in losses.items():
                totals[name] += value.item() * size
            if step == 1 or step % log_every == 0 or step == len(loader):
                elapsed = self.clock() - start
                remaining = elapsed / step * (len(loader) - step)
                print(f"train {step}/{len(loader)}: total={totals['total'] / samples:.6f}, "
                      f"elapsed={elapsed:.1f}s, train_eta={remaining:.1f}s", flush=True)
        if not samples:
            raise RuntimeError("Empty training loader")
        return {name: value / samples for name, value in totals.items()}, samples, self.clock() - start

    @torch.no_grad()
    def validate_epoch(self, loader, log_every=100):
        start = self.clock()
        error = baseline_error = 0.0
        samples = 0
        for step, batch in enumerate(loader, start=1):
            _, losses = self.forward_losses(self.model, batch, "val", self.device, self.kl_weight)
            target = batch["future_motion"].to(self.device)
            baseline = torch.zeros_like(target)
            baseline[..., 3] = 1
            baseline[..., 7] = 1
            size = target.shape[0]
            samples += size
            error += losses["reconstruction"].item() * size
            baseline_error += self.reconstruction(baseline, target).item() * size
            if step == 1 or step % log_every == 0 or step == len(loader):
                elapsed = self.clock() - start
                remaining = elapsed / step * (len(loader) - step)
                print(f"val {step}/{len(loader)}: mse={error / samples:.6f}, "
                      f"elapsed={elapsed:.1f}s, val_eta={remaining:.1f}s", flush=True)
        if not samples:
            raise RuntimeError("Empty validation loader")
        return {"reconstruction": error / samples, "stationary": baseline_error / samples}, samples, self.clock() - start

    def save_checkpoint(self, path, epoch, metrics, siglip_source):
        trainable_state = {
            name: module.state_dict()
            for name, module in (
                ("prior", self.model.prior),
                ("motion_encoder", self.model.motion_encoder),
                ("motion_decoder", self.model.motion_decoder),
            )
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".checkpoint_", dir=path.parent) as directory:
            temporary = Path(directory) / path.name
            torch.save({
                "epoch": epoch, "model": trainable_state,
                "optimizer": self.optimizer.state_dict(), "metrics": metrics,
                "siglip": siglip_source,
            }, temporary)
            temporary.replace(path)

    def run(self, train_loader, val_loader, epochs, log_every, run_dir, siglip_source, *, patience=0):
        run_dir.mkdir(parents=True)
        best = None
        stale_epochs = 0
        print(f"Run directory: {run_dir}", flush=True)
        for epoch in range(1, epochs + 1):
            start = self.clock()
            print(f"Epoch {epoch}/{epochs}", flush=True)
            train, train_count, train_seconds = self.train_epoch(train_loader, log_every)
            val, val_count, val_seconds = self.validate_epoch(val_loader, log_every)
            checkpoint = run_dir / f"epoch_{epoch:03d}.pt"
            save_start = self.clock()
            metrics = {
                "epoch": epoch,
                "train": train, "val": val, "train_samples": train_count, "val_samples": val_count,
                "train_seconds": train_seconds, "val_seconds": val_seconds,
            }
            self.save_checkpoint(checkpoint, epoch, metrics, siglip_source)
            save_seconds = self.clock() - save_start
            elapsed = self.clock() - start
            improved = best is None or val["reconstruction"] < best["validation_mse"]
            if improved:
                best = {
                    "epoch": epoch,
                    "validation_mse": val["reconstruction"],
                    "stationary_mse": val["stationary"],
                    "checkpoint": checkpoint.name,
                }
                best_path = run_dir / "best.json"
                temporary = best_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(best, indent=2) + "\n")
                temporary.replace(best_path)
                stale_epochs = 0
            else:
                stale_epochs += 1
            metrics.update(save_seconds=save_seconds, elapsed_seconds=elapsed,
                           checkpoint=checkpoint.name, is_best=improved)
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(f"train: {train_count} samples, reconstruction={train['reconstruction']:.6f}, "
                  f"kl={train['kl']:.6f}, total={train['total']:.6f}")
            print(f"val: {val_count} samples, reconstruction={val['reconstruction']:.6f}, "
                  f"stationary={val['stationary']:.6f}")
            print(f"epoch elapsed: {elapsed:.1f}s "
                  f"(train={train_seconds:.1f}s, val={val_seconds:.1f}s, save={save_seconds:.1f}s)")
            print(f"checkpoint: {checkpoint}", flush=True)
            print(f"best: epoch {best['epoch']}, val_mse={best['validation_mse']:.6f}, "
                  f"checkpoint={run_dir / best['checkpoint']}", flush=True)
            if patience and stale_epochs >= patience:
                print(f"Early stopping: validation MSE did not improve for {stale_epochs} epochs.", flush=True)
                break
