import argparse
from datetime import datetime, timezone
from itertools import islice
import math
from pathlib import Path

import torch

from src.data_loading.vl_dataset import VLDataLoader, VLDataset
from src.models.leverb_vl import LeVERBVL
from src.models.siglip_encoder import MODEL_ID, MODEL_REVISION, SigLIPEncoder
from src.training.losses import LatentKLLoss, TrajectoryReconstructionLoss
from src.training.trainer import EpochTrainer, optimize_batch


RECONSTRUCTION_LOSS = TrajectoryReconstructionLoss()
KL_LOSS = LatentKLLoss()


def forward_losses(model, batch, split, device, kl_weight):
    training = split == "train"
    model.train(training)
    target_motion = batch["future_motion"].to(device)
    arguments = {"future_motion": target_motion} if training else {}
    output = model(
        batch["fpv_cam_rgb"],
        batch["tpv_cam_rgb"],
        batch["instruction"],
        batch["current_state"].to(device),
        **arguments,
    )
    prediction = output["predicted_motion"]
    if prediction.shape != target_motion.shape:
        raise RuntimeError(f"{split}: unexpected prediction shape {tuple(prediction.shape)}")
    latent_shape = (prediction.shape[0], model.motion_decoder.latent_dim)
    for key in ("prior_mean", "prior_log_variance", "z"):
        if output[key].shape != latent_shape:
            raise RuntimeError(f"{split}: unexpected {key} shape {tuple(output[key].shape)}")
    for key in ("posterior_mean", "posterior_log_variance"):
        if (key in output) != training or (training and output[key].shape != latent_shape):
            raise RuntimeError(f"{split}: unexpected posterior output {key}")
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError(f"{split}: nonfinite model output")
    if model.siglip_encoder.training or any(p.requires_grad for p in model.siglip_encoder.parameters()):
        raise RuntimeError("SigLIP must remain frozen in evaluation mode")
    if not training:
        torch.testing.assert_close(output["z"], output["prior_mean"], rtol=0, atol=0)

    reconstruction = RECONSTRUCTION_LOSS(prediction, target_motion)
    losses = {"reconstruction": reconstruction}
    if training:
        kl = KL_LOSS(
            output["posterior_mean"], output["posterior_log_variance"],
            output["prior_mean"], output["prior_log_variance"],
        )
        losses.update(kl=kl, total=reconstruction + kl_weight * kl)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"{split}: nonfinite loss")
    return output, losses


def format_losses(losses):
    return ", ".join(f"{key}={value.item():.6f}" for key, value in losses.items())


@torch.no_grad()
def check_forward(model, batch, split, device, kl_weight):
    output, losses = forward_losses(model, batch, split, device, kl_weight)
    path = "posterior" if split == "train" else "prior mean"
    print(f"{split}: {path}; z={tuple(output['z'].shape)}, "
          f"predicted_motion={tuple(output['predicted_motion'].shape)}")
    print(f"{split} losses: {format_losses(losses)}")


def run_train_steps(model, loader, device, steps, lr, kl_weight):
    if steps > len(loader):
        raise ValueError(f"Requested {steps} steps, but the training loader has {len(loader)} batches")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    for step, batch in enumerate(islice(loader, steps), start=1):
        losses = optimize_batch(model, batch, device, optimizer, params, forward_losses, kl_weight)
        print(f"step {step}/{steps}: {format_losses(losses)}")


def main():
    parser = argparse.ArgumentParser(description="Check forward passes, run a short trial, or train by epoch.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-steps", type=int, default=0)
    mode.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many epochs without lower validation MSE (0 disables early stopping)")
    parser.add_argument("--siglip-path", help="Optional local SigLIP model directory")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.train_steps < 0:
        parser.error("--train-steps must be nonnegative")
    if args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    if args.num_workers < 0 or args.log_every < 1:
        parser.error("--num-workers must be nonnegative and --log-every must be positive")
    if args.run_dir is not None and not args.epochs:
        parser.error("--run-dir requires --epochs")
    if args.patience < 0 or (args.patience and not args.epochs):
        parser.error("--patience must be nonnegative and requires --epochs when enabled")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.kl_weight) or args.kl_weight < 0:
        parser.error("--kl-weight must be finite and nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check the environment or use --device cpu.")

    device = torch.device(args.device)
    encoder = SigLIPEncoder.from_pretrained(args.siglip_path, local_files_only=args.local_files_only)
    model = LeVERBVL(encoder).to(device)
    train_loader = VLDataLoader(
        VLDataset("train"), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=bool(args.train_steps or args.epochs),
    )
    val_loader = VLDataLoader(
        VLDataset("val"), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
    )
    if args.epochs:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.run_dir or Path(__file__).resolve().parent / "runs" / f"vl_{stamp}"
        source = {"path": str(Path(args.siglip_path).resolve())} if args.siglip_path else {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
        }
        trainer = EpochTrainer(model, device, args.lr, args.kl_weight, forward_losses)
        trainer.run(
            train_loader, val_loader, args.epochs, args.log_every, run_dir.resolve(), source,
            patience=args.patience,
        )
        return
    if args.train_steps:
        run_train_steps(model, train_loader, device, args.train_steps, args.lr, args.kl_weight)
    else:
        check_forward(model, next(iter(train_loader)), "train", device, args.kl_weight)
    check_forward(model, next(iter(val_loader)), "val", device, args.kl_weight)
    if args.train_steps:
        print(f"PASS: {args.train_steps} training updates and validation forward. No checkpoint saved.")
    else:
        print("PASS: training and validation forward paths and losses. No parameters updated.")


if __name__ == "__main__":
    main()
