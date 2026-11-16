import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .prepare_vl_sample import OUT, require, rotation_matrices
else:
    from prepare_vl_sample import OUT, require, rotation_matrices


DEFAULT_SAMPLE = OUT / "episode_000741_frame_000089_h10_s3_wxyz.npz"


def verify(sample):
    meta = json.loads(sample.with_suffix(".json").read_text())
    require(meta["format"] == "leverb_vl_pilot_v1", "Unsupported sample format")
    with np.load(sample, allow_pickle=False) as data:
        state = data["current_state"].astype(np.float64)
        target = data["future_motion"].astype(np.float64)
        pos = data["raw_body_pos"].astype(np.float64)
        quat = data["raw_body_quat"].astype(np.float64)

    links = meta["pilot_links"]
    horizon = meta["horizon"]
    require(len(links) == len(set(links)) == 12, "Expected 12 distinct pilot links")
    require(state.shape == (38,) and target.shape == (horizon, 45), "Unexpected feature shapes")
    require(pos.shape[0] == quat.shape[0] == horizon + 1, "Raw window length differs")
    require(all(np.isfinite(a).all() for a in (state, target, pos, quat)), "Non-finite values")

    pos_names, quat_names = meta["raw_body_pos_names"], meta["raw_body_quat_names"]
    root = pos[:, pos_names.index(meta["root_link"])]
    rotation = rotation_matrices(quat[:, quat_names.index(meta["root_link"])],
                                 meta["assumed_quaternion_order"])
    link_indices = [pos_names.index(link) for link in links]

    x, y = target[:, 3:9].reshape(horizon, 2, 3).transpose(1, 0, 2)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    require(np.all(x_norm > 1e-8), "Invalid first rotation column")
    x = x / x_norm
    y = y - np.sum(x * y, axis=1, keepdims=True) * x
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    require(np.all(y_norm > 1e-8), "Rotation columns are collinear")
    y = y / y_norm
    relative_rotation = np.stack((x, y, np.cross(x, y)), axis=-1)

    recovered_root = root[0] + target[:, :3]
    recovered_rotation = rotation[0] @ relative_rotation
    recovered_local = state[2:].reshape(1, 12, 3) + target[:, 9:].reshape(horizon, 12, 3)
    recovered_links = recovered_root[:, None] + recovered_local @ recovered_rotation.transpose(0, 2, 1)

    root_error = np.linalg.norm(recovered_root - root[1:], axis=-1).max()
    link_error = np.linalg.norm(recovered_links - pos[1:, link_indices], axis=-1).max()
    matrix_error = np.linalg.norm(recovered_rotation - rotation[1:], axis=(1, 2)).max()
    angle_error = np.degrees(2 * np.arcsin(np.clip(matrix_error / (2 * np.sqrt(2)), 0, 1)))
    print(f"Episode {meta['episode']}: checked {horizon} future steps")
    print(f"Root position max error: {root_error * 1000:.6f} mm")
    print(f"Root rotation max error: {angle_error:.6f} degrees")
    print(f"Body link position max error: {link_error * 1000:.6f} mm")
    require(root_error < 1e-4 and link_error < 1e-4 and angle_error < 0.01,
            "Decoded trajectory does not match the raw demonstration")
    print("PASS: saved motion targets reconstruct the raw future trajectory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare decoded pilot motion with the saved raw trajectory.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    verify(parser.parse_args().sample)
