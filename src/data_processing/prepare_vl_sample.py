import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

import av
import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data/raw/LeVERB-Bench-Dataset"
PROCESSED = PROJECT / "data/processed"
OUT = PROCESSED / "vl_pilot"
SELECTION = PROCESSED / "subset_570.json"
SPLITS = PROCESSED / "vl_splits.json"
BATCH = PROCESSED / "vl_batch"
CAMERAS = ("fpv_cam", "tpv_cam")
LINKS = tuple(
    f"{side}_{joint}_link"
    for joint in ("hip_pitch", "knee", "ankle_roll", "shoulder_pitch", "elbow", "wrist_yaw")
    for side in ("left", "right")
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rotation_matrices(quat, order):
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    require(np.allclose(norm, 1, atol=1e-3, rtol=0), "Invalid root quaternions")
    w, x, y, z = (quat / norm).T if order == "wxyz" else (quat / norm)[:, [3, 0, 1, 2]].T
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], axis=1).reshape(-1, 3, 3)


class VLSampleProcessor:
    def __init__(self, raw=RAW, output=OUT):
        self.raw = raw
        self.output = output

    @staticmethod
    def motion_arrays(positions, quaternions, features, quat_order):
        names = features["observation.body_pos"]["names"]
        quat_names = features["observation.body_quat"]["names"]
        root = positions[:, names.index("pelvis")]
        rotation = rotation_matrices(quaternions[:, quat_names.index("pelvis")], quat_order)
        selected = positions[:, [names.index(link) for link in LINKS]]
        local = np.einsum("tji,tkj->tki", rotation, selected - root[:, None, :])
        r = rotation[0]
        pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
        roll = np.arctan2(r[2, 1], r[2, 2])
        current = np.concatenate(([pitch, roll], local[0].ravel()))

        delta_xyz = root[1:] - root[0]
        relative_rotation = r.T @ rotation[1:]
        delta_rot6d = relative_rotation[:, :, :2].swapaxes(1, 2).reshape(-1, 6)
        delta_links = (local[1:] - local[0]).reshape(len(root) - 1, -1)
        target = np.concatenate((delta_xyz, delta_rot6d, delta_links), axis=1)
        require(np.isfinite(current).all() and np.isfinite(target).all(), "Non-finite model features")
        return current.astype(np.float32), target.astype(np.float32)


    @staticmethod
    def aligned_rgb(path, index, time, shape):
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            for i, frame in enumerate(video.decode(stream)):
                if i == index:
                    require(frame.pts is not None and frame.time_base is not None, f"Missing PTS: {path}")
                    pts = float(frame.pts * frame.time_base)
                    require(abs(pts - time) <= 1e-4, f"Video and state time disagree: {path}")
                    rgb = frame.to_ndarray(format="rgb24")
                    require(rgb.shape == tuple(shape) and rgb.dtype == np.uint8, f"Unexpected RGB data: {path}")
                    return rgb, pts
        raise ValueError(f"Frame {index} not found in {path}")


    def make_sample(self, episode, frame, horizon, stride, quat_order):
        require(episode >= 0 and frame >= 0 and horizon > 0 and stride > 0, "Invalid window parameters")
        info = json.loads((self.raw / "meta/info.json").read_text())
        features = info["features"]
        fields = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
        parquet = self.raw / info["data_path"].format(**fields)
        table = pq.read_table(parquet, columns=[
            "episode_index", "frame_index", "timestamp", "task_index", "observation.body_pos",
            "observation.body_quat",
        ])
        indices = frame + stride * np.arange(horizon + 1)
        require(indices[-1] < table.num_rows, "Future window extends past the end of the episode")
        rows = table.take(indices).to_pydict()
        require(np.array_equal(rows["frame_index"], indices), "Window frame indices do not match")
        require(all(i == episode for i in rows["episode_index"]), "Window contains a different episode")
        times = np.asarray(rows["timestamp"], dtype=np.float64)
        require(np.allclose(times, indices / info["fps"], atol=1e-5, rtol=0), "Invalid state timestamps")
        task = rows["task_index"][0]
        require(all(i == task for i in rows["task_index"]), "Task changes inside the window")
        tasks = [json.loads(line) for line in (self.raw / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
        instruction = {row["task_index"]: row["task"] for row in tasks}[task]
        require(isinstance(instruction, str) and instruction.strip(), "Empty instruction")

        pos = np.asarray(rows["observation.body_pos"], dtype=np.float32)
        quat = np.asarray(rows["observation.body_quat"], dtype=np.float32)
        for array, key in ((pos, "observation.body_pos"), (quat, "observation.body_quat")):
            require(array.shape == (horizon + 1, *features[key]["shape"]), f"Wrong shape: {key}")
            require(np.isfinite(array).all(), f"Non-finite source data: {key}")
        state, target = self.motion_arrays(pos, quat, features, quat_order)
        require(state.shape == (38,) and target.shape == (horizon, 45), "Wrong model feature shapes")

        arrays = {"current_state": state, "future_motion": target,
                  "instruction": np.asarray(instruction), "frame_indices": indices,
                  "timestamps": times, "raw_body_pos": pos, "raw_body_quat": quat}
        cameras = {}
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            path = self.raw / info["video_path"].format(video_key=key, **fields)
            arrays[f"{camera}_rgb"], pts = self.aligned_rgb(path, frame, times[0], features[key]["shape"])
            cameras[camera] = {"path": str(path.relative_to(self.raw)), "pts_seconds": pts}

        metadata = {
            "format": "leverb_vl_pilot_v1", "episode": episode, "task_index": task,
            "instruction": instruction, "source_parquet": str(parquet.relative_to(self.raw)),
            "current_frame": frame, "future_frames": indices[1:].tolist(), "source_fps": info["fps"],
            "horizon": horizon, "stride": stride, "cameras": cameras,
            "root_link": "pelvis", "pilot_links": list(LINKS), "assumed_quaternion_order": quat_order,
            "raw_body_pos_names": features["observation.body_pos"]["names"],
            "raw_body_quat_names": features["observation.body_quat"]["names"],
            "state_layout": "root pitch, root roll (radians), 12 link xyz in current pelvis frame",
            "target_layout": "pelvis world xyz change, relative rotation 6D (columns of R_t^T R_k), "
                             "change in 12 pelvis-local link positions from current step",
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
        self.output.mkdir(parents=True, exist_ok=True)
        stem = self.output / f"episode_{episode:06d}_frame_{frame:06d}_h{horizon}_s{stride}_{quat_order}"
        with stem.with_suffix(".npz.part").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        stem.with_suffix(".npz.part").replace(stem.with_suffix(".npz"))
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"Episode {episode}, current frame {frame}, future frames {indices[1:].tolist()}")
        print(f"Instruction: {instruction}")
        for key in ("fpv_cam_rgb", "tpv_cam_rgb", "current_state", "future_motion"):
            print(f"{key}: {arrays[key].shape}")
        print(f"Sample: {stem.with_suffix('.npz')}")
        print(f"Metadata: {stem.with_suffix('.json')}")
        print("PASS: one aligned pilot sample written.")


class VLDatasetSplitter:
    def __init__(self, raw=RAW, selection=SELECTION, output=SPLITS):
        self.raw = raw
        self.selection = selection
        self.output = output

    def load_groups(self):
        selection = json.loads(self.selection.read_text())
        episodes = selection["episodes"]
        require(episodes and all(type(i) is int and i >= 0 for i in episodes),
                "Selection must contain nonnegative integer episode IDs")
        require(len(set(episodes)) == len(episodes), "Duplicate selected episodes")
        info = json.loads((self.raw / "meta/info.json").read_text())
        groups = defaultdict(list)
        for episode in sorted(episodes):
            relative = info["data_path"].format(
                episode_index=episode, episode_chunk=episode // info["chunks_size"],
            )
            names = pq.read_table(self.raw / relative, columns=["info.task_name"])["info.task_name"]
            distinct = names.unique().to_pylist()
            require(len(distinct) == 1 and isinstance(distinct[0], str) and distinct[0].strip(),
                    f"Episode {episode} needs exactly one nonempty task name")
            groups[distinct[0]].append(episode)
        return selection, groups

    @staticmethod
    def validation_groups(groups, val_ratio, seed):
        require(0 < val_ratio < 1, "Validation ratio must be between 0 and 1")
        require(len(groups) >= 2, "At least two task groups are required")
        names = sorted(groups)
        random.Random(seed).shuffle(names)
        total = sum(map(len, groups.values()))
        choices = {0: ()}
        for name in names:
            for count, members in tuple(choices.items()):
                choices.setdefault(count + len(groups[name]), members + (name,))
        size = min((n for n in choices if 0 < n < total),
                   key=lambda n: (abs(n - total * val_ratio), n))
        return set(choices[size])

    def split(self, val_ratio, seed):
        selection, groups = self.load_groups()
        val_groups = self.validation_groups(groups, val_ratio, seed)
        train = sorted(episode for name, ids in groups.items() if name not in val_groups for episode in ids)
        val = sorted(episode for name in val_groups for episode in groups[name])
        require(not set(train) & set(val) and sorted(train + val) == sorted(selection["episodes"]),
                "Split must cover every selected episode exactly once")
        result = {
            "format": "leverb_vl_grouped_split_v1",
            "revision": selection.get("revision"),
            "group_by": "info.task_name",
            "seed": seed,
            "requested_validation_ratio": val_ratio,
            "train": train,
            "val": val,
            "groups": {
                name: {"split": "val" if name in val_groups else "train", "episodes": ids}
                for name, ids in sorted(groups.items())
            },
        }
        payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            require(self.output.read_text() == payload,
                    f"Split already exists with different contents: {self.output}. Use a new --output.")
        else:
            self.output.write_text(payload)
        print(f"Task groups: {len(groups)}")
        print(f"Train: {len(train)} episodes ({len(groups) - len(val_groups)} groups)")
        print(f"Validation: {len(val)} episodes ({len(val_groups)} groups)")
        print("PASS: every episode is assigned once, with no task group shared between splits.")
        print(f"Split manifest: {self.output}")


class VLBatchProcessor:
    def __init__(self, raw=RAW, splits=SPLITS, output=BATCH):
        self.raw = raw
        self.splits = splits
        self.output = output
        self.info = json.loads((raw / "meta/info.json").read_text())
        self.tasks = {
            row["task_index"]: row["task"]
            for row in map(json.loads, (raw / "meta/tasks.jsonl").read_text().splitlines())
        }

    def episode_windows(self, episode, horizon, stride, frame_step, quat_order):
        info = self.info
        features = info["features"]
        fields = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
        parquet = info["data_path"].format(**fields)
        pos_key, quat_key = "observation.body_pos", "observation.body_quat"
        table = pq.read_table(self.raw / parquet, columns=[
            "episode_index", "frame_index", "timestamp", "task_index", pos_key, quat_key,
        ])
        rows = table.to_pydict()
        length = table.num_rows
        indices = np.arange(length)
        times = np.asarray(rows["timestamp"], dtype=np.float64)
        require(length > 0 and np.array_equal(rows["frame_index"], indices), f"Invalid frames: {parquet}")
        require(all(i == episode for i in rows["episode_index"]), f"Wrong episode IDs: {parquet}")
        require(np.allclose(times, indices / info["fps"], atol=1e-5, rtol=0), f"Invalid timestamps: {parquet}")
        task_ids = np.asarray(rows["task_index"], dtype=np.int64)
        instructions = {int(task): self.tasks.get(int(task)) for task in np.unique(task_ids)}
        require(all(isinstance(s, str) and s.strip() for s in instructions.values()), f"Invalid tasks: {parquet}")
        pos, quat = (np.asarray(rows[key], dtype=np.float32) for key in (pos_key, quat_key))
        for array, key in ((pos, pos_key), (quat, quat_key)):
            require(array.shape == (length, *features[key]["shape"]) and np.isfinite(array).all(),
                    f"Invalid {key}: {parquet}")
        videos = {
            camera: info["video_path"].format(video_key=f"observation.images.{camera}", **fields)
            for camera in CAMERAS
        }
        for path in videos.values():
            require((self.raw / path).is_file(), f"Missing video: {path}")

        span = horizon * stride
        starts, states, targets = [], [], []
        for start in range(0, length - span, frame_step):
            stop = start + span + 1
            if not np.all(task_ids[start:stop] == task_ids[start]):
                continue
            window = slice(start, stop, stride)
            state, target = VLSampleProcessor.motion_arrays(pos[window], quat[window], features, quat_order)
            starts.append(start)
            states.append(state)
            targets.append(target)

        frames = np.asarray(starts, dtype=np.int64)
        arrays = {
            "current_frame": frames,
            "timestamp": times[frames],
            "task_index": task_ids[frames],
            "current_state": np.asarray(states, dtype=np.float32).reshape(len(frames), 38),
            "future_motion": np.asarray(targets, dtype=np.float32).reshape(len(frames), horizon, 45),
        }
        metadata = {"episode": episode, "source_parquet": parquet, "source_frames": length,
                    "videos": videos, "instructions": instructions, "samples": len(frames)}
        return arrays, metadata

    def run(self, horizon, stride, frame_step, quat_order, limit=None):
        require(horizon > 0 and stride > 0 and frame_step > 0, "Window settings must be positive")
        require(limit is None or limit > 0, "Episode limit must be positive")
        require(not self.output.exists(), f"Output already exists: {self.output}. Use a new --output.")
        splits = json.loads(self.splits.read_text())
        require(splits["format"] == "leverb_vl_grouped_split_v1", "Unsupported split manifest")
        train, val = splits["train"], splits["val"]
        ids = train + val
        require(train and val and all(type(i) is int and i >= 0 for i in ids) and len(ids) == len(set(ids)),
                "Train and validation must contain disjoint, nonempty lists of episode IDs")
        manifest = {
            "format": "leverb_vl_batch_v1", "revision": splits["revision"],
            "split_manifest": str(self.splits), "source_fps": self.info["fps"],
            "horizon": horizon, "stride": stride, "frame_step": frame_step,
            "quat_order": quat_order, "root_link": "pelvis", "pilot_links": list(LINKS),
            "limit_per_split": limit, "splits": {},
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".vl_batch_", dir=self.output.parent) as temporary:
            staging = Path(temporary) / "output"
            for split, episodes in (("train", train), ("val", val)):
                entries = []
                (staging / split).mkdir(parents=True)
                for episode in episodes[:limit]:
                    arrays, entry = self.episode_windows(episode, horizon, stride, frame_step, quat_order)
                    if entry["samples"]:
                        entry["arrays"] = f"{split}/episode_{episode:06d}.npz"
                        np.savez_compressed(staging / entry["arrays"], **arrays)
                    else:
                        entry["arrays"] = None
                    entries.append(entry)
                count = sum(entry["samples"] for entry in entries)
                require(count > 0, f"No usable windows in {split}")
                manifest["splits"][split] = {"samples": count, "episodes": entries}
                skipped = sum(entry["samples"] == 0 for entry in entries)
                print(f"{split}: {len(entries)} episodes, {count} samples, {skipped} without usable windows")
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            staging.rename(self.output)
        print(f"PASS: batch manifest written to {self.output / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LeVERB-VL samples and grouped dataset splits.")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Generate one aligned pilot sample")
    sample.add_argument("--episode", type=int, default=741)
    sample.add_argument("--frame", type=int, default=89)
    sample.add_argument("--horizon", type=int, default=10)
    sample.add_argument("--stride", type=int, default=3, help="Spacing in source frames")
    sample.add_argument("--quat-order", choices=("wxyz", "xyzw"), required=True)
    split = commands.add_parser("split", help="Split selected episodes by complete task name")
    split.add_argument("--val-ratio", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", type=Path, default=SPLITS)
    batch = commands.add_parser("batch", help="Build motion arrays and video frame indices")
    batch.add_argument("--horizon", type=int, default=10)
    batch.add_argument("--stride", type=int, default=3, help="Spacing between target frames")
    batch.add_argument("--frame-step", type=int, default=3, help="Spacing between sample start frames")
    batch.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    batch.add_argument("--limit", type=int, help="Process at most this many episodes per split")
    batch.add_argument("--output", type=Path)

    argv = sys.argv[1:]
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["sample", *argv]
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    if command == "sample":
        VLSampleProcessor().make_sample(**options)
    elif command == "split":
        output = options.pop("output").resolve()
        require(output.is_relative_to(PROCESSED.resolve()), "Split output must be under data/processed/")
        VLDatasetSplitter(output=output).split(**options)
    else:
        default = BATCH if options["limit"] is None else PROCESSED / "vl_batch_preview"
        output = (options.pop("output") or default).resolve()
        require(output.is_relative_to(PROCESSED.resolve()) and output != PROCESSED.resolve(),
                "Batch output must be a new directory under data/processed/")
        VLBatchProcessor(output=output).run(**options)


if __name__ == "__main__":
    main()
