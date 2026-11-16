import argparse
from bisect import bisect_right
from collections import OrderedDict
import json
from operator import index as integer_index
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data/raw/LeVERB-Bench-Dataset"
MANIFEST = PROJECT / "data/processed/vl_batch/manifest.json"
CAMERAS = ("fpv_cam", "tpv_cam")
TIMESTAMP_TOLERANCE = 1e-4


class VLDataset(Dataset):
    def __init__(self, split="train", manifest=MANIFEST, raw=RAW, cache_size=8):
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split: {split}")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.split = split
        self.manifest = Path(manifest)
        self.raw = Path(raw)
        self.cache_size = cache_size
        self._cache = OrderedDict()
        metadata = json.loads(self.manifest.read_text())
        if metadata["format"] != "leverb_vl_batch_v1":
            raise ValueError("Unsupported batch manifest")
        self.horizon = metadata["horizon"]
        section = metadata["splits"][split]
        self.episodes = []
        self.ends = []
        count = 0
        for episode in section["episodes"]:
            if episode["samples"] == 0:
                continue
            if episode["samples"] < 0 or not episode["arrays"]:
                raise ValueError(f"Invalid sample entry for episode {episode['episode']}")
            self.episodes.append(episode)
            count += episode["samples"]
            self.ends.append(count)
        if count == 0 or count != section["samples"]:
            raise ValueError(f"Invalid sample count for {split}: {count}")
        self.sample_count = count

    def __len__(self):
        return self.sample_count

    def _load_arrays(self, episode):
        path = self.manifest.parent / episode["arrays"]
        if path not in self._cache:
            n = episode["samples"]
            shapes = {
                "current_frame": (n,),
                "timestamp": (n,),
                "task_index": (n,),
                "current_state": (n, 38),
                "future_motion": (n, self.horizon, 45),
            }
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in shapes}
            for key, shape in shapes.items():
                if arrays[key].shape != shape:
                    raise ValueError(f"{path}: {key} has shape {arrays[key].shape}, expected {shape}")
            self._cache[path] = arrays
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(path)
        return self._cache[path]

    @staticmethod
    def _timestamp(frame, path):
        if frame.pts is None or frame.time_base is None:
            raise ValueError(f"Frame is missing a timestamp: {path}")
        return float(frame.pts * frame.time_base)

    @staticmethod
    def _rgb_tensor(frame):
        image = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image))

    def _read_frame(self, path, frame_index, timestamp):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            try:
                container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
                for frame in container.decode(stream):
                    difference = self._timestamp(frame, path) - timestamp
                    if abs(difference) <= TIMESTAMP_TOLERANCE:
                        return self._rgb_tensor(frame)
                    if difference > TIMESTAMP_TOLERANCE:
                        break
            except av.FFmpegError:
                pass

        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_index:
                    if abs(self._timestamp(frame, path) - timestamp) > TIMESTAMP_TOLERANCE:
                        raise ValueError(f"Frame {frame_index} and state timestamp disagree: {path}")
                    return self._rgb_tensor(frame)
        raise IndexError(f"Frame {frame_index} not found: {path}")

    def __getitem__(self, index):
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.ends, index)
        offset = index - (self.ends[episode_index - 1] if episode_index else 0)
        episode = self.episodes[episode_index]
        arrays = self._load_arrays(episode)
        frame = int(arrays["current_frame"][offset])
        timestamp = float(arrays["timestamp"][offset])
        if not 0 <= frame < episode["source_frames"] or not np.isfinite(timestamp):
            raise ValueError(f"Invalid frame index or timestamp for episode {episode['episode']}")
        task = str(int(arrays["task_index"][offset]))
        images = {
            f"{camera}_rgb": self._read_frame(self.raw / episode["videos"][camera], frame, timestamp)
            for camera in CAMERAS
        }
        return {
            **images,
            "instruction": episode["instructions"][task],
            "current_state": torch.from_numpy(arrays["current_state"][offset].copy()),
            "future_motion": torch.from_numpy(arrays["future_motion"][offset].copy()),
            "episode_index": episode["episode"],
            "frame_index": frame,
        }


class VLDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0, shuffle=None, pin_memory=False):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(dataset.split == "train") if shuffle is None else shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


def main():
    parser = argparse.ArgumentParser(description="Read one LeVERB-VL sample and assemble one batch.")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    dataset = VLDataset(split=args.split)
    sample = dataset[0]
    print(f"{args.split}: {len(dataset)} samples")
    print(f"Episode {sample['episode_index']}, frame {sample['frame_index']}: {sample['instruction']}")
    loader = VLDataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    batch = next(iter(loader))
    for key in ("fpv_cam_rgb", "tpv_cam_rgb", "current_state", "future_motion"):
        print(f"{key}: sample {tuple(sample[key].shape)}, batch {tuple(batch[key].shape)}, {batch[key].dtype}")
    print(f"Instructions: {batch['instruction']}")
    print("PASS: sample read and batch assembled.")


if __name__ == "__main__":
    main()
