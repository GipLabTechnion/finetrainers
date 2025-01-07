import math
import random
from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as TT
from torchvision.transforms.functional import resize, crop
from torchvision.transforms import InterpolationMode


def load_video(
    filename: Union[str, Path],
    height: int,
    width: int,
    max_num_frames: int,
    skip_frames_start: int = 0,
    skip_frames_end: int = 0,
    video_reshape_mode: str = "center",
) -> torch.Tensor:
    """
    Loads a single video from `filename` using decord, applies normalization, 
    handles skipping & truncating frames, ensures the number of frames is (4k + 1),
    and resizes/crops it to [F, C, H, W].

    Args:
        filename (str or Path):
            The path to the video file.
        height (int):
            The desired output height of the frames (after cropping).
        width (int):
            The desired output width of the frames (after cropping).
        max_num_frames (int):
            The maximum number of frames to keep (before adjusting to (4k + 1)).
        skip_frames_start (int, optional):
            Number of frames to skip from the start. Defaults to 0.
        skip_frames_end (int, optional):
            Number of frames to skip from the end. Defaults to 0.
        video_reshape_mode (str, optional):
            Crop/reshape mode. One of ["center", "random", "none"]. Defaults to "center".

    Returns:
        torch.Tensor:
            A tensor of shape [F, C, H, W], where F = final number of frames, C=3 for RGB.
    """
    try:
        import decord
    except ImportError:
        raise ImportError(
            "The `decord` package is required for loading the video. "
            "Install with `pip install decord`."
        )

    # Use the PyTorch bridge so that we get torch Tensors directly
    decord.bridge.set_bridge("torch")

    filename = Path(filename)
    video_reader = decord.VideoReader(uri=filename.as_posix())
    video_num_frames = len(video_reader)

    # Figure out valid frame range
    start_frame = min(skip_frames_start, video_num_frames)
    end_frame = max(0, video_num_frames - skip_frames_end)

    # Decide which frames to load
    if end_frame <= start_frame:
        # If we skip more than total frames, just get the last valid frame
        frames = video_reader.get_batch([start_frame])
    elif (end_frame - start_frame) <= max_num_frames:
        # If total frames in [start_frame, end_frame) <= max_num_frames, load all
        frames = video_reader.get_batch(list(range(start_frame, end_frame)))
    else:
        # Subsample frames to not exceed max_num_frames
        stride = (end_frame - start_frame) // max_num_frames
        indices = list(range(start_frame, end_frame, stride))
        frames = video_reader.get_batch(indices)

    # Truncate strictly to max_num_frames if we overshoot
    frames = frames[:max_num_frames]
    selected_num_frames = frames.shape[0]

    # Ensure number of frames is (4k + 1).
    remainder = (3 + (selected_num_frames % 4)) % 4  # e.g. F=13 => remainder=0, F=14 => remainder=3, etc.
    if remainder != 0:
        frames = frames[:-remainder]
    selected_num_frames = frames.shape[0]
    assert (selected_num_frames - 1) % 4 == 0, (
        f"After trimming, the total number of frames must be (4k + 1). "
        f"Got {selected_num_frames} frames."
    )

    # Normalize from [0..255] to [-1..1]
    frames = (frames - 127.5) / 127.5  # [F, H, W, C]

    # Convert [F, H, W, C] -> [F, C, H, W]
    frames = frames.permute(0, 3, 1, 2)

    # Resize & (center or random) crop
    frames = _resize_for_rectangle_crop(
        frames,
        target_height=height,
        target_width=width,
        video_reshape_mode=video_reshape_mode,
    )

    # Make memory contiguous for downstream usage
    frames = frames.contiguous()  # shape [F, C, H, W]
    return frames


def _resize_for_rectangle_crop(
    frames: torch.Tensor,
    target_height: int,
    target_width: int,
    video_reshape_mode: str = "center",
) -> torch.Tensor:
    """
    Resizes the frames so that the shorter dimension matches the target dimension
    (maintaining aspect ratio) and then crops to (target_height, target_width).

    frames.shape = [F, C, H, W].
    """
    # 1. Resize maintaining aspect ratio, so that the final dimension is at least (target_height x target_width)
    _, _, in_h, in_w = frames.shape
    aspect_in = in_w / in_h
    aspect_out = target_width / target_height

    # We'll figure out the new size so that we can crop
    if aspect_in > aspect_out:
        # The video is "wider" than target -> match height exactly
        new_h = target_height
        new_w = int(in_w * (target_height / in_h))
    else:
        # The video is "narrower" or equal -> match width exactly
        new_w = target_width
        new_h = int(in_h * (target_width / in_w))

    frames = resize(
        frames,
        size=[new_h, new_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    )

    # 2. Now we crop to exactly (target_height, target_width)
    delta_h = new_h - target_height
    delta_w = new_w - target_width

    if video_reshape_mode == "random":
        top = random.randint(0, delta_h) if delta_h > 0 else 0
        left = random.randint(0, delta_w) if delta_w > 0 else 0
    elif video_reshape_mode == "none":
        # "none" might just do a top-left crop (or no crop if exact match)
        top = 0
        left = 0
    else:  # default to "center"
        top = delta_h // 2 if delta_h > 0 else 0
        left = delta_w // 2 if delta_w > 0 else 0

    frames = crop(frames, top=top, left=left, height=target_height, width=target_width)
    return frames
