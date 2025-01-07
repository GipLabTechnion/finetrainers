import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        input_video_column: str = "video",
        output_video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.input_video_column = input_video_column
        self.output_video_column = output_video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.input_video_paths,
                self.output_video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.input_video_paths,
                self.output_video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.input_video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and input videos to be the same but found {len(self.prompts)=} and {len(self.input_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        if len(self.output_video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and output videos to be the same but found {len(self.prompts)=} and {len(self.output_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.video_transforms_concat = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], inplace=True),
            ]
        )


    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.input_video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            input_video_latents, output_video_latents, prompt_embeds = self._preprocess_video(self.input_video_paths[index], self.output_video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = input_video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = input_video_latents.size(2) * 8
            width = input_video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "input_video": input_video_latents,
                "output_video": output_video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            input_video, output_video, _ = self._preprocess_video(self.input_video_paths[index], self.output_video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "input_video": input_video,
                "output_video": output_video,
                "video_metadata": {
                    "num_frames": input_video.shape[0],
                    "height": input_video.shape[2],
                    "width": input_video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError(f"Root folder for videos does not exist at {self.data_root=}")

        prompt_path = self.data_root.joinpath(self.caption_column)
        input_video_path = self.data_root.joinpath(self.input_video_column)
        output_video_path = self.data_root.joinpath(self.output_video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not input_video_path.exists() or not input_video_path.is_file():
            raise ValueError(
                "Expected `--input_video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not output_video_path.exists() or not output_video_path.is_file():
            raise ValueError(
                "Expected `--output_video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(input_video_path, "r", encoding="utf-8") as file:
            input_video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(output_video_path, "r", encoding="utf-8") as file:
            output_video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if not self.load_tensors and any(not path.is_file() for path in input_video_paths):
            raise ValueError(
                f"Expected `{self.input_video_column=}` (input videos) to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )
        
        if not self.load_tensors and any(not path.is_file() for path in output_video_paths):
            raise ValueError(
                f"Expected `{self.output_video_column=}` (output videos) to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        # verify length of all lists are the same
        if len(prompts) != len(input_video_paths) or len(prompts) != len(output_video_paths):
            raise ValueError(
                f"Expected the number of lines in `{self.caption_column=}`, `{self.input_video_column=}` and `{self.output_video_column=}` to be the same but found {len(prompts)=}, {len(input_video_paths)=} and {len(output_video_paths)=}."
            )

        return prompts, input_video_paths, output_video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        input_video_paths = df[self.input_video_column].tolist()
        input_video_paths = [self.data_root.joinpath(line.strip()) for line in input_video_paths]

        output_video_paths = df[self.output_video_column].tolist()
        output_video_paths = [self.data_root.joinpath(line.strip()) for line in output_video_paths]

        if any(not path.is_file() for path in input_video_paths):
            raise ValueError(
                f"Expected `{self.input_video_column=}` (input_videos) to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )
        
        if any(not path.is_file() for path in output_video_paths):
            raise ValueError(
                f"Expected `{self.output_video_column=}` (output_videos) to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, input_video_paths, output_video_paths

    def _preprocess_video(self, input_path: Path, output_path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(input_path, output_path)
        else:
            input_video_reader = decord.VideoReader(uri=input_path.as_posix())
            input_video_num_frames = len(input_video_reader)

            input_indices = list(range(0, input_video_num_frames, input_video_num_frames // self.max_num_frames))
            input_frames = input_video_reader.get_batch(input_indices)
            input_frames = input_frames[: self.max_num_frames].float()
            input_frames = input_frames.permute(0, 3, 1, 2).contiguous()

            output_video_reader = decord.VideoReader(uri=output_path.as_posix())
            output_video_num_frames = len(output_video_reader)

            output_indices = list(range(0, output_video_num_frames, output_video_num_frames // self.max_num_frames))
            output_frames = output_video_reader.get_batch(output_indices)
            output_frames = output_frames[: self.max_num_frames].float()
            output_frames = output_frames.permute(0, 3, 1, 2).contiguous()

            c1 = input_frames.shape[1]
            concat_frames = torch.cat([input_frames, output_frames], dim=1)
            concat_frames = torch.stack([self.video_transforms_concat(frame) for frame in concat_frames], dim=1)
            
            input_frames = concat_frames[:c1]
            output_frames = concat_frames[c1:]

            input_frames = input_frames.permute(1, 0, 2, 3)
            output_frames = output_frames.permute(1, 0, 2, 3)
            
            assert input_frames.shape == output_frames.shape

            return input_frames, output_frames, None


    def _load_preprocessed_latents_and_embeds(self, input_path: Path, output_path: Path) -> Tuple[torch.Tensor, torch.Tensor , torch.Tensor]:
        
        input_filename_without_ext = input_path.name.split(".")[0]
        input_pt_filename = f"{input_filename_without_ext}.pt"

        output_filename_without_ext = output_path.name.split(".")[0]
        output_pt_filename = f"{output_filename_without_ext}.pt"
        
        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt

        input_video_latents_path = input_path.parent.parent.joinpath("input_video_latents")
        output_video_latents_path = output_path.parent.parent.joinpath("output_video_latents")
        embeds_path = input_path.parent.parent.joinpath("prompt_embeds")

        if (
            not input_video_latents_path.exists()
            or not output_video_latents_path.exists()
            or not embeds_path.exists()
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        input_video_latent_filepath = input_video_latents_path.joinpath(input_pt_filename)
        output_video_latent_filepath = output_video_latents_path.joinpath(output_pt_filename)
        embeds_filepath = embeds_path.joinpath(input_pt_filename)

        if not input_video_latent_filepath.is_file() or not output_video_latent_filepath.is_file() or not embeds_filepath.is_file():
            input_video_latent_filepath = input_video_latent_filepath.as_posix()
            output_video_latent_filepath = output_video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {input_video_latent_filepath=} or the file {output_video_latent_filepath=} or the file {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        input_video_latents = torch.load(input_video_latent_filepath, map_location="cpu", weights_only=True)
        output_video_latents = torch.load(output_video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return input_video_latents, output_video_latents, embeds


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, input_path: Path, output_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(input_path, output_path)
        else:

            input_video = decord.VideoReader(uri=input_path.as_posix())
            input_video_num_frames = len(input_video)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(input_video_num_frames, self.max_num_frames))
            )

            input_frame_indices = list(range(0, input_video_num_frames, input_video_num_frames // nearest_frame_bucket))

            input_frames = input_video.get_batch(input_frame_indices)
            input_frames = input_frames[:nearest_frame_bucket].float()
            input_frames = input_frames.permute(0, 3, 1, 2).contiguous()

            input_nearest_res = self._find_nearest_resolution(input_frames.shape[2], input_frames.shape[3])
            input_frames_resized = torch.stack([resize(frame, input_nearest_res) for frame in input_frames], dim=0)

            output_video = decord.VideoReader(uri=output_path.as_posix())
            output_video_num_frames = len(output_video)

            output_frame_indices = list(range(0, output_video_num_frames, output_video_num_frames // nearest_frame_bucket))

            output_frames = output_video.get_batch(output_frame_indices)
            output_frames = output_frames[:nearest_frame_bucket].float()
            output_frames = output_frames.permute(0, 3, 1, 2).contiguous()

            output_nearest_res = self._find_nearest_resolution(output_frames.shape[2], output_frames.shape[3])
            output_frames_resized = torch.stack([resize(frame, output_nearest_res) for frame in output_frames], dim=0)

            c1 = input_frames_resized.shape[1]

            concat_frames = torch.cat([input_frames_resized, output_frames_resized], dim=1)
            concat_frames = torch.stack([self.video_transforms_concat(frame) for frame in concat_frames], dim=1)

            input_frames = concat_frames[:c1]
            output_frames = concat_frames[c1:]

            input_frames = input_frames.permute(1, 0, 2, 3)
            output_frames = output_frames.permute(1, 0, 2, 3)

            assert input_frames.shape == output_frames.shape

            return input_frames, output_frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr1, arr2, image_size):
        """
        Resizes both `arr1` and `arr2` to a rectangle determined by `image_size`, 
        then applies the same center/random crop to both.

        Args:
            arr1, arr2 (torch.Tensor):
                Tensors of shape [F, C, H, W] (e.g. frames x channels x height x width)
                that should be resized/cropped in the same way.
            image_size (tuple):
                A (height, width) tuple to resize/crop to.

        Returns:
            (torch.Tensor, torch.Tensor):
                The two tensors after having the same resize & crop operations applied.
        """
        reshape_mode = self.video_reshape_mode

        # Decide on resized shape from arr1's aspect ratio (arr1 and arr2 are expected to be same shape or same ratio).
        if arr1.shape[3] / arr1.shape[2] > image_size[1] / image_size[0]:
            new_size = [image_size[0], int(arr1.shape[3] * image_size[0] / arr1.shape[2])]
        else:
            new_size = [int(arr1.shape[2] * image_size[1] / arr1.shape[3]), image_size[1]]

        # Resize both arr1 and arr2 to the same new shape
        arr1 = resize(
            arr1,
            size=new_size,
            interpolation=InterpolationMode.BICUBIC,
        )
        arr2 = resize(
            arr2,
            size=new_size,
            interpolation=InterpolationMode.BICUBIC,
        )

        # Squeeze out the first dimension if needed (e.g. if shape was [1, C, H, W])
        h, w = arr1.shape[2], arr1.shape[3]
        arr1 = arr1.squeeze(0)
        arr2 = arr2.squeeze(0)

        # Compute how much room we have to randomly/center crop
        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        # Determine crop offsets
        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top = delta_h // 2
            left = delta_w // 2
        else:
            raise NotImplementedError(f"Unsupported reshape_mode: {reshape_mode}")

        # Crop both arr1 and arr2 using the same offsets
        arr1 = TT.functional.crop(arr1, top=top, left=left, height=image_size[0], width=image_size[1])
        arr2 = TT.functional.crop(arr2, top=top, left=left, height=image_size[0], width=image_size[1])

        return arr1, arr2

    def _preprocess_video(self, input_path: Path, output_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(input_path, output_path)
        else:

            input_video_reader = decord.VideoReader(uri=input_path.as_posix())
            input_video_num_frames = len(input_video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(input_video_num_frames, self.max_num_frames))
            )

            input_frame_indices = list(range(0, input_video_num_frames, input_video_num_frames // nearest_frame_bucket))
            input_frames = input_video_reader.get_batch(input_frame_indices)
            input_frames = input_frames[:nearest_frame_bucket].float()
            input_frames = input_frames.permute(0, 3, 1, 2).contiguous()

            input_nearest_res = self._find_nearest_resolution(input_frames.shape[2], input_frames.shape[3])
            input_frames_resized = self._resize_for_rectangle_crop(input_frames, input_nearest_res)

            output_video_reader = decord.VideoReader(uri=output_path.as_posix())
            output_video_num_frames = len(output_video_reader)

            output_frame_indices = list(range(0, output_video_num_frames, output_video_num_frames // nearest_frame_bucket))
            output_frames = output_video_reader.get_batch(output_frame_indices)
            output_frames = output_frames[:nearest_frame_bucket].float()
            output_frames = output_frames.permute(0, 3, 1, 2).contiguous()

            output_nearest_res = self._find_nearest_resolution(output_frames.shape[2], output_frames.shape[3])
            output_frames_resized = self._resize_for_rectangle_crop(output_frames, output_nearest_res)

            c1 = input_frames_resized.shape[1]
            concat_frames = torch.cat([input_frames_resized, output_frames_resized], dim=1)
            concat_frames = torch.stack([self.video_transforms_concat(frame) for frame in concat_frames], dim=1)

            input_frames = concat_frames[:c1]
            output_frames = concat_frames[c1:]

            input_frames = input_frames.permute(1, 0, 2, 3)
            output_frames = output_frames.permute(1, 0, 2, 3)

            assert input_frames.shape == output_frames.shape

            return input_frames, output_frames, None


    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []