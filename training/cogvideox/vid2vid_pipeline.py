from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import PIL

from diffusers.utils import logging, replace_example_docstring
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers import CogVideoXImageToVideoPipeline
import inspect
from diffusers.utils import BaseOutput
from dataclasses import dataclass


@dataclass
class CogVideoXPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


# This docstring now has an **empty** Examples: block.
VIDEO2VIDEO_DOC_STRING = """
Examples:
"""


class CogVideoXVideoToVideoPipeline(CogVideoXImageToVideoPipeline):
    r"""
    Pipeline for **video-to-video** generation using CogVideoX.

    This class **inherits** from [`CogVideoXImageToVideoPipeline`] and overrides only the parts that differ.  
    Use it when you have an input video you want to transform into another video, optionally guided by a text prompt.
    """

    @torch.no_grad()
    @replace_example_docstring(VIDEO2VIDEO_DOC_STRING)
    def __call__(
        self,
        video: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        r"""
        Pipeline for video-to-video generation using CogVideoX.

        Examples:

        Usage:
            ```py
            # Put your actual usage examples here, under "Usage:" instead of "Examples:".
            # For instance:
            # pipe = CogVideoXVideoToVideoPipeline.from_pretrained(...)
            # pipe(...)
            ```

        Args:
            video: ...
            prompt: ...
            negative_prompt: ...
            ... (rest of your docstrings) ...

        Returns:
            CogVideoXPipelineOutput or tuple of frames.
        """

        # 1. Check the inputs
        # self.check_inputs(
        #     image=video,  # reusing parent's "image" logic
        #     prompt=prompt,
        #     height=height,
        #     width=width,
        #     negative_prompt=negative_prompt,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        #     latents=latents,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        # )

        # 2. Some shared setup
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode (text) prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=self._execution_device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare scheduler timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self._execution_device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Preprocess & encode the input video into latents
        if not isinstance(video, torch.Tensor):
            video = self.video_processor.preprocess(video, height=height, width=width)

        video = video.to(device=self._execution_device, dtype=prompt_embeds.dtype)
        batch_size = 1 if isinstance(prompt, str) else (len(prompt) if prompt is not None else 1)
        latent_channels = self.transformer.config.in_channels // 2

        # shape is [B, F, C, H, W]
        video = video.unsqueeze(0) if video.ndim == 4 else video
        video = video.permute(0, 2, 1, 3, 4)

        video_latent_dist = self.vae.encode(video)
        video_latents = video_latent_dist.latent_dist.sample(generator=generator)
        video_latents = video_latents * self.vae.config.scaling_factor
        video_latents = video_latents.permute(0, 2, 1, 3, 4)

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        shape = (
            batch_size * num_videos_per_prompt,
            num_frames,
            latent_channels,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self._execution_device, dtype=prompt_embeds.dtype)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(self._execution_device)

        cond_latents = video_latents

        # 6. Extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Possibly create rotary embeddings
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), self._execution_device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                cond_model_input = cond_latents
                if do_classifier_free_guidance:
                    cond_model_input = torch.cat([cond_model_input, cond_model_input], dim=0)

                try:
                    latent_model_input = torch.cat([latent_model_input, cond_model_input], dim=2)
                except:
                    raise ValueError(f"Latent and cond latents have different shapes: {latent_model_input.shape} and {cond_model_input.shape}, number of frames: {num_frames}")

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t.expand(latent_model_input.shape[0])
                

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    ofs=torch.tensor([2.0]).to(latent_model_input.device),
                )[0]
                noise_pred = noise_pred.float()

                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_text - noise_pred_uncond)

                if hasattr(self.scheduler, "step") and "return_dict" in self.scheduler.step.__code__.co_varnames:
                    if self.scheduler.__class__.__name__ == "CogVideoXDPMScheduler":
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    else:
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[0]

                latents = latents.to(prompt_embeds.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals().get(k, None)
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 9. decode latents to frames
        if output_type != "latent":
            latents = latents.permute(0, 2, 1, 3, 4)
            latents = latents / self.vae.config.scaling_factor

            frames = self.vae.decode(latents).sample
            frames = self.video_processor.postprocess_video(frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (frames,)

        return CogVideoXPipelineOutput(frames=frames)



def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


