# define the function & import all libraries
# Code is built on top of: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput

XLA_AVAILABLE = False
# ScoreFusion inference pipeline, adapted from the `__call__` function of Diffusers's SDXL inference pipeline.
@torch.no_grad()
def ScoreFusion_inference(
    pipeline: StableDiffusionXLPipeline,
    l1: torch.Tensor,
    l2: torch.Tensor,
    unet1: Optional[UNet2DConditionModel] = None,
    unet2: Optional[UNet2DConditionModel] = None,
    num_images_per_prompt: Optional[int] = 1,
    num_inference_steps: int = 50,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    eta: float = 0.0,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"]
):
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    original_size = (height, width)
    target_size = (height, width)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device

    # 3. Encode input prompt
    lora_scale = ( None )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=pipeline.clip_skip,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta = 0) # eta is only used for DDIMScheduler, so might as well set it to 0

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipeline._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if pipeline.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipeline.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            pipeline.do_classifier_free_guidance,
        )

    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)

    # 8.1 Apply denoising_end
    if (
        pipeline.denoising_end is not None
        and isinstance(pipeline.denoising_end, float)
        and pipeline.denoising_end > 0
        and pipeline.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                pipeline.scheduler.config.num_train_timesteps
                - (pipeline.denoising_end * pipeline.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]
    timestep_cond = None

    pipeline._num_timesteps = len(timesteps)

    with torch.no_grad():
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipeline.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents

                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds


                # predict the noise residual using the weighted average of the two UNet outputs
                noise_pred_1 = unet1(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                noise_pred_2 = unet2(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred_1 * l1 + noise_pred_2 * l2
                # might need to delete noise_pred_1 and noise_pred_2 to free up memory TODO

                # perform guidance
                if pipeline.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if pipeline.do_classifier_free_guidance and pipeline.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipeline.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     step_idx = i // getattr(self.scheduler, "order", 1)
                    #     callback(step_idx, t, latents)

                # clear cuda cache
                torch.cuda.empty_cache()

    # Decode using VAE
    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

        if needs_upcasting:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != pipeline.vae.dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                pipeline.vae = pipeline.vae.to(latents.dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
        has_latents_std = hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / pipeline.vae.config.scaling_factor

        image = pipeline.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            pipeline.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        # apply watermark if available
        if pipeline.watermark is not None:
            image = pipeline.watermark.apply_watermark(image)

        image = pipeline.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    pipeline.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)

# Post-process function (handles final VAE step to lower GPU memory load)
@torch.no_grad()
def postprocess_image(pipeline: StableDiffusionXLPipeline, latents: torch.Tensor):
    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

    if needs_upcasting:
        pipeline.upcast_vae()
        latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipeline.vae.dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            pipeline.vae = pipeline.vae.to(latents.dtype)

    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    has_latents_mean = hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipeline.vae.config.scaling_factor

    image = pipeline.vae.decode(latents, return_dict=False)[0]

    # cast back to fp16 if needed
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float16)

    image = pipeline.image_processor.postprocess(image, output_type='pil')
    pipeline.maybe_free_model_hooks()

    return StableDiffusionXLPipelineOutput(images=image)

# test the inference pipeline
if '__name__' == '__main__':
    positive_prompt = "a photo of a mathematics scientist, looking at the camera, ultra quality, sharp focus"
    negative_prompt = "duplicate, cartoon, anime, painting, low quality"
    w1 = torch.tensor(0.5, device="cuda", dtype=torch.float16)
    w2 = torch.tensor(0.5, device="cuda", dtype=torch.float16)
    SD_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to("cuda")
    u1 = SD_pipeline.unet1; u2 = SD_pipeline.unet2
    generator = torch.Generator(device="cuda"); generator.manual_seed(0)     
    images = ScoreFusion_inference(pipeline=SD_pipeline,
        l1 = w1,
        l2 = w2,
        unet1 = u1,
        unet2 = u2,
        num_images_per_prompt = 4,
        num_inference_steps = 10,
        prompt = positive_prompt,
        negative_prompt = negative_prompt,
        height = 512,
        width = 512,
        generator = generator
    )