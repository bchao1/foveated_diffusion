from .base_pipeline import BasePipeline
import torch


def _create_random_foveation_mask(h: int, w: int, device, dtype=torch.float32, center_range=(-0.3, 0.3), r_range=(0.2, 0.5)):
    """
    Create a random circular foveation mask for training.
    Mask shape [h, w], 1=high-res, 0=low-res.
    center in normalized coords (-0.5, 0.5), r relative to half-diagonal.
    """
    cx = (torch.rand(1, device=device).item() * (center_range[1] - center_range[0]) + center_range[0] + 0.5) * w
    cy = (torch.rand(1, device=device).item() * (center_range[1] - center_range[0]) + center_range[0] + 0.5) * h
    r = torch.rand(1, device=device).item() * (r_range[1] - r_range[0]) + r_range[0]
    diagonal = (h**2 + w**2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist_sq <= radius_px**2).to(device)


def _create_fixed_foveation_mask(h: int, w: int, device, dtype=torch.float32, center=(0.0, 0.0), r=0.5):
    """
    Create a fixed circular foveation mask for training.
    Same geometry as _create_random_foveation_mask but center and radius are fixed.
    center in normalized coords (-0.5, 0.5), r relative to half-diagonal.
    """
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    diagonal = (h**2 + w**2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist_sq <= radius_px**2).to(device)


def _resolve_foveation_mask(pipe: BasePipeline, inputs: dict, h: int, w: int):
    """
    Resolve foveation mask from foveated_training_mode.
    - "fixed": fixed circular mask (center=0, r=0.5).
    - "random": random circular mask.
    - "saliency": use inputs["foveation_mask"], move to pipe device/dtype, ensure shape (h, w).
    - "bbox": same as "saliency" but typically with masks created from bounding boxes (e.g. via run_bbox_detection.py).
    """
    mode = inputs.get("foveated_training_mode", "random")
    if mode == "fixed":
        return _create_fixed_foveation_mask(h, w, pipe.device, pipe.torch_dtype, center=(0.0, 0.0), r=0.5)
    if mode == "random":
        return _create_random_foveation_mask(h, w, pipe.device, pipe.torch_dtype)
    if mode in ("saliency", "bbox"):
        mask = inputs.get("foveation_mask")
        if mask is None:
            raise ValueError(
                f"foveated_training_mode='{mode}' but inputs['foveation_mask'] is None "
                "(e.g. no input_image or missing precomputed mask)."
            )
        mask = mask.to(device=pipe.device, dtype=pipe.torch_dtype)
        while mask.dim() > 2:
            mask = mask.squeeze(0)
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0).squeeze(0)
        return mask
    return _create_random_foveation_mask(h, w, pipe.device, pipe.torch_dtype)


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

def FoveatedFlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    """
    Foveated Flow Matching (FFM) training objective.
    Combines Flow Matching with foveation: for each step, a random foveation mask
    is generated and used for token upsampling/downsampling. The final noise
    prediction is high-resolution (same shape as training target).
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    # Use foveated forward if pipe supports it and has token AE
    has_foveated_forward = hasattr(pipe, "foveated_training_forward") and callable(
        getattr(pipe, "foveated_training_forward", None)
    )
    has_token_ae = (
        getattr(pipe, "token_upsampler", None) is not None
        and getattr(pipe, "token_downsampler", None) is not None
    )

    if has_foveated_forward and has_token_ae:
        height = inputs["height"]
        width = inputs["width"]
        h, w = height // 16, width // 16
        foveation_mask = _resolve_foveation_mask(pipe, inputs, h, w)
        inputs["foveation_mask"] = foveation_mask
        prediction_type = inputs.get("prediction_type", "clean")
        noise_pred = pipe.foveated_training_forward(inputs, timestep, timestep_id, prediction_type)
    else:
        # Fallback to standard Flow Matching
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FoveatedFlowMatchNoTokenAESFTLoss(pipe: BasePipeline, **inputs):
    """
    Foveated Flow Matching (FFM) training objective.
    Combines Flow Matching with foveation: for each step, a random foveation mask
    is generated and used for token upsampling/downsampling. The final noise
    prediction is high-resolution (same shape as training target).
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    # downsampled_noise
    noise_downsampled = torch.randn_like(inputs["input_latents_downsampled"])
    inputs["latents_downsampled"] = pipe.scheduler.add_noise(inputs["input_latents_downsampled"], noise_downsampled, timestep)
    training_target_downsampled = pipe.scheduler.training_target(inputs["input_latents_downsampled"], noise_downsampled, timestep)

    # Use foveated forward if pipe supports it and has token AE
    has_foveated_forward = hasattr(pipe, "foveated_training_forward") and callable(
        getattr(pipe, "foveated_training_forward", None)
    )

    if has_foveated_forward:
        height = inputs["height"]
        width = inputs["width"]
        batch_size, _, channels = inputs["latents"].shape
        h, w = height // 16, width // 16
        foveation_mask = _resolve_foveation_mask(pipe, inputs, h, w)

        # contruct merged training target
        h_2, w_2 = h // 2, w // 2
        training_target_merged = training_target.view(batch_size, h, w, channels)
        training_target_downsampled = training_target_downsampled.view(batch_size, h_2, w_2, channels)
        training_target_merged = training_target_merged.view(batch_size, h_2, 2, w_2, 2, channels)
        training_target_merged = training_target_merged.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, h_2, w_2, 4, channels)

        mask_blocks = foveation_mask.view(h_2, 2, w_2, 2).permute(0, 2, 1, 3).reshape(h_2, w_2, 4)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        training_target_merged[:, is_low_res_block, 0, :] = training_target_downsampled[:, is_low_res_block, :].to(training_target_merged.dtype)

        valid_tokens = torch.ones(h_2, w_2, 4, device=training_target_merged.device, dtype=torch.bool)
        valid_tokens[is_low_res_block, 1:] = False

        training_target_merged = training_target_merged.view(batch_size, -1, channels)
        valid_tokens = valid_tokens.view(-1)
        
        training_target_merged = training_target_merged[:, valid_tokens, :]
        training_target = training_target_merged
        # end of contruct merged training target


        inputs["foveation_mask"] = foveation_mask
        prediction_type = inputs.get("prediction_type", "clean")
        tokenAE_training = inputs.get("tokenAE_training", False)
        noise_pred = pipe.foveated_training_forward(inputs, timestep, timestep_id, prediction_type, tokenAE_training)
    else:
        # Fallback to standard Flow Matching
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    #print('noise_pred shape', noise_pred.shape, 'training_target_merged shape', training_target_merged.shape)
    #exit()

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

def FoveatedDistillLoss(pipe: BasePipeline, **inputs):
    """
    Foveated Distillation training objective.
    Combines Flow Matching with foveation: for each step, a random foveation mask
    is generated and used for token upsampling/downsampling. The final noise
    prediction is high-resolution (same shape as training target). Distill the token AE weights to the main model.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    # Use foveated forward if pipe supports it and has token AE
    has_foveated_forward = hasattr(pipe, "foveated_training_forward") and callable(
        getattr(pipe, "foveated_training_forward", None)
    )
    has_token_ae = (
        getattr(pipe, "token_upsampler", None) is not None
        and getattr(pipe, "token_downsampler", None) is not None
    )

    if has_foveated_forward and has_token_ae:
        height = inputs["height"]
        width = inputs["width"]
        h, w = height // 16, width // 16
        foveation_mask = _resolve_foveation_mask(pipe, inputs, h, w)
        prediction_type = inputs.get("prediction_type", "clean")

        # unfoveated training target
        inputs["foveation_mask"] = None
        training_target = pipe.foveated_training_forward(inputs, timestep, timestep_id, prediction_type)

        # foveated prediction
        inputs["foveation_mask"] = foveation_mask
        noise_pred = pipe.foveated_training_forward(inputs, timestep, timestep_id, prediction_type)
    else:
        # Fallback to standard Flow Matching
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
