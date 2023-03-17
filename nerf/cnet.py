import pdb
from src.controlnet.share import *
from src.controlnet import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from src.controlnet.annotator.util import resize_image, HWC3
from src.controlnet.annotator.midas import MidasDetector
from src.controlnet.cldm.model import create_model, load_state_dict
from src.controlnet.ldm.models.diffusion.ddpm import DDPM

apply_midas = MidasDetector()
depth_model = create_model('./src/controlnet/models/cldm_v15.yaml').cpu()
depth_model.load_state_dict(load_state_dict('./src/controlnet/models/control_sd15_depth.pth', location='cuda'))
normal_model = create_model('./src/controlnet/models/cldm_v15.yaml').cpu()
normal_model.load_state_dict(load_state_dict('./src/controlnet/models/control_sd15_normal.pth', location='cuda'))

def controlnet(sd_model, prompt, mode='depth', target_map=None, input_image=None,
    n_prompt="", image_resolution=512, detect_resolution=384,
    ddim_steps=70, guidance_scale=7.5, seed=-1, eta=0, bg_threshold=.4):
    if target_map is not None:
        target_map = (target_map.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
    if mode == 'depth':
        model = depth_model.cuda()
        if target_map is None:
            input_image = HWC3(input_image)
            target_map, _ = apply_midas(resize_image(input_image, detect_resolution))
    elif mode == 'normal':
        model = normal_model.cuda()
        if target_map is None:
            input_image = HWC3(input_image)
            _, target_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
    target_map = HWC3(target_map)

    H = W = image_resolution

    target_map = cv2.resize(target_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(target_map.copy()).float().cuda() / 255.0
    control = control.unsqueeze(0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])]}
    un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt])]}
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    self = sd_model
    t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
    pdb.set_trace()
    latents = self.encode_imgs(input_image)
    with torch.no_grad():
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred
        
    samples, intermediates = sampler.sample(ddim_steps, 1,
                    shape, cond, verbose=False, eta=eta,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    return x_samples[0]
