"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ldm.models.diffusion.sampling_util import renorm_thresholding, norm_thresholding, spatial_norm_thresholding
import ipdb
from PIL import Image
import os
# from video_script import consistency_loss
from RoMa_functions.SED import consistency_loss
from RoMa_functions.utils_RoMa import draw_matches
import matplotlib.pyplot as plt
import sys
import random
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def to(self, device):
        """Same as to in torch module
        Don't really underestand why this isn't a module in the first place"""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_v = getattr(self, k).to(device)
                setattr(self, k, new_v)


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    #@torch.no_grad() # TODO no_grad inactivated
    def sample(self,
               S,
               batch_size,
               shape,
               input_dict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None, return_second_feature=False,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        # print("HELLO IN DDIM(sample)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("Num ddim steps standard: ", S)
        if 'ddim_steps' in input_dict.keys():
            ddim_steps=input_dict['ddim_steps']
        else:
            ddim_steps = S
        # print("Num ddim steps changed to: ", ddim_steps)
        self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        outs = self.ddim_sampling(conditioning, size,
                                                    input_dict=input_dict,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    return_second_feature=return_second_feature
                                                    )
        return outs #samples, intermediates

    #@torch.no_grad() # TODO no_grad inactivated
    def ddim_sampling(self, cond, shape, input_dict=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      t_start=-1, return_second_feature=False):
        device = self.model.betas.device
        b = shape[0]
        # If input_dict has key 'z_T' then extract z_T from input_dict
        if 'i_opt' in input_dict.keys():
            i_opt = input_dict['i_opt']
            print(f"HELLO IN ddim_sampling i_opt = {i_opt}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print(f"HELLO IN ddim_sampling!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Correct z_T shape in ddim_sampling: ", shape)
        if 'z_T' in input_dict.keys():
            z_T = input_dict['z_T']
            print(f"Using provided z_T with shape {z_T.shape} and norm {torch.norm(z_T).item()}")
            img = z_T
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        timesteps = timesteps[:t_start]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        #print(f"Running DDIM Sampling with {total_steps} timesteps")

        #iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        #for i, step in enumerate(iterator):
        if input_dict['bUse_ug'] == True:
            print("Performing Universal guidance (ddim_sampling)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        diffusion_counter = 0
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                print("Hello in mask is not None in ddim_sampling!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            if input_dict['bUse_ug'] == True:
                z_tmin1, z0, good_samples = self.p_sample_ddim_ug(img, cond, ts, input_dict = input_dict, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          dynamic_threshold=dynamic_threshold,
                                          return_second_feature=return_second_feature,
                                          diffusion_counter=diffusion_counter
                                          )
                outs = (z_tmin1, z0)
                if good_samples is not None:
                    input_dict['good_samples'] = good_samples
                diffusion_counter += 1
            else:
                outs = self.p_sample_ddim(img, cond, ts, index=index,
                                          use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          dynamic_threshold=dynamic_threshold,
                                          return_second_feature=return_second_feature
                                          )
            if return_second_feature:
                img, pred_x0, second_feature = outs
            else:
                img, pred_x0 = outs
            if callback:
                img = callback(i, img, pred_x0)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        if return_second_feature:
            return img, intermediates, second_feature

        return img, intermediates # eq 12 in ddim, returns x0 not the noise

    #@torch.no_grad() # TODO no_grad inactivated
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, return_second_feature=False):
        b, *_, device = *x.shape, x.device
        # print(f"&&&&&&&&&&&&&&&&&&&& HELLO IN DDIM (p_sample_ddim) t = {t.item()} &&&&&&&&&&&&&&&&&&&&")
        c['c_crossattn'][0] = c['c_crossattn'][0].detach()
        c['c_concat'][0] = c['c_concat'][0].detach()
        # print("x norm in p_sample_ddim: ", torch.norm(x))
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # print("no uncond")
            # print("shapes: ", x.shape, c)
            e_t = self.model.apply_model(x, t, c, return_feature=return_second_feature)
            if return_second_feature:
                e_t, second_feature = e_t
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])

            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        # sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sigma_t = 0 # TODO Temporarily deactivated
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t


        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature

        if noise_dropout > 0.: # TODO Temporarily deactivated
            print("Noise dropout activated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if return_second_feature:
            return x_prev, pred_x0, second_feature

        return x_prev, pred_x0  # eq 12 in ddim, returns x0 not the noise

    # p_sample_ddim with saving z0 images
    # def p_sample_ddim(self, x, c, t, index, input_dict=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   dynamic_threshold=None, return_second_feature=False):
    #     #print("HELLO IN DDIM (p_sample_ddim)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     #print("Input dict keys: ", input_dict.keys())
    #     save_path = input_dict['save_path']
    #     i_pose = input_dict['i_pose']
    #     cfg_scale = input_dict['cfg_scale']
    #     z0_folder = save_path + f"/z0/{i_pose}/"
    #     os.makedirs(z0_folder, exist_ok=True)
    #     b, *_, device = *x.shape, x.device
    #     #print("----------------------- cfg_scale --------: ", cfg_scale)
    #     skip_uncond = False
    #     megascenes_version = True
    #     if (unconditional_conditioning is None or cfg_scale == 1.) and skip_uncond:
    #         #print("no uncond case!!!")
    #         #print("shapes: ", x.shape, c)
    #         e_t = self.model.apply_model(x, t, c, return_feature=return_second_feature)
    #         if return_second_feature:
    #             e_t, second_feature = e_t
    #     elif megascenes_version:
    #         #print("cond and uncond case!!!")
    #         x_in = torch.cat([x] * 2)
    #         t_in = torch.cat([t] * 2)
    #         if isinstance(c, dict):
    #             assert isinstance(unconditional_conditioning, dict)
    #             c_in = dict()
    #             for k in c:
    #                 if isinstance(c[k], list):
    #                     c_in[k] = [torch.cat([ unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
    #                 else:
    #                     c_in[k] = torch.cat([
    #                             unconditional_conditioning[k],
    #                             c[k]])
    #         else:
    #             c_in = torch.cat([unconditional_conditioning, c])
    #
    #         e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
    #         e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)
    #
    #
    #     if score_corrector is not None:
    #         print("Hello in score_corrector!!!!!!!!!!!!")
    #         assert self.model.parameterization == "eps"
    #         e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
    #
    #     alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    #     # select parameters corresponding to the currently considered timestep
    #     a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
    #
    #     # current prediction for x_0
    #     pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     if quantize_denoised:
    #         pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
    #
    #     if dynamic_threshold is not None:
    #         pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)
    #     pred_image = self.model.decode_first_stage(pred_x0)
    #
    #     pred_image = torch.clamp(pred_image, -1., 1.)
    #
    #     npout = ((pred_image.permute(0,2,3,1).cpu().numpy()+1)*127.5).astype(np.uint8)
    #     pred_image = npout[0,:,:,:]
    #
    #     pred_image_PIL = Image.fromarray(pred_image)
    #     pred_image_PIL.save(z0_folder + f"{index}.png")
    #     # direction pointing to x_t
    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    #     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    #     if noise_dropout > 0.:
    #         noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #
    #     if return_second_feature:
    #         return x_prev, pred_x0, second_feature
    #
    #     return x_prev, pred_x0 # eq 12 in ddim, returns x0 not the noise

    # def decode_latents(
    #     self,
    #     latents,
    # ):
    #     input_dtype = latents.dtype
    #     image = self.model.decode_first_stage(latents)
    #     print("In Decode Latents Checking Min Max")
    #     print("image min: ", torch.min(image))
    #     print("image max: ", torch.max(image))
    #     image_rescaled = (image * 0.5 + 0.5)
    #     print("image_rescaled min: ", torch.min(image_rescaled))
    #     print("image_rescaled max: ", torch.max(image_rescaled))
    #     image_clamped = image_rescaled.clamp(0, 1)
    #     return image_clamped.to(input_dtype)

    def p_sample_ddim_ug(self, z_t, c, t, input_dict=None,index=None, repeat_noise=False, use_original_steps=False,
                      quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, return_second_feature=False, diffusion_counter=0):
        # print("Input dict keys: ", input_dict.keys())
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ug step {index} with t: {t[0].item()} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("z_t norm in p_sample_ddim_ug: ", torch.norm(z_t))

        i_view = input_dict['i_view']
        save_path = input_dict['output_folder']
        cfg_scale = input_dict['cfg_scale']
        lr_ug = input_dict['lr_ug']
        ug_start_idx = input_dict['ug_start_idx']
        cond_img = input_dict['cond_img']
        F_i = input_dict['F_i']
        num_matches = input_dict['num_matches']
        seed = input_dict['seed']
        z0_folder = save_path + f"/z0/"
        z0_epi_folder = save_path + f"/z0_epi/"
        z0_keypoints_folder = save_path + f"/z0_keypoints/"
        histogram_folder = save_path + f"/histograms/"

        run = input_dict['wandb']
        run.define_metric(f"ug/{i_view}/ug_step", "ug_step")
        run.define_metric(f"ug/{i_view}/*", step_metric=f"ug/{i_view}/ug_step")



        os.makedirs(z0_folder, exist_ok=True)
        os.makedirs(z0_epi_folder, exist_ok=True)
        os.makedirs(z0_keypoints_folder, exist_ok=True)

        os.makedirs(histogram_folder, exist_ok=True)
        b, *_, device = *z_t.shape, z_t.device
        # print("----------------------- cfg_scale --------: ", cfg_scale)
        if index <= ug_start_idx:
            num_self_recurrence_steps = 1
        else:
            num_self_recurrence_steps = 1

        skip_uncond = False
        megascenes_version = True

        for k_self in range(num_self_recurrence_steps):

            z_t = z_t.detach()
            z_t.requires_grad = True
            # print("c keys: ", c.keys())
            c['c_crossattn'][0] = c['c_crossattn'][0].detach()
            c['c_concat'][0] = c['c_concat'][0].detach()

            if (unconditional_conditioning is None or cfg_scale == 1.) and skip_uncond:
                # print("no uncond case!!!")
                # print("shapes: ", z.shape, c)
                noise_pred = self.model.apply_model(z_t, t, c, return_feature=return_second_feature)
                if return_second_feature:
                    noise_pred, second_feature = noise_pred
            elif megascenes_version:
                # print("cond and uncond case!!!")
                z_in = torch.cat([z_t] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
                else:
                    c_in = torch.cat([unconditional_conditioning, c])

                noise_pred_uncond, noise_pred_cond = self.model.apply_model(z_in, t_in, c_in).chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            if score_corrector is not None:
                print("Hello in score_corrector!!!!!!!!!!!!")
                assert self.model.parameterization == "eps"
                noise_pred = score_corrector.modify_score(self.model, noise_pred, z_t, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            #sigma_t = 0
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            z0 = (z_t - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
            if quantize_denoised:
                z0, _, *_ = self.model.first_stage_model.quantize(z0)

            if dynamic_threshold is not None:
                pred_z0 = norm_thresholding(z0, dynamic_threshold)
            x_0 = self.model.decode_first_stage(z0)
            # x_0 = self.decode_latents(z0)
            print("Clamping investigation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("x0 max: ", torch.max(x_0))
            print("x0 min: ", torch.min(x_0))
            x_0 = torch.clamp(x_0, -1., 1.)
            npout = ((x_0.permute(0, 2, 3, 1).cpu().detach().numpy() + 1) * 127.5).astype(np.uint8)
            print("npout shape: ", npout.shape)
            if npout.shape[0] == 2:
                extract_index = 1
            elif npout.shape[0] == 1:
                extract_index = 0
            else:
                # Throw error since npout.shape[0] should be 1 or 2
                raise ValueError("Error: npout.shape[0] should be 1 or 2")

            x_0_image = npout[extract_index, :, :, :]
            x_0_image_PIL = Image.fromarray(x_0_image)
            x_0_image_PIL.save(z0_folder + f"{index}_{k_self}.png")
            if index > ug_start_idx:
                input_dict["good_samples"] = None
            # Compute epipolar loss
            print("x_0[extract_index] shape: ", x_0[extract_index].shape)
            loss, output_dict_consistency_loss = consistency_loss(input_dict, F_i, cond_img,
                                                                  x_0[extract_index])
            loss_consistency = output_dict_consistency_loss['loss_consistency']
            print("loss_consistency: ", loss_consistency)
            if loss_consistency is not None:
                print("Consistency loss is not None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                SEDs = output_dict_consistency_loss['SEDs']
                loss_rgb = output_dict_consistency_loss['loss_rgb']
                percentage_within_5 = round(torch.sum(SEDs < 5).item() / SEDs.shape[0],3)
                percentage_within_2 = round(torch.sum(SEDs < 2).item() / SEDs.shape[0],3)
                percentage_within_1 = round(torch.sum(SEDs < 1).item() / SEDs.shape[0],3)

                log_dict = {f'ug/{i_view}/ug_step': diffusion_counter,
                            f'ug/{i_view}/Percentage of SEDs within 5': percentage_within_5,
                            f'ug/{i_view}/Percentage of SEDs within 2': percentage_within_2,
                            f'ug/{i_view}/Percentage of SEDs within 1': percentage_within_1,
                            f'ug/{i_view}/Consistency loss': loss_consistency.item(), f'ug/{i_view}/RGB loss': loss_rgb.item()}

                run.log(log_dict)

                print(f"Percentage of SEDs within 5/2/1: {percentage_within_5}/{percentage_within_2}/{percentage_within_1}")
                # print("Percentage of SEDs within 2: ", percentage_within_2)
                # print("Percentage of SEDs within 1: ", percentage_within_1)
                # print("Consistency loss: ", loss_consistency.item())
                # Print consistency loss rounded to 3 decimal places
                print(f"Consistency loss: {round(loss_consistency.item(),3)}")

                # Visualize epipolar results
                good_samples = output_dict_consistency_loss['good_samples']
                SEDs = output_dict_consistency_loss['SEDs']
                e_line_im_1 = output_dict_consistency_loss['e_line_im_1']
                e_line_im_2 = output_dict_consistency_loss['e_line_im_2']
                closest_point_im1 = output_dict_consistency_loss['closest_point_im1']
                closest_point_im2 = output_dict_consistency_loss['closest_point_im2']
                kpts1 = output_dict_consistency_loss['kpts1']
                kpts2 = output_dict_consistency_loss['kpts2']

                SEDs_np = SEDs.cpu().detach().numpy()
                plt.clf()
                plt.hist(SEDs_np, bins=100)
                plt.title("SEDs Histogram")
                plt.xlabel("SEDs")
                plt.ylabel("Frequency")
                plt.xlim(0, 40)
                plt.ylim(0, int(0.3 * num_matches))
                plt.savefig(histogram_folder + f"/{index}_{k_self}.png")

                refimg_path = save_path + '/reference.png'
                z0_img_path = z0_folder + f"{index}_{k_self}.png"
                num_matches_show = 5
                print("refimg_path: ", refimg_path)
                print("z0_img_path: ", z0_img_path)
                random.seed(0)
                epi_both_img_no_matches = draw_matches(kpts1, kpts2, refimg_path,z0_img_path ,
                                                       num_matches_show, draw_lines=False, random_sampling=True,
                                                       closest_point_im1=closest_point_im1,
                                                       e_line_im_1=e_line_im_1, closest_point_im2=closest_point_im2,
                                                       e_line_im_2=e_line_im_2, SEDs=SEDs)
                random.seed(seed)

                epi_both_img_no_matches.save(z0_epi_folder + f"{index}_{k}.png")

                num_matches_all = kpts1.shape[0]
                keypoints_img = draw_matches(kpts1, kpts2, refimg_path, z0_img_path,
                                             num_matches_all, draw_lines=False, random_sampling=False)
                keypoints_img.save(z0_keypoints_folder + f"/{index}_{k}.png")


                if index <= ug_start_idx:
                    loss.backward()
                    # print("z_t.grad norm: ", torch.norm(z_t.grad))
                    print("lr_ug: ", lr_ug)
                    grad_update = lr_ug * torch.squeeze(z_t.grad, 0)
                    noise_pred = noise_pred + grad_update
                    # noise_pred = noise_pred - grad_update  # Flipped


            # DDIM sampling
            # Update z_t with the gradients
            # direction pointing to z_t
            dir_zt = (1. - a_prev - sigma_t ** 2).sqrt() * noise_pred
            noise = sigma_t * noise_like(z_t.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            z_tmin1 = a_prev.sqrt() * z0 + dir_zt + noise

            # Self-recurrence
            epsilon_prime = torch.randn_like(z_tmin1)
            z_t = torch.sqrt(a_t/a_prev) * z_tmin1 + torch.sqrt(1 - a_t/a_prev) * epsilon_prime

            if return_second_feature:
                return z_tmin1, z0, second_feature
            # # DDPM sampling
            # z_tmin1 = self.scheduler.step(noise_pred, t, z_t, eta=ddim_eta)[
            #     "prev_sample"
            # ]

        return z_tmin1, z0, good_samples  # eq 12 in ddim, returns x0 not the noise

    #@torch.no_grad() # TODO no_grad inactivated
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    #@torch.no_grad() # TODO no_grad inactivated
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    #@torch.no_grad() # TODO no_grad inactivated
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec