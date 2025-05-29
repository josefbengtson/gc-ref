import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings
from warnings import warn
from PIL import Image

# import roma
import sys
from RoMa_functions.RoMa.roma.utils import get_tuple_transform_ops
from RoMa_functions.RoMa.roma.utils.local_correlation import local_correlation
from RoMa_functions.RoMa.roma.utils.utils import cls_to_flow_refine
from RoMa_functions.RoMa.roma.utils.kde import kde

class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, x, y, flow, scale_factor = 1, logits = None):
        b,c,hs,ws = x.shape
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            with torch.no_grad():
                x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                )
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = flow-im_A_coords
                emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)
                if self.local_corr_radius:
                    if self.corr_in_other:
                        # Corr in other means take a kxk grid around the predicted coordinate in other image
                        local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow = flow, 
                                                       sample_mode = self.sample_mode)
                    else:
                        raise NotImplementedError("Local corr in own frame should not be used.")
                    if self.no_im_B_fm:
                        x_hat = torch.zeros_like(x)
                    d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
                else:    
                    d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        with warnings.catch_warnings():
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats

class Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, gps, proj, conv_refiner, detach=False, scales="all", pos_embeddings = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0, displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        
    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    def get_positional_embedding(self, b, h ,w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(self, f1, f2, gt_warp = None, gt_prob = None, upsample = False, flow = None, certainty = None, scale_factor = 1):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.hidden_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        corresps = {}
        print("Upsample in Decoder forward: ", upsample)
        print("Self.detach: ", self.detach)
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                    flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            certainty = F.interpolate(
                    certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                with torch.autocast("cuda", dtype = self.amp_dtype):
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )
                
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)
                    corresps[ins].update({"gm_cls": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                else:
                    corresps[ins].update({"gm_flow": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                    flow = gm_warp_or_cls.detach()
                    
            if new_scale in self.conv_refiner:
                corresps[ins].update({"flow_pre_delta": flow}) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor = scale_factor, logits = certainty,
                )                    
                corresps[ins].update({"delta_flow": delta_flow,}) if self.training else None
                displacement = ins*torch.stack((delta_flow[:, 0].float() / (self.refine_init * w),
                                                delta_flow[:, 1].float() / (self.refine_init * h),),dim=1,)
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,             
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    print("Hello in decoder self.detach section")
                    flow = flow.detach()
                    certainty = certainty.detach()
            #torch.cuda.empty_cache()                
        return corresps


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        symmetric = False,
        name = None,
        attenuate_cert = None,
        recrop_upsample = False,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = (14*16*6, 14*16*6)
        self.symmetric = symmetric
        self.sample_thresh = 0.05
        self.recrop_upsample = recrop_upsample
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res

    def extract_backbone_features(self, batch, batched=True, upsample=False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim=0)
            feature_pyramid = self.encoder(X, upsample=upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample=upsample), self.encoder(x_s, upsample=upsample)
        return feature_pyramid


    def extract_backbone_featuresOLD(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        print("x_q requires_grad: ", x_q.requires_grad)
        print("x_s requires_grad: ", x_s.requires_grad)
        print("x_q shape: ", x_q.shape)
        print("x_s shape: ", x_s.shape)
        print("x_q dtype: ", x_q.dtype)
        print("x_s dtype: ", x_s.dtype)
        if batched:
            print("In extract_backbone_features batched")
            X = torch.cat((x_s, x_q), dim=0) # TODO Test swap (Standard is x_q,x_s)
            # X = torch.stack((x_q.unsqueeze(0), x_s.unsqueeze(0)), dim=0).squeeze(1)
            print("X.shape: ", X.shape)
            feature_pyramid = self.encoder(X, upsample=upsample)
        else:
            print("In extract_backbone_features not batched")
            feature_pyramid = self.encoder(x_q, upsample=upsample), self.encoder(x_s, upsample=upsample)
        return feature_pyramid

    def extract_backbone_features_SD(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        # correct_dtpye = x_q.dtype
        # # Example tensors
        # x_s = torch.randn(1, 3, 560, 560, requires_grad=False, dtype = correct_dtpye)
        # x_q = torch.randn(1, 3, 560, 560, requires_grad=True, dtype = correct_dtpye)

        print("x_q requires_grad: ", x_q.requires_grad)
        print("x_s requires_grad: ", x_s.requires_grad)
        print("x_q shape: ", x_q.shape)
        print("x_s shape: ", x_s.shape)
        print("x_q dtype: ", x_q.dtype)
        print("x_s dtype: ", x_s.dtype)
        if batched:
            print("In extract_backbone_features batched")
            X_cat = torch.cat((x_s, x_q), dim=0) # TODO Test swap (Standard is x_q,x_s)
            # X = X_cat.flip(0)
            X = X_cat
            # temp = X[0, :, :, :].clone()
            # X[0, :, :, :] = X[1, :, :, :]
            # X[1, :, :, :] = temp

            # X = torch.stack((x_s.unsqueeze(0), x_q.unsqueeze(0)), dim=0).squeeze(1)
            # print("X.shape 1: ", X.shape)
            # X = X.squeeze(1)
            print("X.shape 2: ", X.shape)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            print("In extract_backbone_features not batched")
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid, X_cat, X, x_s, x_q

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):

        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid,
                                f_s_pyramid,
                                upsample = upsample,
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)

        return corresps

    def forward_SD(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid, X_cat, X_encoder_in, x_s, x_q = self.extract_backbone_features_SD(batch, batched = batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid

        print("X_encoder_in requires_grad: ", X_encoder_in.requires_grad)
        print("X_encoder_in shape: ", X_encoder_in.shape)
        # if X_encoder_in.requires_grad:
        #     print("X_encoder_in does require_grad")
        #     X_encoder_in.retain_grad()

        print("X_cat requires_grad: ", X_cat.requires_grad)
        print("X_cat shape: ", X_cat.shape)
        # if X_cat.requires_grad:
        #     print("X_cat does require_grad")
        #     X_cat.retain_grad()

        print("x_s requires_grad: ", x_s.requires_grad)
        print("x_s shape: ", x_s.shape)
        # if x_s.requires_grad:
        #     print("x_s does require_grad")
        #     x_s.retain_grad()

        print("x_q requires_grad: ", x_q.requires_grad)
        print("x_q shape: ", x_q.shape)
        # if x_q.requires_grad:
        #     print("x_q does require_grad")
        #     x_q.retain_grad()

        for key in f_q_pyramid.keys():
            print("f_q_pyramid key: ", key)
        print("f_q_pyramid[1] requires_grad: ", f_q_pyramid[1].requires_grad)
        f_q_pyramid_1_test = f_q_pyramid[1]

        # if f_q_pyramid_1_test.requires_grad:
        #     print("f_q_pyramid[1] does require_grad")
        #     f_q_pyramid_1_test.retain_grad()
        # else:
        #     print("f_q_pyramid[1] does not require_grad")


        corresps = self.decoder(f_q_pyramid,
                                f_s_pyramid,
                                upsample=upsample,
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)

        return corresps, f_q_pyramid_1_test, X_cat, X_encoder_in, x_s, x_q

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(f_q_pyramid,
                                f_s_pyramid,
                                upsample=upsample,
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps

    def forward_symmetric_SD(self, batch, batched = True, upsample = False, scale_factor = 1):
        print("---------- Pre extract backbone features ----------")
        print("batched: ", batched)
        print("upsample: ", upsample)
        feature_pyramid, X_encoder_in, x_s = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        print("X_encoder_in requires_grad: ", X_encoder_in.requires_grad)
        print("X_encoder_in shape: ", X_encoder_in.shape)
        # if X_encoder_in.requires_grad:
        #     print("X_encoder_in does require_grad")
        #     X_encoder_in.retain_grad()
        print("x_s requires_grad: ", x_s.requires_grad)
        print("x_s shape: ", x_s.shape)
        # if x_s.requires_grad:
        #     print("x_s does require_grad")
        #     x_s.retain_grad()

        for key in f_q_pyramid.keys():
            print("f_q_pyramid key: ", key)
        print("f_q_pyramid[1] requires_grad: ", f_q_pyramid[1].requires_grad)
        f_q_pyramid_1_test = f_q_pyramid[1]

        # if f_q_pyramid_1_test.requires_grad:
        #     print("f_q_pyramid[1] does require_grad")
        #     f_q_pyramid_1_test.retain_grad()
        # else:
        #     print("f_q_pyramid[1] does not require_grad")


        # print("f_s_pyramid requires_grad: ", f_s_pyramid.requires_grad)
        corresps = self.decoder(f_q_pyramid,
                                f_s_pyramid,
                                upsample = upsample,
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps, f_q_pyramid_1_test, X_encoder_in, x_s
    
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((W_A/2 * (kpts_A[...,0]+1), H_A/2 * (kpts_A[...,1]+1)),axis=-1)
        kpts_B = torch.stack((W_B/2 * (kpts_B[...,0]+1), H_B/2 * (kpts_B[...,1]+1)),axis=-1)
        return kpts_A, kpts_B
    
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)
    
    def get_roi(self, certainty, W, H, thr = 0.025):
        raise NotImplementedError("WIP, disable for now")
        hs,ws = certainty.shape
        certainty = certainty/certainty.sum(dim=(-1,-2))
        cum_certainty_w = certainty.cumsum(dim=-1).sum(dim=-2)
        cum_certainty_h = certainty.cumsum(dim=-2).sum(dim=-1)
        print(cum_certainty_w)
        print(torch.min(torch.nonzero(cum_certainty_w > thr)))
        print(torch.min(torch.nonzero(cum_certainty_w < thr)))
        left = int(W/ws * torch.min(torch.nonzero(cum_certainty_w > thr)))
        right = int(W/ws * torch.max(torch.nonzero(cum_certainty_w < 1 - thr)))
        top = int(H/hs * torch.min(torch.nonzero(cum_certainty_h > thr)))
        bottom = int(H/hs * torch.max(torch.nonzero(cum_certainty_h < 1 - thr)))
        print(left, right, top, bottom)
        return left, top, right, bottom

    def recrop(self, certainty, image_path):
        roi = self.get_roi(certainty, *Image.open(image_path).size)
        return Image.open(image_path).crop(roi)

    # # Trying to run optimization through
    # @torch.inference_mode() # TODO Turned of inference_mode for now
    # def match(
    #     self,
    #     im_A_path,
    #     im_B_path,
    #     *args,
    #     batched=False,
    #     device=None,
    # ):
    #     print("HELLO IN ROMA .match()")
    #     if device is None:
    #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     if isinstance(im_A_path, (str, os.PathLike)):
    #         im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
    #     else:
    #         # Assume its not a path
    #         im_A, im_B = im_A_path, im_B_path
    #     symmetric = self.symmetric
    #     print("Symmetric in match(): ", symmetric)
    #     self.train(False)
    #
    #
    #     # with torch.no_grad(): # TODO Turned of no_grad() for now
    #     if not batched:
    #         b = 1
    #         # w, h = im_A.size
    #         # w2, h2 = im_B.size
    #
    #         # channels, w, h = im_A.shape  # TODO change how extract shape
    #         # channels2, w2, h2 = im_B.shape  # TODO change how extract shape
    #
    #         # Get images in good format
    #         ws = self.w_resized
    #         hs = self.h_resized
    #         print("---Check resizing---")
    #         print("ws in match(): ", ws)
    #         print("hs in match(): ", hs)
    #         print("im_A shape: ", im_A.shape)
    #         print("im_B shape: ", im_B.shape)
    #
    #         test_transform = get_tuple_transform_ops(
    #             resize=(hs, ws), normalize=True, clahe = False
    #         )
    #         im_A, im_B = test_transform((im_A, im_B))
    #         batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
    #     else:
    #         b, c, h, w = im_A.shape
    #         b, c, h2, w2 = im_B.shape
    #         assert w == w2 and h == h2, "For batched images we assume same size"
    #         batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
    #         if h != self.h_resized or self.w_resized != w:
    #             warn("Model resolution and batch resolution differ, may produce unexpected results")
    #         hs, ws = h, w
    #     finest_scale = 1
    #     # Run matcher
    #     if symmetric:
    #         corresps  = self.forward_symmetric(batch)
    #     else:
    #         corresps = self.forward(batch, batched=True)  # TODO batched is usually True
    #
    #
    #     # print("Turn of upsample_preds for now")
    #     # self.upsample_preds = False  # TODO: Turned off for now
    #     print("self.upsample_preds: ", self.upsample_preds)
    #     if self.upsample_preds:
    #         hs, ws = self.upsample_res
    #
    #     if self.attenuate_cert:
    #         low_res_certainty = F.interpolate(
    #         corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
    #         )
    #         cert_clamp = 0
    #         factor = 0.5
    #         low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
    #
    #     print("Right before upsample preds")
    #     print("self.upsample_preds: ", self.upsample_preds)
    #     if self.upsample_preds:
    #         print("IN UPSAMPLE PREDS!!!!")
    #         finest_corresps = corresps[finest_scale]
    #         torch.cuda.empty_cache()
    #         test_transform = get_tuple_transform_ops(
    #             resize=(hs, ws), normalize=True
    #         )
    #         if self.recrop_upsample:
    #             certainty = corresps[finest_scale]["certainty"]
    #             print(certainty.shape)
    #             im_A = self.recrop(certainty[0,0], im_A_path)
    #             im_B = self.recrop(certainty[1,0], im_B_path)
    #             #TODO: need to adjust corresps when doing this (From before)
    #         elif isinstance(im_A_path, (str, os.PathLike)):  # TODO only open if path
    #             im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
    #         im_A, im_B = test_transform((im_A, im_B))
    #         im_A, im_B = im_A[None].to(device), im_B[None].to(device)
    #         scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
    #         batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
    #         print("DTYPE TEST im_B: ", im_B.dtype)
    #
    #         if symmetric:
    #             corresps = self.forward_symmetric(batch, upsample = True, batched=True, scale_factor = scale_factor)
    #         else:
    #             # corresps = self.forward(batch, batched = True, upsample=True, scale_factor = scale_factor)
    #             corresps, f_q_pyramid_1_test, X_encoder_in, x_s, x_q = self.forward_SD(batch, batched=True, upsample=True, scale_factor = scale_factor)
    #             # TODO batched is usually True
    #
    #     im_A_to_im_B = corresps[finest_scale]["flow"]
    #     print("corresps[finest_scale][flow] requires grad : ", corresps[finest_scale]["flow"].requires_grad)
    #     print("DTYPE TEST im_A_to_im_B: ", im_A_to_im_B.dtype)
    #
    #     certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
    #     if finest_scale != 1:
    #         im_A_to_im_B = F.interpolate(
    #         im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
    #         )
    #         certainty = F.interpolate(
    #         certainty, size=(hs, ws), align_corners=False, mode="bilinear"
    #         )
    #     im_A_to_im_B = im_A_to_im_B.permute(
    #         0, 2, 3, 1
    #         )
    #     # Create im_A meshgrid
    #     im_A_coords = torch.meshgrid(
    #         (
    #             torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
    #             torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
    #         )
    #     )
    #     im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    #     im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
    #
    #     certainty = certainty.sigmoid()  # logits -> probs
    #
    #     im_A_coords = im_A_coords.permute(0, 2, 3, 1)
    #     if (im_A_to_im_B.abs() > 1).any() and True:
    #         wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
    #
    #         # certainty[wrong[:,None]] = 0 # TODO Original assignment
    #         # TODO attempt replace inplace assignment
    #         certainty = certainty.clone()
    #         certainty[wrong[:, None]] = 0  # TODO Original assignment
    #
    #     im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
    #     if symmetric:
    #         A_to_B, B_to_A = im_A_to_im_B.chunk(2)
    #         q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
    #         im_B_coords = im_A_coords
    #         s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
    #         warp = torch.cat((q_warp, s_warp),dim=2)
    #         certainty = torch.cat(certainty.chunk(2), dim=3)
    #     else:
    #         print("In warp creation:")
    #         print("im_A_coords shape: ", im_A_coords.shape)
    #         print("im_A_to_im_B shape: ", im_A_to_im_B.shape)
    #         print("im_A_to_im_B requires grad: ", im_A_to_im_B.requires_grad)
    #         warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    #         print("warp in match requires grad: ", warp.requires_grad)
    #         print("warp shape: ", warp.shape)
    #     if batched:
    #         print("HELLO IN BATCHED RETURN")
    #         return (
    #             warp,
    #             certainty[:, 0]
    #         )
    #     else:
    #         print("HELLO IN NOT BATCHED RETURN")
    #         return (
    #             warp[0],
    #             certainty[0, 0],
    #             f_q_pyramid_1_test,
    #             X_encoder_in,
    #             x_s,
    #             x_q,
    #         )


    def match_SD(
        self,
        im_A_path,
        im_B_path,
        *args,
        batched=False,
        device=None,
    ):
        print("HELLO IN ROMA SD .match() Current")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(im_A_path, (str, os.PathLike)):
            im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
        else:
            print("Assuming its not a path")
            # Assume its not a path
            im_A, im_B = im_A_path, im_B_path
        symmetric = self.symmetric
        self.train(False)
        # print("self.train(True)")
        # self.train(True)  # TODO Turned on training for now


        # with torch.no_grad(): # TODO Turned of no_grad() for now
        if not batched:
            print("In not batched beginning")
            b = 1
            # w, h = im_A.size
            # w2, h2 = im_B.size

            # channels, w, h = im_A.shape  # TODO change how extract shape
            # channels2, w2, h2 = im_B.shape  # TODO change how extract shape

            # Get images in good format
            ws = self.w_resized
            hs = self.h_resized

            hs_upsample, ws_upsample = self.upsample_res
            # self.upsample_preds = False  #  TODO Temporarily turned off
            print("RoMa Resolution Printing:")
            print("ws: ", ws)
            print("hs: ", hs)
            print("ws_upsample: ", ws_upsample)
            print("hs_upsample: ", hs_upsample)
            print("im_A shape: ", im_A.shape)
            print("im_B shape: ", im_B.shape)

            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True, clahe = False
            )
            im_A, im_B = test_transform((im_A, im_B))
            im_B_test = im_B
            print("im_B_test requires grad: ", im_B_test.requires_grad)
            # if im_B_test.requires_grad:
            #     im_B_test.retain_grad()  # TODO Retain grad for im_B

            batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
        # else: #  TODO Turned of if batched for now
        #     print("In batched beginning")
        #
        #     b, c, h, w = im_A.shape
        #     b, c, h2, w2 = im_B.shape
        #     assert w == w2 and h == h2, "For batched images we assume same size"
        #     batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
        #     if h != self.h_resized or self.w_resized != w:
        #         warn("Model resolution and batch resolution differ, may produce unexpected results")
        #     hs, ws = h, w
        finest_scale = 1
        # Run matcher
        print("Symmetric: ", symmetric)
        if symmetric:
            print("Inside symmetric true")
            corresps, f_q_pyramid_1_test, X_encoder_in, x_s = self.forward_symmetric_SD(batch)
        else:
            corresps, f_q_pyramid_1_test, X_cat, X_encoder_in, x_s, x_q = self.forward_SD(batch, batched = True)

        # if self.upsample_preds:  # TODO Commented for now
        #     hs, ws = self.upsample_res

        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)

        print("self.upsample_preds", self.upsample_preds)
        self.upsample_preds = False #  TODO: Turned off for now

        # TODO Skipping upsample_preds step for now
        # if self.upsample_preds:
        #     print("In Upsample_preds")
        #     finest_corresps = corresps[finest_scale]
        #     # torch.cuda.empty_cache() # TODO: Turned off for now
        #     test_transform = get_tuple_transform_ops(
        #         resize=(hs, ws), normalize=True
        #     )
        #     if self.recrop_upsample:
        #         certainty = corresps[finest_scale]["certainty"]
        #         print(certainty.shape)
        #         im_A = self.recrop(certainty[0,0], im_A_path)
        #         im_B = self.recrop(certainty[1,0], im_B_path)
        #         #TODO: need to adjust corresps when doing this (From before)
        #     elif isinstance(im_A_path, (str, os.PathLike)):  # TODO only open if path
        #         im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
        #     im_A, im_B = test_transform((im_A, im_B))
        #     im_A, im_B = im_A[None].to(device), im_B[None].to(device)
        #     scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
        #     print("im_A requires grad: ", im_A.requires_grad)
        #     print("im_B requires grad: ", im_B.requires_grad)
        #     batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
        #     im_B_test = im_B
        #     print("DTYPE TEST im_B: ", im_B.dtype)
        #     if im_B_test.requires_grad:
        #         im_B_test.retain_grad()
        #
        #     if symmetric:
        #         print("Symmetric TRUE")
        #         corresps, f_q_pyramid_1_test = self.forward_symmetric_SD(batch, upsample = True, batched=True, scale_factor = scale_factor)
        #     else:
        #         print("Symmeteric FALSE")
        #         corresps = self.forward(batch, batched = True, upsample=True, scale_factor = scale_factor)
        #
        # else:
        #     im_B_test = im_B
        #     f_q_pyramid_1_test = None


        im_A_to_im_B = corresps[finest_scale]["flow"]
        print("DTYPE TEST im_A_to_im_B: ", im_A_to_im_B.dtype)

        print("im_A_to_im_B requires grad: ", im_A_to_im_B.requires_grad)
        im_A_to_im_B_test = im_A_to_im_B
        # if im_A_to_im_B_test.requires_grad:
        #     im_A_to_im_B_test.retain_grad()

        certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
        im_A_to_im_B = im_A_to_im_B.permute(
            0, 2, 3, 1
            )
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)

        certainty = certainty.sigmoid()  # logits -> probs

        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0

            # certainty[wrong[:,None]] = 0 # TODO Original assignment
            # TODO attempt replace inplace assignment
            certainty = certainty.clone()
            certainty[wrong[:, None]] = 0  # TODO Original assignment

        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        if symmetric:
            A_to_B, B_to_A = im_A_to_im_B.chunk(2)
            q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
            im_B_coords = im_A_coords
            s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp),dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        if batched:
            print("HELLO IN BATCHED RETURN")
            return (
                warp,
                certainty[:, 0]
            )
        else:
            print("HELLO IN NOT BATCHED RETURN")

            return (
                warp[0],
                certainty[0, 0],
                im_B_test,
                f_q_pyramid_1_test,
                im_A_to_im_B_test,
                X_cat,
                X_encoder_in,
                x_s,
                x_q,
            )
    # def extract_backbone_features_SD_opt(self, batch, batched = True, upsample = False):
    #     print("In extract_backbone_features_SD_opt!")
    #     x_q = batch["im_A"]
    #     x_s = batch["im_B"]
    #     print("x_q requires_grad: ", x_q.requires_grad)
    #     print("x_s requires_grad: ", x_s.requires_grad)
    #     print("x_q shape: ", x_q.shape)
    #     print("x_s shape: ", x_s.shape)
    #     print("x_q dtype: ", x_q.dtype)
    #     print("x_s dtype: ", x_s.dtype)
    #     if batched:
    #         print("In extract_backbone_features batched")
    #         print("s first then q")
    #         X = torch.cat((x_s, x_q), dim=0) # TODO Test swap (Standard is x_q,x_s)
    #         # X = torch.stack((x_q.unsqueeze(0), x_s.unsqueeze(0)), dim=0).squeeze(1)
    #         print("X.shape: ", X.shape)
    #         feature_pyramid = self.encoder(X, upsample=upsample)
    #     else:
    #         print("In extract_backbone_features not batched")
    #         feature_pyramid = self.encoder(x_q, upsample=upsample), self.encoder(x_s, upsample=upsample)
    #     return feature_pyramid

    # def forward_SD_opt(self, batch, batched = True, upsample = False, scale_factor = 1):
    #     print("In forward_SD_opt!")
    #     feature_pyramid = self.extract_backbone_features_SD_opt(batch, batched=batched, upsample = upsample)
    #     if batched:
    #         f_q_pyramid = {
    #             scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
    #         }
    #         f_s_pyramid = {
    #             scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
    #         }
    #     else:
    #         f_q_pyramid, f_s_pyramid = feature_pyramid
    #     corresps = self.decoder(f_q_pyramid,
    #                             f_s_pyramid,
    #                             upsample = upsample,
    #                             **(batch["corresps"] if "corresps" in batch else {}),
    #                             scale_factor=scale_factor)
    #
    #     return corresps
    def extract_backbone_features_SD_opt(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        print("x_q requires_grad: ", x_q.requires_grad)
        print("x_s requires_grad: ", x_s.requires_grad)
        print("x_q shape: ", x_q.shape)
        print("x_s shape: ", x_s.shape)
        print("x_q dtype: ", x_q.dtype)
        print("x_s dtype: ", x_s.dtype)
        if batched:
            print("In extract_backbone_features batched")
            X = torch.cat((x_q, x_s), dim=0) # TODO Test swap (Standard is x_q,x_s)
            # X = torch.stack((x_q.unsqueeze(0), x_s.unsqueeze(0)), dim=0).squeeze(1)
            print("X.shape: ", X.shape)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            print("In extract_backbone_features not batched")
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid, X, x_s, x_q

    def forward_SD_opt(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid, X_encoder_in, x_s, x_q = self.extract_backbone_features_SD_opt(batch, batched = batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid

        print("X_encoder_in requires_grad: ", X_encoder_in.requires_grad)
        print("X_encoder_in shape: ", X_encoder_in.shape)
        # if X_encoder_in.requires_grad:
        #     print("X_encoder_in does require_grad")
        #     X_encoder_in.retain_grad()
        print("x_s requires_grad: ", x_s.requires_grad)
        print("x_s shape: ", x_s.shape)
        # if x_s.requires_grad:
        #     print("x_s does require_grad")
        #     x_s.retain_grad()

        print("x_q requires_grad: ", x_q.requires_grad)
        print("x_q shape: ", x_q.shape)
        # if x_q.requires_grad:
        #     print("x_q does require_grad")
        #     x_q.retain_grad()

        for key in f_q_pyramid.keys():
            print("f_q_pyramid key: ", key)
        print("f_q_pyramid[1] requires_grad: ", f_q_pyramid[1].requires_grad)
        f_q_pyramid_1_test = f_q_pyramid[1]

        # if f_q_pyramid_1_test.requires_grad:
        #     print("f_q_pyramid[1] does require_grad")
        #     f_q_pyramid_1_test.retain_grad()
        # else:
        #     print("f_q_pyramid[1] does not require_grad")


        corresps = self.decoder(f_q_pyramid,
                                f_s_pyramid,
                                upsample=upsample,
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)

        return corresps, f_q_pyramid_1_test, X_encoder_in, x_s, x_q

    # def match_SD_opt(
    #     self,
    #     im_A_path,
    #     im_B_path,
    #     *args,
    #     batched=False,
    #     device=None,
    # ):
    #     print("HELLO IN ROMA .match()")
    #     if device is None:
    #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     if isinstance(im_A_path, (str, os.PathLike)):
    #         im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
    #     else:
    #         # Assume its not a path
    #         im_A, im_B = im_A_path, im_B_path
    #     symmetric = self.symmetric
    #     self.train(False)
    #
    #
    #     # with torch.no_grad(): # TODO Turned of no_grad() for now
    #     if not batched:
    #         b = 1
    #         # w, h = im_A.size
    #         # w2, h2 = im_B.size
    #
    #         # channels, w, h = im_A.shape  # TODO change how extract shape
    #         # channels2, w2, h2 = im_B.shape  # TODO change how extract shape
    #
    #         # Get images in good format
    #         ws = self.w_resized
    #         hs = self.h_resized
    #         print("ws in match(): ", ws)
    #         print("hs in match(): ", hs)
    #         test_transform = get_tuple_transform_ops(
    #             resize=(hs, ws), normalize=True, clahe = False
    #         )
    #         im_A, im_B = test_transform((im_A, im_B))
    #         batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
    #     else:
    #         b, c, h, w = im_A.shape
    #         b, c, h2, w2 = im_B.shape
    #         assert w == w2 and h == h2, "For batched images we assume same size"
    #         batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
    #         if h != self.h_resized or self.w_resized != w:
    #             warn("Model resolution and batch resolution differ, may produce unexpected results")
    #         hs, ws = h, w
    #     finest_scale = 1
    #     # Run matcher
    #     if symmetric:
    #         corresps  = self.forward_symmetric(batch)
    #     else:
    #         corresps = self.forward(batch, batched=True)  # TODO batched is usually True
    #
    #     if self.upsample_preds:
    #         hs, ws = self.upsample_res
    #
    #     if self.attenuate_cert:
    #         low_res_certainty = F.interpolate(
    #         corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
    #         )
    #         cert_clamp = 0
    #         factor = 0.5
    #         low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
    #
    #     print("Right before upsample preds")
    #     print("self.upsample_preds: ", self.upsample_preds)
    #     if self.upsample_preds:
    #         print("IN UPSAMPLE PREDS!!!!")
    #         finest_corresps = corresps[finest_scale]
    #         torch.cuda.empty_cache()
    #         test_transform = get_tuple_transform_ops(
    #             resize=(hs, ws), normalize=True
    #         )
    #         if self.recrop_upsample:
    #             certainty = corresps[finest_scale]["certainty"]
    #             print(certainty.shape)
    #             im_A = self.recrop(certainty[0,0], im_A_path)
    #             im_B = self.recrop(certainty[1,0], im_B_path)
    #             #TODO: need to adjust corresps when doing this (From before)
    #         elif isinstance(im_A_path, (str, os.PathLike)):  # TODO only open if path
    #             im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
    #         im_A, im_B = test_transform((im_A, im_B))
    #         im_A, im_B = im_A[None].to(device), im_B[None].to(device)
    #         scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
    #         batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
    #         print("Symmetric: ", symmetric)
    #         print("DTYPE TEST im_B: ", im_B.dtype)
    #
    #         if symmetric:
    #             corresps = self.forward_symmetric(batch, upsample = True, batched=True, scale_factor = scale_factor)
    #         else:
    #             corresps = self.forward_SD_opt(batch, batched = True, upsample=True, scale_factor = scale_factor) # TODO batched is usually True
    #
    #     im_A_to_im_B = corresps[finest_scale]["flow"]
    #
    #     print("DTYPE TEST im_A_to_im_B: ", im_A_to_im_B.dtype)
    #
    #     certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
    #     if finest_scale != 1:
    #         im_A_to_im_B = F.interpolate(
    #         im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
    #         )
    #         certainty = F.interpolate(
    #         certainty, size=(hs, ws), align_corners=False, mode="bilinear"
    #         )
    #     im_A_to_im_B = im_A_to_im_B.permute(
    #         0, 2, 3, 1
    #         )
    #     # Create im_A meshgrid
    #     im_A_coords = torch.meshgrid(
    #         (
    #             torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
    #             torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
    #         )
    #     )
    #     im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    #     im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
    #
    #     certainty = certainty.sigmoid()  # logits -> probs
    #
    #     im_A_coords = im_A_coords.permute(0, 2, 3, 1)
    #     if (im_A_to_im_B.abs() > 1).any() and True:
    #         wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
    #
    #         # certainty[wrong[:,None]] = 0 # TODO Original assignment
    #         # TODO attempt replace inplace assignment
    #         certainty = certainty.clone()
    #         certainty[wrong[:, None]] = 0  # TODO Original assignment
    #
    #     im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
    #     if symmetric:
    #         A_to_B, B_to_A = im_A_to_im_B.chunk(2)
    #         q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
    #         im_B_coords = im_A_coords
    #         s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
    #         warp = torch.cat((q_warp, s_warp),dim=2)
    #         certainty = torch.cat(certainty.chunk(2), dim=3)
    #     else:
    #         print("In warp creation:")
    #         print("im_A_coords shape: ", im_A_coords.shape)
    #         print("im_A_to_im_B shape: ", im_A_to_im_B.shape)
    #
    #         warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    #         print("warp shape: ", warp.shape)
    #     if batched:
    #         print("HELLO IN BATCHED RETURN")
    #         return (
    #             warp,
    #             certainty[:, 0]
    #         )
    #     else:
    #         print("HELLO IN NOT BATCHED RETURN")
    #         return (
    #             warp[0],
    #             certainty[0, 0],
    #         )





    def match_SD_opt(
        self,
        im_A_path,
        im_B_path,
        *args,
        batched=False,
        device=None,
    ):
        print("HELLO IN ROMA SD .match()")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(im_A_path, (str, os.PathLike)):
            im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
        else:
            print("Assuming its not a path")
            # Assume its not a path
            im_A, im_B = im_A_path, im_B_path
        symmetric = self.symmetric
        self.train(False)
        # print("self.train(True)")
        # self.train(True)  # TODO Turned on training for now


        # with torch.no_grad(): # TODO Turned of no_grad() for now
        if not batched:
            # print("In not batched beginning")
            b = 1
            # w, h = im_A.size
            # w2, h2 = im_B.size

            # channels, w, h = im_A.shape  # TODO change how extract shape
            # channels2, w2, h2 = im_B.shape  # TODO change how extract shape

            # Get images in good format
            ws = self.w_resized
            hs = self.h_resized

            hs_upsample, ws_upsample = self.upsample_res
            # self.upsample_preds = False  #  TODO Temporarily turned off
            print("RoMa Resolution Printing:")
            print("ws: ", ws)
            print("hs: ", hs)
            print("ws_upsample: ", ws_upsample)
            print("hs_upsample: ", hs_upsample)
            print("im_A shape: ", im_A.shape)
            print("im_B shape: ", im_B.shape)

            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True, clahe = False
            )
            im_A, im_B = test_transform((im_A, im_B))
            im_B_test = im_B
            # print("im_B_test requires grad: ", im_B_test.requires_grad)
            # if im_B_test.requires_grad:
            #     im_B_test.retain_grad()  # TODO Retain grad for im_B

            batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
        else:
            print("In batched beginning")

            b, c, h, w = im_A.shape
            b, c, h2, w2 = im_B.shape
            assert w == w2 and h == h2, "For batched images we assume same size"
            batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
            if h != self.h_resized or self.w_resized != w:
                warn("Model resolution and batch resolution differ, may produce unexpected results")
            hs, ws = h, w
        finest_scale = 1
        # Run matcher
        print("Symmetric: ", symmetric)
        if symmetric:
            # print("Inside symmetric true")
            corresps, f_q_pyramid_1_test, X_encoder_in, x_s = self.forward_symmetric_SD(batch)
        else:
            corresps, f_q_pyramid_1_test, X_encoder_in, x_s, x_q = self.forward_SD_opt(batch, batched = True)

        if self.upsample_preds:  # TODO Commented for now
            hs, ws = self.upsample_res

        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)

        # print("self.upsample_preds", self.upsample_preds)
        # self.upsample_preds = False #  TODO: Turned off for now

        # TODO Skipping upsample_preds step for now
        if self.upsample_preds:
            print("In Upsample_preds")
            finest_corresps = corresps[finest_scale]
            # torch.cuda.empty_cache() # TODO: Turned off for now
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True
            )
            if self.recrop_upsample:
                certainty = corresps[finest_scale]["certainty"]
                print(certainty.shape)
                im_A = self.recrop(certainty[0,0], im_A_path)
                im_B = self.recrop(certainty[1,0], im_B_path)
                #TODO: need to adjust corresps when doing this (From before)
            elif isinstance(im_A_path, (str, os.PathLike)):  # TODO only open if path
                im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
            im_A, im_B = test_transform((im_A, im_B))
            im_A, im_B = im_A[None].to(device), im_B[None].to(device)
            scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
            # print("im_A requires grad: ", im_A.requires_grad)
            # print("im_B requires grad: ", im_B.requires_grad)
            batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
            im_B_test = im_B
            # print("DTYPE TEST im_B: ", im_B.dtype)
            # if im_B_test.requires_grad:
            #     im_B_test.retain_grad()

            if symmetric:
                print("Symmetric TRUE")
                corresps, f_q_pyramid_1_test = self.forward_symmetric_SD(batch, upsample = True, batched=True, scale_factor = scale_factor)
            else:
                print("Symmeteric FALSE")
                corresps = self.forward(batch, batched = True, upsample=True, scale_factor = scale_factor)

        else:
            im_B_test = im_B
            f_q_pyramid_1_test = None


        im_A_to_im_B = corresps[finest_scale]["flow"]
        print("DTYPE TEST im_A_to_im_B: ", im_A_to_im_B.dtype)

        print("im_A_to_im_B requires grad: ", im_A_to_im_B.requires_grad)
        im_A_to_im_B_test = im_A_to_im_B
        # if im_A_to_im_B_test.requires_grad:
        #     im_A_to_im_B_test.retain_grad()

        certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
        im_A_to_im_B = im_A_to_im_B.permute(
            0, 2, 3, 1
            )
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)

        certainty = certainty.sigmoid()  # logits -> probs

        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0

            # certainty[wrong[:,None]] = 0 # TODO Original assignment
            # TODO attempt replace inplace assignment
            certainty = certainty.clone()
            certainty[wrong[:, None]] = 0  # TODO Original assignment

        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        if symmetric:
            A_to_B, B_to_A = im_A_to_im_B.chunk(2)
            q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
            im_B_coords = im_A_coords
            s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp),dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        if batched:
            print("HELLO IN BATCHED RETURN")
            return (
                warp,
                certainty[:, 0]
            )
        else:
            print("HELLO IN NOT BATCHED RETURN")

            return (
                warp[0],
                certainty[0, 0],
                im_B_test,
                f_q_pyramid_1_test,
                im_A_to_im_B_test,
                X_encoder_in,
                x_s,
                x_q,
            )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None):
        assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
        im_A = im_A.resize((W,H))
        im_B = im_B.resize((W,H))
            
        x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)

        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        im_B_transfer_rgb = F.grid_sample(
        x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
        )[0]
        warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
        white_im = torch.ones((H,2*W),device=device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from roma.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=False).save(save_path)
        return vis_im

    # @torch.inference_mode()
    def match_david(
            self,
            im_A_path,
            im_B_path,
            *args,
            batched=False,
            device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(im_A_path, (str, os.PathLike)):
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        else:
            im_A, im_B = im_A_path, im_B_path
        print("In match david")
        print("im_A shape: ", im_A.shape)
        print("im_B shape: ", im_B.shape)
        print("self.symmetric: ", self.symmetric)
        print("batched: ", batched)
        print("self.upsample_preds: ", self.upsample_preds)

        symmetric = self.symmetric
        self.train(False)
        if not batched:
            b = 1
            w, h = im_A.size
            w2, h2 = im_B.size
            # Get images in good format
            ws = self.w_resized
            hs = self.h_resized

            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True, clahe=False
            )
            im_A, im_B = test_transform((im_A, im_B))
            batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
        else:
            b, c, h, w = im_A.shape
            b, c, h2, w2 = im_B.shape
            assert w == w2 and h == h2, "For batched images we assume same size"
            batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
            if h != self.h_resized or self.w_resized != w:
                warn("Model resolution and batch resolution differ, may produce unexpected results")
            hs, ws = h, w
        finest_scale = 1
        # Run matcher
        if symmetric:
            corresps = self.forward_symmetric(batch)
        else:
            corresps = self.forward(batch, batched=True)

        if self.upsample_preds:
            hs, ws = self.upsample_res

        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
                corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

        if self.upsample_preds:
            finest_corresps = corresps[finest_scale]
            torch.cuda.empty_cache()
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True
            )
            im_A, im_B = test_transform((Image.open(im_A_path).convert('RGB'), Image.open(im_B_path).convert('RGB')))
            im_A, im_B = im_A[None].to(device), im_B[None].to(device)
            scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
            batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
            if symmetric:
                corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
            else:
                corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

        im_A_to_im_B = corresps[finest_scale]["flow"]
        certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
                im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
                certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
        im_A_to_im_B = im_A_to_im_B.permute(
            0, 2, 3, 1
        )
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            ),
            indexing='ij'
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
        certainty = certainty.sigmoid()  # logits -> probs
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        if symmetric:
            A_to_B, B_to_A = im_A_to_im_B.chunk(2)
            q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
            im_B_coords = im_A_coords
            s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp), dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        if batched:
            return (
                warp,
                certainty[:, 0]
            )
        else:
            return (
                warp[0],
                certainty[0, 0],
            )
