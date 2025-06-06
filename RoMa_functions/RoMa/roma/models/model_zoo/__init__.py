from typing import Union, Tuple
import torch
import sys
from RoMa_functions.RoMa.roma.models.model_zoo.roma_models import roma_model

weight_urls = {
    "roma": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    # hopefully this doesnt change :D
}


def roma_outdoor(device, weights=None, dinov2_weights=None, coarse_res: Union[int, Tuple[int, int]] = 560,
                 upsample_res: Union[int, Tuple[int, int]] = 864):
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["roma"]["outdoor"],
                                                     map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                            map_location=device)
    model = roma_model(resolution=coarse_res, upsample_preds=True,
                       weights=weights, dinov2_weights=dinov2_weights, device=device)
    model.upsample_res = upsample_res
    print(f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model


def roma_indoor(device, weights=None, dinov2_weights=None, coarse_res: Union[int, Tuple[int, int]] = 560,
                upsample_res: Union[int, Tuple[int, int]] = 864):
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["roma"]["indoor"],
                                                     map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                            map_location=device)
    model = roma_model(resolution=coarse_res, upsample_preds=True,
                       weights=weights, dinov2_weights=dinov2_weights, device=device)
    model.upsample_res = upsample_res
    print(f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model
