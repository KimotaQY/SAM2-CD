# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import yaml

from peft import LoraConfig, get_peft_model

from models.modeling.sam2_base import SAM2Base
from models.modeling.backbones.hieradet import Hiera
from models.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from models.modeling.position_encoding import PositionEmbeddingSine
from models.modeling.sam.mask_decoder import MaskDecoder

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    apply_postprocessing=False,
    **kwargs,
):
    # 读取配置
    with open(config_file, "r", encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    image_encoder_cfg = config_data["model"]["image_encoder"]
    trunk_cfg = image_encoder_cfg["trunk"]
    neck_cfg = image_encoder_cfg["neck"]
    position_encoding_cfg = neck_cfg["position_encoding"]
    
    model = SAM2Base(
        image_encoder = ImageEncoder(
            trunk=Hiera(
                embed_dim=trunk_cfg["embed_dim"], 
                num_heads=trunk_cfg["num_heads"],
                stages=trunk_cfg.get("stages", (2, 3, 16, 3)),
                global_att_blocks=trunk_cfg.get("global_att_blocks", (12, 16, 20,)),
                window_pos_embed_bkg_spatial_size=trunk_cfg.get("window_pos_embed_bkg_spatial_size", (14, 14)),
                window_spec=trunk_cfg.get("window_spec", (8, 4, 14, 7,))
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=position_encoding_cfg["num_pos_feats"],
                    normalize=position_encoding_cfg["normalize"],
                    scale=position_encoding_cfg["scale"],
                    temperature=position_encoding_cfg["temperature"]
                ),
                d_model=neck_cfg["d_model"],
                backbone_channel_list=neck_cfg["backbone_channel_list"],
                fpn_top_down_levels=neck_cfg["fpn_top_down_levels"],
                fpn_interp_model=neck_cfg["fpn_interp_model"],
            )
        ),
        mask_decoder=MaskDecoder(),
        image_size = config_data["model"]["image_size"]
    )
    
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     target_modules=["qkv"]
    # )
    # model = get_peft_model(model, lora_config)
    
    _load_checkpoint(model, ckpt_path)

    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        # if missing_keys:
        #     logging.error(missing_keys)
        #     raise RuntimeError()
        # if unexpected_keys:
        #     logging.error(unexpected_keys)
        #     raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
