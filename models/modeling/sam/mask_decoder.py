import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Net import (
    FeatureReinforcementModule,
    TemporalFusionModule,
    GlobalContextAggregation,
    Decoder,
    WaveFusion,
)


class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()

        channles = [256] * 5
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.frm = FeatureReinforcementModule(channles, self.mid_d, drop_rate=0.2)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        # self.tfm = TemporalFusionModule(256, self.mid_d)
        # self.gca = GlobalContextAggregation(self.mid_d, self.mid_d, drop_rate=0.2)
        self.wf = WaveFusion(self.mid_d, "haar")
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, feats):
        half_batch_size = feats[0].shape[0] // 2
        x1_2, x1_3, x1_4, x1_5 = [x[:half_batch_size] for x in feats]
        x2_2, x2_3, x2_4, x2_5 = [x[half_batch_size:] for x in feats]

        x1_2, x1_3, x1_4, x1_5 = self.frm(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.frm(x2_2, x2_3, x2_4, x2_5)

        # temporal fusion
        c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        # global context of high-level and low-level
        # gc_c4 = self.gca(c4, c5)
        gc_c4 = self.wf(c4, c5)

        # fpn
        mask_p2, mask_p3, mask_p4 = self.decoder(c2, c3, c4, c5, gc_c4)

        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode="bilinear")
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode="bilinear")
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode="bilinear")

        return mask_p2, mask_p3, mask_p4
