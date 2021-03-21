import time
import torch
import torch.nn as nn
from .backbone import build_backbone
from .fpn import build_fpn
from .head import build_head
import pytorch_lightning as pl
from .ModelBase import ModelBase
import numpy as np


class OneStage(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg=None,
                 head_cfg=None):
        super(OneStage, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'fpn') and self.fpn is not None:
            x = self.fpn(x)
        if hasattr(self, 'head'):
            out = []
            for xx in x:
                out.append(self.head(xx))
            x = tuple(out)
        return x

    def inference(self, meta):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            results = self.head.post_process(preds, meta)
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return results

    def forward_train(self, gt_meta):
        images = gt_meta['image']
        preds = self(images)
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states


class GFL(OneStage):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg,
                 head_cfg, ):
        super(GFL, self).__init__(backbone_cfg,
                                   fpn_cfg,
                                   head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)
        return x


class NanodetLightning(ModelBase):
    def __init__(self, model_cfg):
        super(NanodetLightning, self).__init__(model_cfg)
        backbone_cfg = model_cfg['backbone']
        fpn_cfg = model_cfg['fpn']
        head_cfg = model_cfg['head']
        self.model = GFL(backbone_cfg, fpn_cfg, head_cfg)

    def training_step(self, batch, batch_idx):
        preds, loss, loss_states = self.model.forward_train(batch)
        return {'loss': loss, 'log': loss_states}

    def validation_step(self, batch, batch_idx):
        preds, loss, loss_states = self.model.forward_train(batch)
        return {'loss': loss, 'log': loss_states}



