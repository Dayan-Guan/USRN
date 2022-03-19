import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.modeling.aspp import build_aspp
from models.modeling.decoder import build_decoder
# from models.modeling.backbone import build_backbone
from models.modeling.backbone import build_backbone_AuxLayers234

class Deeplab_SubCls(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False):
        super(Deeplab_SubCls, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        assert sync_bn == False
        assert freeze_bn == False

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        # self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.backbone = build_backbone_AuxLayers234(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.aspp_SubCls = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder_SubCls = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        # high_level_feat, low_level_feat = self.backbone(input)
        high_level_feat, low_level_feat, high_level_feat_SubCls = self.backbone(input)
        x = self.aspp(high_level_feat)
        x = self.decoder(x, low_level_feat)
        # return x
        x_SubCls = self.aspp_SubCls(high_level_feat_SubCls)
        x_SubCls = self.decoder_SubCls(x_SubCls, low_level_feat)
        return x, x_SubCls

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_module_params(self):
        # modules = [self.aspp, self.decoder]
        modules = [self.aspp, self.decoder, self.aspp_SubCls, self.decoder_SubCls]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


