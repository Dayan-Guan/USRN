from models.modeling.backbone import resnet, xception, drn, mobilenet
from models.modeling.backbone import resnet_AuxLayers234

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError

def build_backbone_AuxLayers234(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet_AuxLayers234.ResNet101_AuxLayers234(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet_AuxLayers234.ResNet50_AuxLayers234(output_stride, BatchNorm)
    else:
        raise NotImplementedError