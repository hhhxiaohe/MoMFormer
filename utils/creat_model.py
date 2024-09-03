# TODO:Creat model
# from Code.model.build_model.Build_MixImages import Build_MixImages
from model.build_model.Build_swin import Build_swin
# # from Code.model.UNetFormer import UNetFormer
# from Code.model.build_model.Build_cmt import Build_cmt
from model.UisNet import UisNet
from model.D_UNet import D_UNet
from model.DIR_DeepLabv3plus import DIR_DeepLabv3plus
from model.LASNet import LASNet
from model.TransUNet import transNet
from model.MANet import MANet
from model.MoMFormer import MoMFormer
from model.BuildFormer import BuildFormerSegDP
from model.ABCNet import ABCNet
from model.HRNet import HighResolutionNet
from model.swin_unet import SwinTransformerSys
from model.Convnextv2_ import ConvNeXtV2
from model.swin_v2 import Swin_v2


def creat_model(model_type, n_classes, in_chans, img_size=None):
    # Unimodal
    # if model_type == 'MixImages':
    #     model = Build_MixImages(n_classes, img_size=img_size, in_chans=in_chans, use_uper=False, aux=True)
    if model_type == 'MoMFormer':
        model = MoMFormer(n_classes=n_classes, in_chans=in_chans)
    elif model_type == 'BuildFormer':
        model = BuildFormerSegDP(n_classes=n_classes)
    elif model_type == 'MANet':
        model = MANet(n_classes=n_classes, in_chans=3)
    elif model_type == 'ABCNet':
        model = ABCNet(n_classes = n_classes ,in_chans=3)
    elif model_type == 'HRNet':
        model = HighResolutionNet(n_classes = n_classes,in_chans=3)
    elif model_type == 'SwinUNet':
        model = SwinTransformerSys(n_classes=n_classes)
    elif model_type == 'Swin_Transformer':
        model = Build_swin(n_classes=n_classes)
    elif model_type == 'Convnextv2_':
        model = ConvNeXtV2(n_classes=n_classes)
    elif model_type == 'Swin_v2':
        model = Swin_v2(n_classes=n_classes)
    # elif model_type == 'swin':
    #     model = Build_swin(n_classes, in_chans=in_chans, use_uper=True, aux=True)
    # elif model_type == 'CMT':
    #     model = Build_cmt(n_classes, img_size=img_size, in_chans=in_chans, use_uper=True, aux=False)
    # elif model_type == 'TransUNet':
    #     model = transNet(n_classes, img_size=img_size, in_chans=in_chans)
    # elif model_type == 'MANet':
    #     model = MANet(n_classes, in_chans=in_chans)
    # elif model_type == 'UNetFormer':
    #     model = UNetFormer(n_classes, in_chans=in_chans, decode_channels=512, aux=True)
    # # Multimodal
    elif model_type == 'UisNet':
        model = UisNet(n_classes, in_chans=in_chans)
    elif model_type == 'D_UNet':
        model = D_UNet(n_classes, in_chans=in_chans, dim=350)
    elif model_type == 'v3+':
        model = DIR_DeepLabv3plus(n_classes, in_chans=in_chans)
    elif model_type == 'LASNet':
        model = LASNet(n_classes)
    else:
        raise NotImplementedError(f"Model type:'{model_type}' does not exist, please check the 'model_type' arg!")

    return model
