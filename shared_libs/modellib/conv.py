
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import module_starGAN as md_starGAN
from .dnn_models import SincNet as CNN 
from .dnn_models import MLP
from .RawNetBasicBlock import Bottle2neck, PreEmphasis
from asteroid_filterbanks import Encoder, ParamSincFB
# import module_starGAN as md_starGAN
import logging
import math

import numpy as np
import torch

from .parallel_wavegan_layers import Conv1d, Conv1d1x1
from .parallel_wavegan_layers import WaveNetResidualBlock as ResidualBlock
from .parallel_wavegan_layers import upsample
# from parallel_wavegan.utils import read_hdf5

def init_weights(layer):
    """
    Initialize weights.
    """
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None: layer.bias.data.zero_()


########################################################################################################################
# Modules
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, class_dim, num_classes):
        super(Decoder, self).__init__()
        # 1. Architecture
        self._fc = nn.Linear(in_features=class_dim, out_features=num_classes, bias=False)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, emb):
        output = self._fc(emb)
        # average output on dim 1 (time axis)
        output = output.mean(dim=1)
        return output


class DensityEstimator(nn.Module):
    """
    Estimating probability density.
    """
    def __init__(self, style_dim, class_dim):
        super(DensityEstimator, self).__init__()
        # 1. Architecture
        # (1) Pre-fc
        self._fc_style = nn.Linear(in_features=style_dim, out_features=128, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=128, bias=True)
        # (2) FC blocks
        self._fc_blocks = nn.Sequential(
            # Layer 1
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Linear(in_features=256, out_features=1, bias=True))
        # 2. Init weights
        self.apply(init_weights)

    def _call_method(self, style_emb, class_emb):
        # print("style_emb.size():", style_emb.size())
        # print("class_emb.size():", class_emb.size())
        style_emb = self._fc_style(style_emb)
        class_emb = self._fc_class(class_emb)
        # print("After fc_style, style_emb.size():", style_emb.size())
        # print("After fc_class, class_emb.size():", class_emb.size())
        # print("torch.cat([style_emb, class_emb]).size():", torch.cat([style_emb, class_emb], dim=2).size())
        return self._fc_blocks(torch.cat([style_emb, class_emb], dim=2))

    def forward(self, style_emb, class_emb, mode):
        assert mode in ['orig', 'perm']
        # 1. q(s, t)
        if mode == 'orig':
            return self._call_method(style_emb, class_emb)
        # 2. q(s)q(t)
        else:
            # Permutation
            style_emb_permed = style_emb[torch.randperm(style_emb.size(0)).to(style_emb.device)]
            class_emb_permed = class_emb[torch.randperm(class_emb.size(0)).to(class_emb.device)]
            return self._call_method(style_emb_permed, class_emb_permed)


# ----------------------------------------------------------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------------------------------------------------------

class EncoderMNIST(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderMNIST, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorMNIST(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorMNIST, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x


class DiscriminatorMNIST(nn.Module):
    """
    Discriminator Module.
    """
    def __init__(self):
        super(DiscriminatorMNIST, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=512, out_features=2, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        # 1. Convolution
        x = self._conv_blocks(x)
        # 2. FC
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self._fc(x)
        # Return
        return x

# ----------------------------------------------------------------------------------------------------------------------
# VC
# ----------------------------------------------------------------------------------------------------------------------
import torch.nn.functional as F

import torch
import torch.nn as nn

class EncoderTIMIT(nn.Module):

# ###########SINCNET################
#     def __init__(self, nz):
#         super(EncoderTIMIT, self).__init__()

#         # wlen = 3200 # 16000*200/1000 = 3200
#         fs = 16000
#         # cnn_N_filt = [20, 10, 10] # cnn_N_filt = [80, 60, 60]
#         cnn_N_filt = [80, 60, 60]
#         # cnn_len_filt = [32, 8, 6] # cnn_len_filt = [251, 5, 5]
#         cnn_len_filt = [251, 5, 5]
#         # cnn_max_pool_len = [2, 2, 2] # cnn_max_pool_len = [3, 3, 3]
#         # cnn_max_pool_len = [3, 3, 3]
#         cnn_max_pool_len = [2, 2, 2]
#         cnn_use_laynorm_inp = True
#         cnn_use_batchnorm_inp = False
#         cnn_use_laynorm = [True, True, True]
#         cnn_use_batchnorm = [False, False, False]
#         cnn_act = ['leaky_relu', 'leaky_relu', 'leaky_relu']
#         cnn_drop = [0.0, 0.0, 0.0]
        
#         self.cnn_arch = {
#             'input_dim': 320, # mel: 80
#             'fs': fs,
#             'cnn_N_filt': cnn_N_filt,
#             'cnn_len_filt': cnn_len_filt,
#             'cnn_max_pool_len': cnn_max_pool_len,
#             'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
#             'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
#             'cnn_use_laynorm': cnn_use_laynorm,
#             'cnn_use_batchnorm': cnn_use_batchnorm,
#             'cnn_act': cnn_act,
#             'cnn_drop': cnn_drop
#         }
                
#         self.cnn_net = CNN(self.cnn_arch)

#         fc_lay = [2048, 2048, nz] # fc_lay = [2048, 2048, 2048]
#         fc_drop = [0.0, 0.0, 0.0]
#         fc_use_laynorm_inp = True
#         fc_use_batchnorm_inp = False
#         fc_use_batchnorm = [True, True, True]
#         fc_use_laynorm = [False, False, False]
#         fc_act = ['leaky_relu', 'leaky_relu', 'leaky_relu']

#         self.dnn1_arch = {
#             'input_dim':self.cnn_net.out_dim,
#             'fc_lay': fc_lay,
#             'fc_drop': fc_drop,
#             'fc_use_batchnorm': fc_use_batchnorm,
#             'fc_use_laynorm': fc_use_laynorm,
#             'fc_use_laynorm_inp': fc_use_laynorm_inp,
#             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
#             'fc_act': fc_act
#         }

        
#         self.dnn1_net = MLP(self.dnn1_arch)

#         self.d_vector_dim = nz

#     def forward(self, signal_chunks):
        
#         batch_size, num_frames, features = signal_chunks.size()  # [64, 512, 3200] (batch_size, frames_nums, features)
#         signal_chunks = signal_chunks.view(batch_size * num_frames, features)  # (batch_size*num_frames, features)
#         # print("signal_chunks:", signal_chunks) 
#         # print("signal_chunks_size:", signal_chunks.size()) # [32768, 3200] (3200, cnn_arch input dim, from TIMITDataset)
       
#         cnn_out = self.cnn_net(signal_chunks)
#         # print("cnn_out:",cnn_out)
#         # print("cnn_out_size:", cnn_out.size()) # 32768, 60 (10, dnn1_arch input dim)
        
#         d_vectors = self.dnn1_net(cnn_out)
#         # print("d_vectors:", d_vectors)
#         # print("d_vectors_size:", d_vectors.size()) # 32768, 16 (16, d_vectors size, configs: style_dim / class_dim )
                    
#         d_vectors = d_vectors.view(batch_size, num_frames, self.d_vector_dim) # [batch_size, num_frames, d_vector_dim]
#         # print("d_vectors:", d_vectors)
#         # print("d_vectors_size:", d_vectors.size()) # 64, 512, 16        
        
#         # batch_size, num_frames, mel_features = signal_chunks.size()  # [64, 512, 80] (7, mean in datasets)
#         # signal_chunks = signal_chunks.view(batch_size * num_frames, mel_features)  # [batch_size*num_frames, mel_features*extra_dim]
#         # print("signal_chunks:", signal_chunks) 
#         # print("signal_chunks_size:", signal_chunks.size()) # 32768, 80 (80, cnn_arch input dim)
       
#         # cnn_out = self.cnn_net(signal_chunks)
#         # print("cnn_out:",cnn_out)
#         # print("cnn_out_size:", cnn_out.size()) # 32768, 10 (10, dnn1_arch input dim)
        
#         # d_vectors = self.dnn1_net(cnn_out)
#         # print("d_vectors:", d_vectors)
#         # print("d_vectors_size:", d_vectors.size()) # 32768, 16 (16, d_vectors size, configs: style_dim / class_dim )
                    
#         # d_vectors = d_vectors.view(batch_size, num_frames, self.d_vector_dim) # [batch_size, num_frames, d_vector_dim]
#         # print("d_vectors:", d_vectors)
#         # print("d_vectors_size:", d_vectors.size()) # 64, 512, 16         
        
#         return d_vectors


###########RAWNET################
    def __init__(self, block=Bottle2neck, model_scale=8, context=True, summed=True, C=1024,encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1, **kwargs):
        super().__init__()

        nOut = nOut

        self.context = context
        self.encoder_type = encoder_type
        self.log_sinc = log_sinc
        self.norm_sinc = norm_sinc
        self.out_bn = out_bn
        self.summed = summed

        self.preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.conv1 = Encoder(
            ParamSincFB(
                C // 4,
                251,
                stride=sinc_stride,
            )
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C // 4)

        self.layer1 = block(
            C // 4, C, kernel_size=3, dilation=2, scale=model_scale, pool=5
        )
        self.layer2 = block(
            C, C, kernel_size=3, dilation=3, scale=model_scale, pool=3
        )
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        if self.context:
            attn_input = 1536 * 3
        else:
            attn_input = 1536
        print("self.encoder_type", self.encoder_type)
        if self.encoder_type == "ECA":
            attn_output = 1536
        elif self.encoder_type == "ASP":
            attn_output = 1
        else:
            raise ValueError("Undefined encoder")

        #############################################
        attention_output = nOut

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)

        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

        self.mp3 = nn.MaxPool1d(3)

    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """

        with torch.cuda.amp.autocast(enabled=False):
            x = self.preprocess(x)
            x = torch.abs(self.conv1(x))
            if self.log_sinc:
                x = torch.log(x + 1e-6)
            if self.norm_sinc == "mean":
                x = x - torch.mean(x, dim=-1, keepdim=True)
            elif self.norm_sinc == "mean_std":
                m = torch.mean(x, dim=-1, keepdim=True)
                s = torch.std(x, dim=-1, keepdim=True)
                s[s < 0.001] = 0.001
                x = (x - m) / s

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(self.mp3(x1) + x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        # t = x.size()[-1]

        # if self.context:
        #     global_x = torch.cat(
        #         (
        #             x,
        #             torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
        #             torch.sqrt(
        #                 torch.var(x, dim=2, keepdim=True).clamp(
        #                     min=1e-4, max=1e4
        #                 )
        #             ).repeat(1, 1, t),
        #         ),
        #         dim=1,
        #     )
        # else:
        #     global_x = x

        # w = self.attention(global_x)

        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt(
        #     (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        # )

        # x = torch.cat((mu, sg), 1)

        # x = self.bn5(x)

        # x = self.fc6(x)
        # print ("x.size():", x.size())   

        # if self.out_bn:
        #     x = self.bn6(x)

        return x


# # From StarGAN
# class ReconstructorVC(nn.Module):
#     """
#     Decoder Module.
#     """
#     def __init__(self, in_ch, n_spk, style_dim, mid_ch, class_dim, normtype='IN', src_conditioning=False):
#         # in_ch,    n_spk, z_ch, mid_ch, s_ch
#         # num_mels, n_spk, zdim, hdim,   sdim(class_dim)
#         # num_mels, # of speakers, dimension of bottleneck layer in generator, dimension of middle layers in generator, dimension of speaker embedding
#         super(ReconstructorVC, self).__init__()
#         self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(n_spk, class_dim))))

#         # Convolution, which is done in encoder
#         # self.le1 = md_starGAN.ConvGLU1D(in_ch+add_ch, mid_ch, 9, 1, normtype)
#         # self.le2 = md_starGAN.ConvGLU1D(mid_ch+add_ch, mid_ch, 8, 2, normtype)
#         # self.le3 = md_starGAN.ConvGLU1D(mid_ch+add_ch, mid_ch, 8, 2, normtype)
#         # self.le4 = md_starGAN.ConvGLU1D(mid_ch+add_ch, mid_ch, 5, 1, normtype)
#         # self.le5 = md_starGAN.ConvGLU1D(mid_ch+add_ch, z_ch, 5, 1, normtype)

#         # Deconvolution
#         self.le6 = md_starGAN.DeconvGLU1D(style_dim+class_dim, mid_ch, 5, 1, normtype)
#         self.le7 = md_starGAN.DeconvGLU1D(mid_ch+class_dim, 2*mid_ch, 5, 1, normtype)
#         self.le8 = md_starGAN.DeconvGLU1D(2*mid_ch+class_dim, 4*mid_ch, 5, 1, normtype)
#         self.le9 = md_starGAN.DeconvGLU1D(4*mid_ch+class_dim, 8*mid_ch, 5, 1, normtype)
#         self.le10 = nn.Conv1d(8*mid_ch+class_dim, 320, 9, stride=1, padding=(9-1)//2)

#     def forward(self, style_emb, class_label):
#         # Get class dim
#         class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
#         # 2. Convolution
#         # class_emb = class_emb.unsqueeze(-1)
#         # print(class_emb.size())
#         # class_emb = class_emb.repeat(1, 1, 100)
#         # x = torch.cat((style_emb, class_emb), dim=1)
        
#         # print("In reconstructor, style_emb.size():", style_emb.size())
#         # print("In reconstructor, class_emb.size():", class_emb.size())
#         x = md_starGAN.concat_dim1(style_emb,class_emb)
#         # print("After concat_dim1, x.size():", x.size())
#         x = self.le6(x)
#         # print("After le6, x.size():", x.size())
#         x = md_starGAN.concat_dim1(x,class_emb) 
#         # print("After concat_dim1, x.size():", x.size())
#         x = self.le7(x)
#         # print("After le7, x.size():", x.size()) 
#         x = md_starGAN.concat_dim1(x,class_emb)
#         # print("After concat_dim1, x.size():", x.size())
#         x = self.le8(x)
#         # print("After le8, x.size():", x.size())
#         x = md_starGAN.concat_dim1(x,class_emb)
#         x = self.le9(x)
#         # print("After le9, x.size():", x.size())
#         x = md_starGAN.concat_dim1(x,class_emb)
#         x = self.le10(x)
#         # print("After le10, x.size():", x.size())
#         return x

# From ParallelWaveGAN
class ReconstructorVC(nn.Module):
    """
    Decoder Module.
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=30,
        stacks=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        aux_context_window=2,
        dropout=0.0,
        bias=True,
        use_weight_norm=True,
        use_causal_conv=False,
        upsample_conditional_features=True,
        upsample_net="ConvInUpsampleNetwork",
        upsample_params={"upsample_scales": [4, 4, 4, 4]},
    ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super(ReconstructorVC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define conv + upsampling network
        if upsample_conditional_features:
            upsample_params.update(
                {
                    "use_causal_conv": use_causal_conv,
                }
            )
            if upsample_net == "MelGANGenerator":
                print("Do not support MelGANGenerator!")
                exit()
                # assert aux_context_window == 0
                # upsample_params.update(
                #     {
                #         "use_weight_norm": False,  # not to apply twice
                #         "use_final_nonlinear_activation": False,
                #     }
                # )
                # self.upsample_net = getattr(models, upsample_net)(**upsample_params)
            else:
                if upsample_net == "ConvInUpsampleNetwork":
                    upsample_params.update(
                        {
                            "aux_channels": aux_channels,
                            "aux_context_window": aux_context_window,
                        }
                    )
                self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            self.upsample_factor = np.prod(upsample_params["upsample_scales"])
        else:
            self.upsample_net = None
            self.upsample_factor = 1

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, z, c):
        """Calculate forward propagation.

        Args:
            z (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        # perform upsampling
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == z.size(-1)

        # encode to hidden representation
        x = self.first_conv(z)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, c=None, x=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(
                next(self.parameters()).device
            )
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            if normalize_before:
                c = (c - self.mean) / self.scale
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)

class DiscriminatorVC(nn.Module):
    """
    Discriminator Module.
    """
    def __init__(self):
        super(DiscriminatorVC, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=314496, out_features=2, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        # 1. Convolution
        # print("Before conv_blocks, x.size():", x.size())
        x = x.unsqueeze(1)
        x = self._conv_blocks(x)
        # print("After conv_blocks, x.size():", x.size())
        # 2. FC
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        # print("After view, x.size():", x.size())
        x = self._fc(x)
        # print("After fc, x.size():", x.size())
        # Return
        return x

# ----------------------------------------------------------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------------------------------------------------------


class EncoderShapes(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderShapes, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorShapes(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorShapes, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=8, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 4
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=8, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 5
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x


# ----------------------------------------------------------------------------------------------------------------------
# Sprites
# ----------------------------------------------------------------------------------------------------------------------

class EncoderSprites(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderSprites, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=512, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorSprites(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorSprites, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=8, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 4
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x

# ----------------------------------------------------------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------------------------------------------------------

class EncoderSpeech(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderSpeech, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorSpeech(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorSpeech, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x


class DiscriminatorSpeech(nn.Module):
    """
    Discriminator Module.
    """
    def __init__(self):
        super(DiscriminatorSpeech, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=512, out_features=2, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        # 1. Convolution
        x = self._conv_blocks(x)
        # 2. FC
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self._fc(x)
        # Return
        return x