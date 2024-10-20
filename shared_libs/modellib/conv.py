
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import module_starGAN as md_starGAN
from .dnn_models import SincNet as CNN 
from .dnn_models import MLP
# import module_starGAN as md_starGAN

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
        style_emb = self._fc_style(style_emb)
        class_emb = self._fc_class(class_emb)
        return self._fc_blocks(torch.cat([style_emb, class_emb], dim=1))

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
    def __init__(self, nz):
        super(EncoderTIMIT, self).__init__()

        # wlen = 3200 # 16000*200/1000 = 3200
        fs = 16000
        # cnn_N_filt = [20, 10, 10] # cnn_N_filt = [80, 60, 60]
        cnn_N_filt = [80, 60, 60]
        # cnn_len_filt = [32, 8, 6] # cnn_len_filt = [251, 5, 5]
        cnn_len_filt = [251, 5, 5]
        # cnn_max_pool_len = [2, 2, 2] # cnn_max_pool_len = [3, 3, 3]
        cnn_max_pool_len = [3, 3, 3]
        cnn_use_laynorm_inp = True
        cnn_use_batchnorm_inp = False
        cnn_use_laynorm = [True, True, True]
        cnn_use_batchnorm = [False, False, False]
        cnn_act = ['leaky_relu', 'leaky_relu', 'leaky_relu']
        cnn_drop = [0.0, 0.0, 0.0]
        
        self.cnn_arch = {
            'input_dim': 3200, # mel: 80
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop
        }
                
        self.cnn_net = CNN(self.cnn_arch)

        fc_lay = [2048, 2048, nz] # fc_lay = [2048, 2048, 2048]
        fc_drop = [0.0, 0.0, 0.0]
        fc_use_laynorm_inp = True
        fc_use_batchnorm_inp = False
        fc_use_batchnorm = [True, True, True]
        fc_use_laynorm = [False, False, False]
        fc_act = ['leaky_relu', 'leaky_relu', 'leaky_relu']

        self.dnn1_arch = {
            'input_dim':self.cnn_net.out_dim,
            'fc_lay': fc_lay,
            'fc_drop': fc_drop,
            'fc_use_batchnorm': fc_use_batchnorm,
            'fc_use_laynorm': fc_use_laynorm,
            'fc_use_laynorm_inp': fc_use_laynorm_inp,
            'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
            'fc_act': fc_act
        }

        
        self.dnn1_net = MLP(self.dnn1_arch)

        self.d_vector_dim = nz

    def forward(self, signal_chunks):
        
        batch_size, num_frames, features = signal_chunks.size()  # [64, 512, 3200] (batch_size, frames_nums, features)
        signal_chunks = signal_chunks.view(batch_size * num_frames, features)  # (batch_size*num_frames, features)
        print("signal_chunks:", signal_chunks) 
        print("signal_chunks_size:", signal_chunks.size()) # [32768, 3200] (3200, cnn_arch input dim, from TIMITDataset)
       
        cnn_out = self.cnn_net(signal_chunks)
        print("cnn_out:",cnn_out)
        print("cnn_out_size:", cnn_out.size()) # 32768, 60 (10, dnn1_arch input dim)
        
        d_vectors = self.dnn1_net(cnn_out)
        print("d_vectors:", d_vectors)
        print("d_vectors_size:", d_vectors.size()) # 32768, 16 (16, d_vectors size, configs: style_dim / class_dim )
                    
        d_vectors = d_vectors.view(batch_size, num_frames, self.d_vector_dim) # [batch_size, num_frames, d_vector_dim]
        print("d_vectors:", d_vectors)
        print("d_vectors_size:", d_vectors.size()) # 64, 512, 16        
        
        # batch_size, num_frames, mel_features = signal_chunks.size()  # [64, 512, 80] (7, mean in datasets)
        # signal_chunks = signal_chunks.view(batch_size * num_frames, mel_features)  # [batch_size*num_frames, mel_features*extra_dim]
        # print("signal_chunks:", signal_chunks) 
        # print("signal_chunks_size:", signal_chunks.size()) # 32768, 80 (80, cnn_arch input dim)
       
        # cnn_out = self.cnn_net(signal_chunks)
        # print("cnn_out:",cnn_out)
        # print("cnn_out_size:", cnn_out.size()) # 32768, 10 (10, dnn1_arch input dim)
        
        # d_vectors = self.dnn1_net(cnn_out)
        # print("d_vectors:", d_vectors)
        # print("d_vectors_size:", d_vectors.size()) # 32768, 16 (16, d_vectors size, configs: style_dim / class_dim )
                    
        # d_vectors = d_vectors.view(batch_size, num_frames, self.d_vector_dim) # [batch_size, num_frames, d_vector_dim]
        # print("d_vectors:", d_vectors)
        # print("d_vectors_size:", d_vectors.size()) # 64, 512, 16         
        
        return d_vectors

class ReconstructorVC(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, in_ch, n_spk, style_dim, mid_ch, class_dim, normtype='IN', src_conditioning=False):
        # in_ch,    n_spk, z_ch, mid_ch, s_ch
        # num_mels, n_spk, zdim, hdim,   sdim(class_dim)
        # num_mels, # of speakers, dimension of bottleneck layer in generator, dimension of middle layers in generator, dimension of speaker embedding
        super(ReconstructorVC, self).__init__()
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(n_spk, class_dim))))

        # Convolution, which is done in encoder
        # self.le1 = md_starGAN.ConvGLU1D(in_ch+add_ch, mid_ch, 9, 1, normtype)
        # self.le2 = md_starGAN.ConvGLU1D(mid_ch+add_ch, mid_ch, 8, 2, normtype)
        # self.le3 = md_starGAN.ConvGLU1D(mid_ch+add_ch, mid_ch, 8, 2, normtype)
        # self.le4 = md_starGAN.ConvGLU1D(mid_ch+add_ch, mid_ch, 5, 1, normtype)
        # self.le5 = md_starGAN.ConvGLU1D(mid_ch+add_ch, z_ch, 5, 1, normtype)

        # Deconvolution
        self.le6 = md_starGAN.DeconvGLU1D(style_dim+class_dim, mid_ch, 5, 1, normtype)
        self.le7 = md_starGAN.DeconvGLU1D(mid_ch+class_dim, mid_ch, 5, 1, normtype)
        self.le8 = md_starGAN.DeconvGLU1D(mid_ch+class_dim, mid_ch, 8, 2, normtype)
        self.le9 = md_starGAN.DeconvGLU1D(mid_ch+class_dim, mid_ch, 8, 2, normtype)
        self.le10 = nn.Conv1d(mid_ch+class_dim, in_ch, 9, stride=1, padding=(9-1)//2)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 2. Convolution
        # class_emb = class_emb.unsqueeze(-1)
        # print(class_emb.size())
        # class_emb = class_emb.repeat(1, 1, 100)
        # x = torch.cat((style_emb, class_emb), dim=1)
        
        # print(x.size())
        x = md_starGAN.concat_dim1(style_emb,class_emb)
        x = self.le6(x)
        x = md_starGAN.concat_dim1(x,class_emb)
        x = self.le7(x)
        x = md_starGAN.concat_dim1(x,class_emb)
        x = self.le8(x)
        x = md_starGAN.concat_dim1(x,class_emb)
        x = self.le9(x)
        x = md_starGAN.concat_dim1(x,class_emb)
        x = self.le10(x)

        print(x.size())
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