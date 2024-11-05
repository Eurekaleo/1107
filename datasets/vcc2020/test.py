import os
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset

# class VCC2020Dataset(Dataset):
#     """
#     VCC2020 dataset for English-only source speakers.
#     """
#     def __init__(self): # def __init__(self, data_folder, transforms):
#         self.data_folder = "/home/cuizhouying/CS4347/CS4347/datasets/vcc2020" # self.data_folder = data_folder # 没用config直接把地址放在里面了
#         # self.transforms = transforms
#         self.wav_files = self._load_wav_files()

#     def _load_wav_files(self):
#         wav_files = []
        
#         srcspks = ["SEF1", "SEF2", "SEM1", "SEM2"]
#         trgspks_task1 = ["TEF1", "TEF2", "TEM1", "TEM2"] # Only Task1: English 只用了srcspks和Task1里的一些，如果需要还可以加上task2的
#         speakers = srcspks + trgspks_task1

#         for spk in speakers:
#             spk_folder = os.path.join(self.data_folder, spk)
#             if os.path.isdir(spk_folder):
#                 files = [f for f in os.listdir(spk_folder) if f.endswith(".wav")]
#                 files.sort()  
#                 for file_name in files:
#                     wav_files.append(os.path.join(spk_folder, file_name))
#         return wav_files

#     def __len__(self):
#         return len(self.wav_files)

#     def __getitem__(self, idx):
#         wav_path = self.wav_files[idx]
#         # print(wav_path)
#         signal, fs = sf.read(wav_path)
#         signal = signal / np.max(np.abs(signal)) 
#         signal = torch.from_numpy(signal).float() # 只是去了窗，没用mel spectrum，其他还是一样

#         label = os.path.basename(os.path.dirname(wav_path)) # 这里label直接用人名了，还没有编码啥的

#         return signal, label
    
class VCC2020Dataset(Dataset):
    """
    VCC2020 dataset for English-only source speakers with optional data augmentation.
    """
    def __init__(self, data_folder="/home/cuizhouying/CS4347/CS4347/datasets/vcc2020", transforms=None):
        self.data_folder = data_folder
        self.transforms = transforms  # List of transformations (e.g., [NoiseAug(), RIRAug()])
        self.wav_files = self._load_wav_files()

    def _load_wav_files(self):
        wav_files = []
        
        srcspks = ["SEF1", "SEF2", "SEM1", "SEM2"]
        trgspks_task1 = ["TEF1", "TEF2", "TEM1", "TEM2"]  # Only Task1: English
        speakers = srcspks + trgspks_task1

        for spk in speakers:
            spk_folder = os.path.join(self.data_folder, spk)
            if os.path.isdir(spk_folder):
                files = [f for f in os.listdir(spk_folder) if f.endswith(".wav")]
                files.sort()  
                for file_name in files:
                    wav_files.append(os.path.join(spk_folder, file_name))
        return wav_files

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        signal, fs = sf.read(wav_path)
        signal = signal / np.max(np.abs(signal)) 
        signal = torch.from_numpy(signal).float()

        # Apply transformations if any
        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)

        # label = os.path.basename(os.path.dirname(wav_path))  # Use speaker name as label
        speaker_to_id = {
            "SEF1": 0,
            "SEF2": 1,
            "SEM1": 2,
            "SEM2": 3,
            "TEF1": 4,
            "TEF2": 5,
            "TEM1": 6,
            "TEM2": 7
        }
        label = speaker_to_id[os.path.basename(os.path.dirname(wav_path))]

        return signal, label

# data_folder = "/home/cuizhouying/CS4347/CS4347/datasets/vcc2020"
dataset = VCC2020Dataset()

signal, label = dataset[0]
print("original:", "\n","signal.size():", signal.size(), "\n", "label:", label, "\n", "signal:", signal)

import torchaudio
import glob
import scipy

## Noise augmentation ----------------------------------------------------------
import torch
import torchaudio
import numpy as np
import glob

class NoiseAug(object):
    def __init__(self, noise_dir='/data1/cuizhouying/dataset/musan/noise/', prob=1):
        self.prob = prob
        self.noises = glob.glob(noise_dir + '/*/*.wav')
        
    def __call__(self, x):
        if np.random.uniform() < self.prob:
            noise = np.random.choice(self.noises)
            print("noise:", noise)
            n = torchaudio.load(noise)[0][0]
            
            if len(n) < len(x):
                n = torch.nn.functional.pad(n, (0, len(x) - len(n)), value=0)
            elif len(n) > len(x):
                t0 = np.random.randint(0, len(n) - len(x))
                n = n[t0:t0 + len(x)]
            
            # Compute power and scale noise
            p_x = x.std() ** 2
            p_n = n.std() ** 2
            snr = np.random.uniform(5, 15)
            scaling_factor = (p_x / p_n).sqrt() * torch.pow(torch.tensor(10.0), -snr / 20)
            n = n * scaling_factor
            
            # Add noise to the input signal
            x = x + n

        return x

    
## RIR augmentation ------------------------------------------------------------
import torch
import torchaudio
import numpy as np
import glob
import scipy.signal

class RIRAug(object):
    def __init__(self, rir_dir='/data1/cuizhouying/dataset/RIRS_NOISES/simulated_rirs/smallroom/', prob=1):
        self.prob = prob
        self.rirs = glob.glob(rir_dir + '/*/*.wav') 

    def __call__(self, x):
        if np.random.uniform() < self.prob:
            n = len(x)
            
            noise = np.random.choice(self.rirs)
            print("noise:", noise)
            rir = torchaudio.load(noise)[0][0]
            
            # rir = torchaudio.load(np.random.choice(self.rirs))[0][0]
            rir = rir.numpy()
            rir = rir / np.max(np.abs(rir))
            x = scipy.signal.convolve(x.numpy(), rir, mode='full')  # Use 'full' mode for convolution
            t0 = np.argmax(np.abs(rir))
            x = x[t0:t0+n]  # Truncate to the original length
            x = torch.from_numpy(x).float()  # Convert back to torch.Tensor

        return x

dataset_musan = VCC2020Dataset(transforms=[NoiseAug()])
signal, label = dataset_musan[0]
torchaudio.save("/home/cuizhouying/CS4347/CS4347/datasets/vcc2020/musan_augmented.wav", signal.unsqueeze(0), sample_rate=16000)
print("musan:", "\n","signal.size():", signal.size(), "\n", "label:", label, "\n", "signal:", signal)

dataset_rir = VCC2020Dataset(transforms=[RIRAug()])
signal, label = dataset_rir[0]
torchaudio.save("/home/cuizhouying/CS4347/CS4347/datasets/vcc2020/rir_augmented.wav", signal.unsqueeze(0), sample_rate=16000)
print("RIR:", "\n","signal.size():", signal.size(), "\n", "label:", label, "\n", "signal:", signal)