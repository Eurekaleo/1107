
# Dataset Download Link:
# musan: https://www.openslr.org/17/
# rir: https://www.openslr.org/28/

# use the same noise? use which noise? probability?

# ----------------------------------------------------------------------------------------------------------------------
# Musan Noise augmentation, can also try Music/Speech
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# RIR augmentation
# ----------------------------------------------------------------------------------------------------------------------    
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