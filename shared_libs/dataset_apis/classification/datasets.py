
import os
import numpy as np
import torchvision
from torch.utils.data import Dataset
from shared_libs.dataset_apis.classification import utils
from shared_libs.dataset_apis.classification import transforms as custom_transforms
from shared_libs.dataset_apis import __DATA_ROOT__
import torchaudio


########################################################################################################################
# Dataset
########################################################################################################################

class BaseClassification(Dataset):
    """
    Base class for dataset for classification.
    """
    @property
    def classes(self):
        """
        :return: A list, whose i-th element is the name of the i-th category.
        """
        raise NotImplementedError

    @property
    def class_to_idx(self):
        """
        :return: A dict, where dict[key] is the category index corresponding to the category 'key'.
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """
        :return: A integer indicating number of categories.
        """
        raise NotImplementedError

    @property
    def class_counter(self):
        """
        :return: A list, whose i-th element equals to the total sample number of the i-th category.
        """
        raise NotImplementedError

    @property
    def sample_indices(self):
        """
        :return: A list, whose i-th element is a numpy.array containing sample indices of the i-th category.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Should return x, y, where y is the class label.
        :param index:
        :return:
        """
        raise NotImplementedError


########################################################################################################################
# Instances
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MNIST & FashionMNIST
# ----------------------------------------------------------------------------------------------------------------------

def mnist_paths(name):
    assert name in ['mnist', 'fashion_mnist']
    return {
        'train': (os.path.join(__DATA_ROOT__, "%s/train-images.idx3-ubyte" % name),
                  os.path.join(__DATA_ROOT__, "%s/train-labels.idx1-ubyte" % name)),
        'test': (os.path.join(__DATA_ROOT__, "%s/t10k-images.idx3-ubyte" % name),
                 os.path.join(__DATA_ROOT__, "%s/t10k-labels.idx1-ubyte" % name))}

class MNIST(BaseClassification):
    """
    MNIST dataset.
    """
    def __init__(self, images_path, labels_path, transforms=None):
        # Member variables
        self._transforms = transforms
        # Load from file
        # (1) Data & label
        self._dataset = utils.decode_idx3_ubyte(images_path).astype('float32')[:, :, :, np.newaxis] / 255.0
        self._label = utils.decode_idx1_ubyte(labels_path).astype('int64')
        # (2) Samples per category
        self._sample_indices = [np.argwhere(self._label == label)[:, 0].tolist() for label in range(self.num_classes)]
        # (3) Class counter
        self._class_counter = [len(samples) for samples in self._sample_indices]

    @property
    def num_classes(self):
        return len(set(self._label))

    @property
    def class_counter(self):
        return self._class_counter

    @property
    def sample_indices(self):
        return self._sample_indices

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        # 1. Get image & label
        image, label = self._dataset[index], self._label[index]
        # 2. Transform
        if self._transforms:
            image = self._transforms(image)
        # Return
        return image, label

class FlattenMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, ), (0.5, )),
            custom_transforms.Flatten()]
        ) if 'transforms' not in kwargs.keys() else kwargs['transforms']
        # Init
        super(FlattenMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)

class ImageMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        if 'transforms' in kwargs.keys():
            transforms = kwargs['transforms']
        else:
            transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, ), (0.5, ))]
            if 'to32x32' in kwargs.keys() and kwargs['to32x32']: transforms = [custom_transforms.To32x32()] + transforms
            transforms = torchvision.transforms.Compose(transforms)
        # Init
        super(ImageMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


# ----------------------------------------------------------------------------------------------------------------------
# TIMIT
# ----------------------------------------------------------------------------------------------------------------------

import os
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class TIMITDataset(BaseClassification):
    """
    TIMIT dataset
    """
    def __init__(self, phase, data_list, label_file, data_folder, transforms=None):
        assert phase in ['train', 'test']
        self.phase = phase
        self.data_folder = data_folder
        self.transforms = transforms
        self.wav_files = self._load_data_list(data_list)
        self.labels = self._load_labels(label_file)

    def _load_data_list(self, data_list):
        with open(data_list, 'r') as file:
            wav_files = [line.strip() for line in file.readlines()]
        return wav_files

    def _load_labels(self, label_file):
        data = np.load(label_file, allow_pickle=True)
        return data.item() 

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        dir_path, file_name = os.path.split(self.wav_files[idx])
        path_parts = dir_path.split(os.sep)
        path_parts = [part.upper() if part.islower() else part for part in path_parts]
        dir_path = os.path.join(*path_parts)
        wav_path = os.path.join(self.data_folder, dir_path, file_name)

        signal, fs = sf.read(wav_path)
        signal = signal / np.max(np.abs(signal))  
        signal = torch.from_numpy(signal).float()

        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)

        label = self.labels[self.wav_files[idx]]  
        # print("signal:", signal.size(),"label:", label)
        return signal, label

# ----------------------------------------------------------------------------------------------------------------------
# VCC2020
# ----------------------------------------------------------------------------------------------------------------------

class VCC2020Dataset(Dataset):
    """
    VCC2020 dataset for English-only source speakers with optional data augmentation.
    """
    def __init__(self, phase, data_folder="/home/cuizhouying/CS4347/CS4347/datasets/vcc2020", transforms=None):
        assert phase in ['train', 'test']
        self.phase = phase
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
        
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
        
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

# ----------------------------------------------------------------------------------------------------------------------
# VCTK
# ----------------------------------------------------------------------------------------------------------------------
class VCTKDataset(Dataset):
    """
    VCTK dataset for loading audio signals and speaker labels.
    """
    def __init__(self, phase, data_folder="/data1/cuizhouying/dataset/VCTK-Corpus", transforms=None):
        assert phase in ['train', 'test']
        self.phase = phase
        self.data_folder = data_folder
        self.transforms = transforms  # 
        self.wav_files = self._load_wav_files()

    def _load_wav_files(self):
        wav_files = []
        wav_dir = os.path.join(self.data_folder, "wav48")

        speakers = [d for d in os.listdir(wav_dir) if os.path.isdir(os.path.join(wav_dir, d))]
        speakers.sort()  

        for spk in speakers:
            spk_folder = os.path.join(wav_dir, spk)
            files = [f for f in os.listdir(spk_folder) if f.endswith(".wav")]
            files.sort()  
            for file_name in files:
                wav_files.append(os.path.join(spk_folder, file_name))
        
        return wav_files

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        # print(wav_path)
        signal, fs = sf.read(wav_path)
        signal = signal / np.max(np.abs(signal))  
        signal = torch.from_numpy(signal).float()

        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)

        label = os.path.basename(os.path.dirname(wav_path)).lstrip('p')

        return signal, int(label)  