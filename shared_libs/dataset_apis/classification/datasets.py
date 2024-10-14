
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
    def __init__(self, phase, data_list, label_file, data_folder, cw_len, cw_shift, transforms=None):
        assert phase in ['train', 'test']
        self.phase = phase
        self.data_folder = data_folder
        self.transforms = transforms
        self.wav_files = self._load_data_list(data_list)
        self.labels = self._load_labels(label_file)
        self.cw_len = cw_len
        self.cw_shift = cw_shift
        self.max_frames = 512 # add: max frame 512

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
        # epsilon = 1e-8
        # signal = signal / (np.max(np.abs(signal)) + epsilon)
        signal = signal / np.max(np.abs(signal))
        signal = torch.from_numpy(signal).float()

        # segment signal into frames
        wlen = int(fs * self.cw_len / 1000)  
        wshift = int(fs * self.cw_shift / 1000) 

        chunks = []
        beg_samp = 0
        end_samp = wlen
        while end_samp <= signal.shape[0]:
            chunk = signal[beg_samp:end_samp]
            chunks.append(chunk)
            beg_samp += wshift
            end_samp = beg_samp + wlen

        # padding the last chunk if necessary
        if beg_samp < signal.shape[0]:
            chunk = signal[beg_samp:]
            padding = torch.zeros(wlen - chunk.shape[0])
            chunk = torch.cat((chunk, padding))
            chunks.append(chunk)

        signal_chunks = torch.stack(chunks)

        # padding -> 512 frames # 512, 3200 (same as cnn input dim)
        if signal_chunks.size(0) < self.max_frames:
            padding_frames = torch.zeros(self.max_frames - signal_chunks.size(0), wlen)
            signal_chunks = torch.cat((signal_chunks, padding_frames), dim=0)

        elif signal_chunks.size(0) > self.max_frames:
            signal_chunks = signal_chunks[:self.max_frames] 

        # mel_features: 
        # if self.transforms:
        #     signal_chunks = self.transforms(signal_chunks)
        # signal_chunks = torch.mean(signal_chunks, dim=-1) # 512, 80, 7 -> 512, 80
        
        label = self.labels[self.wav_files[idx]] # integer
        
        # label = F.one_hot(torch.tensor(label), num_classes=630) # one-hot?
        print(signal_chunks.size(), label) 

        return signal_chunks, label

        # signal, fs = sf.read(wav_path)
        # signal = signal / np.max(np.abs(signal))  
        # signal = torch.from_numpy(signal).float()

        # if self.transforms:
        #     signal = self.transforms(signal)

        # label = self.labels[self.wav_files[idx]]
        # print("TIMIT:", signal.size())
        
        # return signal, label


