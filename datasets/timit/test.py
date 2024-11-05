import os
import torch
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
import torchaudio

class TIMITDataset(Dataset):
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

        # Load the audio signal
        signal, fs = sf.read(wav_path)
        signal = signal / np.max(np.abs(signal))  # Normalize the signal
        signal = torch.from_numpy(signal).float()

        label = self.labels[self.wav_files[idx]]  # Load the label

        return signal, label
    
train_dataset = TIMITDataset(
    phase='train',
    data_list="/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/data_lists/TIMIT_train.scp",
    label_file="/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/data_lists/TIMIT_labels.npy",
    data_folder="/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/output"
)
# for n in range(20):
#     sample_signal, sample_label = train_dataset[n]  # Access the first sample

#     print("Signal shape:", sample_signal.shape)
#     print("Label:", sample_label)

max_length = 0
for idx in range(len(train_dataset)):
    sample_signal, _ = train_dataset[idx]
    max_length = max(max_length, sample_signal.shape[0])

print(max_length)
