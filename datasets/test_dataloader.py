# Import necessary libraries
import torch
# from methods.disenib_conv_VC.dataloader import generate_data
# from CS4347.methods.disenib_conv_VC.dataloader import generate_data
import sys
sys.path.append('/home/cuizhouying/CS4347/CS4347')

from methods.disenib_conv_VC.dataloader import generate_data

class Config:
    class Args:
        batch_size = 4  
        dataset_drop_last = False
        dataset_shuffle = True
        dataset_num_threads = 2  
        fs = 16000  
        train_list = "/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/data_lists/TIMIT_train.scp"
        labels_list = "/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/data_lists/TIMIT_labels.npy"
        dataset_path = "/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/output"

    args = Args()


cfg = Config()

data = generate_data(cfg)


train_loader = data['train_data']

import soundfile as sf
for idx, (signals, labels) in enumerate(train_loader):
    print(f"idx: {idx}")  
    if idx == 0:  
        for i, signal in enumerate(signals):
            signal_np = signal.numpy()
            save_path = f"audio_batch_1_signal_{i + 1}.wav"
            sf.write(save_path, signal_np, 16000)
            print(f"Saved {save_path}")
        break



