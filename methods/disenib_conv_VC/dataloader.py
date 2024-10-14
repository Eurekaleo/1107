
from torch.utils.data import DataLoader
from shared_libs.custom_packages.custom_pytorch.operations import DataCycle
from shared_libs.dataset_apis.classification.datasets import TIMITDataset
from torchaudio.transforms import MelSpectrogram

# def collate_fn(batch):
#     signals, labels = zip(*batch)

#     max_length = max([signal.size(1) for signal in signals])

#     padded_signals = []
#     for signal in signals:
#         if signal.size(1) < max_length:
#             padding = torch.zeros((signal.size(0), max_length - signal.size(1)))
#             signal = torch.cat((signal, padding), dim=1)
#         padded_signals.append(signal)

#     signals = torch.stack(padded_signals, dim=0)
#     labels = torch.tensor(labels)
#     print("collate_fn:", signals.size())
#     return signals, labels

def generate_data(cfg):
    """
    Generate DataLoader instances for the TIMIT dataset.
    """
    def _get_dataloader(_dataset, **kwargs):
        return DataLoader(
            dataset=_dataset,
            batch_size=cfg.args.batch_size if 'batch_size' not in kwargs.keys() else kwargs['batch_size'],
            drop_last=cfg.args.dataset_drop_last,
            shuffle=cfg.args.dataset_shuffle,
            num_workers=cfg.args.dataset_num_threads,
        )

    audio_transforms = MelSpectrogram(
        sample_rate=cfg.args.fs,
        n_mels=cfg.args.num_mels,
        n_fft=cfg.args.n_fft,
        hop_length=cfg.args.hop_length
    )

    train_dataset = TIMITDataset(
        phase='train',
        data_list=cfg.args.train_list,
        data_folder=cfg.args.dataset_path,
        label_file = cfg.args.labels_list,
        cw_len=cfg.args.cw_len,  
        cw_shift=cfg.args.cw_shift,
        transforms=audio_transforms
    )

    train_est_dataset = TIMITDataset(
        phase='train',
        data_list=cfg.args.train_list,
        data_folder=cfg.args.dataset_path,
        label_file = cfg.args.labels_list,
        cw_len=cfg.args.cw_len,  
        cw_shift=cfg.args.cw_shift,
        transforms=audio_transforms
    )

    # test_dataset = TIMITDataset(
    #     phase='test',
    #     data_list=cfg.args.test_list,
    #     data_folder=cfg.args.dataset_path,
    #     label_file = cfg.args.labels_list,
    #     cw_len=cfg.args.cw_len,  
    #     cw_shift=cfg.args.cw_shift,
    #     transforms=audio_transforms
    # )

    return {
        'train_data': DataCycle(_get_dataloader(train_dataset)),
        'train_est_data': DataCycle(_get_dataloader(train_est_dataset)),
        # 'test_data': DataCycle(_get_dataloader(test_dataset))  
    }
    
