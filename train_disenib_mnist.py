
from methods.disenib_conv_mnist.config import ConfigTrain
from methods.disenib_conv_mnist.disen_ib import DisenIB
from methods.disenib_conv_mnist.dataloader import generate_data


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    model = DisenIB(cfg=cfg)
    dataloader = generate_data(cfg)
    # 3. Train
    model.train_parameters(**dataloader)
