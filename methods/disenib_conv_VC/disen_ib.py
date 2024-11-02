
import os
from shared_libs.modellib.conv import *
from shared_libs.utils.operations import *
from shared_libs.utils.criterions import *
from shared_libs.custom_packages.custom_io.logger import Logger
from shared_libs.utils.evaluations import vis_grid_disentangling
from shared_libs.custom_packages.custom_pytorch.base_models import IterativeBaseModel
from shared_libs.custom_packages.custom_basic.operations import fet_d, ValidContainer
from shared_libs.custom_packages.custom_basic.metrics import FreqCounter, TriggerPeriod, TriggerLambda
from shared_libs.custom_packages.custom_pytorch.operations import summarize_losses_and_backward, set_requires_grad
import numpy as np
from tqdm import tqdm

# class DisenIB(IterativeBaseModel):
#     """
#     Disentangled IB model.
#     """
#     def _build_architectures(self, **modules):
#         super(DisenIB, self)._build_architectures(
#             # Encoder, decoder, reconstructor, estimator
#             # Encoder: Enc_style for extracting style info (can use d-vector) & Enc_class for extracting class info (e.g. label for MNIST)
            
#             Enc_style=EncoderMNIST(self._cfg.args.style_dim), Enc_class=EncoderMNIST(self._cfg.args.class_dim),
#             # num_mels 设的80
#             Dec=Decoder(self._cfg.args.class_dim, self._cfg.args.num_classes),
#             Rec=ReconstructorVC(self._cfg.args.num_mels, self._cfg.args.num_classes, self._cfg.args.style_dim, self._cfg.args.mid_ch, self._cfg.args.class_dim),
#             Est=DensityEstimator(self._cfg.args.style_dim, self._cfg.args.class_dim),
#             # Discriminator for improving generated quality
#             Disc=DiscriminatorMNIST())

class DisenIB(IterativeBaseModel):
    """
    Disentangled IB model modified for TIMIT audio data.
    """
    def __init__(self, cfg):
        super(DisenIB, self).__init__(cfg)  # 调用父类的初始化方法
        self.global_iter = 0  # 初始化全局迭代计数器
        
    def _build_architectures(self, **modules):
        super(DisenIB, self)._build_architectures(
            # Encoder, decoder, reconstructor, estimator
            # Encoder: Enc_style for extracting style info (can use d-vector) & Enc_class for extracting class info (e.g. label for TIMIT)
            
            # Enc_style=EncoderTIMIT(self._cfg.args.style_dim),  # Updated for TIMIT, 16
            # Enc_class=EncoderTIMIT(self._cfg.args.class_dim),  # Updated for TIMIT, 16
            Enc_style=EncoderTIMIT(nOut = self._cfg.args.style_dim),  # Updated for RawNet3
            Enc_class=EncoderTIMIT(nOut =self._cfg.args.class_dim),  # Updated for RawNet3
            # num_mels set to 80
            Dec=Decoder(self._cfg.args.class_dim, self._cfg.args.num_classes),
            # Rec=ReconstructorVC(self._cfg.args.num_mels, self._cfg.args.num_classes, self._cfg.args.style_dim, self._cfg.args.mid_ch, self._cfg.args.class_dim),
            Rec=ReconstructorVC(),
            Est=DensityEstimator(self._cfg.args.style_dim, self._cfg.args.class_dim),
            # Discriminator (update as needed for TIMIT)
            Disc=DiscriminatorVC()
        )

    def _set_criterions(self):
        self._criterions['dec'] = CrossEntropyLoss(lmd=self._cfg.args.lambda_dec)
        self._criterions['rec'] = RecLoss(lmd=self._cfg.args.lambda_rec)
        self._criterions['est'] = EstLoss(radius=self._cfg.args.emb_radius)
        # Discriminator
        self._criterions['disc'] = GANLoss()

    def _set_optimizers(self):
        self._optimizers['main'] = torch.optim.Adam(
            list(self._Enc_style.parameters()) + list(self._Enc_class.parameters()) +
            list(self._Dec.parameters()) + list(self._Rec.parameters()),
            lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))
        self._optimizers['est'] = torch.optim.Adam(
            self._Est.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))
        # Discriminator
        self._optimizers['disc'] = torch.optim.Adam(
            self._Disc.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))

    def _set_meters(self, **kwargs):
        super(DisenIB, self)._set_meters()
        self._meters['counter_eval'] = FreqCounter(self._cfg.args.freq_step_eval)
        self._meters['trigger_est'] = TriggerLambda(lambda n: n >= self._cfg.args.est_thr)
        self._meters['trigger_est_style_optimize'] = TriggerPeriod(
            period=self._cfg.args.est_style_optimize + 1, area=self._cfg.args.est_style_optimize)
        self._meters['trigger_disc'] = TriggerLambda(lambda n: n >= self._cfg.args.disc_thr)

    # ------------------------------------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------------------------------------

    def _deploy_batch_data(self, batch_data):
        audio, label = map(lambda x: x.to(self._cfg.args.device), batch_data)
        return audio.size(0), (audio, label)

    def _train_step(self, packs):
        # 获取 epoch 总数和当前 epoch
        current_epoch = self._meters['i']['epoch']
        steps = self._cfg.args.steps  # 假设你有 total_epochs 配置
        disc_acc = 0  # 初始化为 0 或其他默认值

        # tqdm 进度条设置
        tqdm_bar = tqdm(total=self._cfg.args.n_times_main, desc=f"Epoch {current_epoch + 1}/{steps}", unit="iter")

        ################################################################################################################
        # Main
        ################################################################################################################
        for i in range(self._cfg.args.n_times_main):
            tqdm_bar.update(1)  # 更新进度条
            # audios, label = self._fetch_batch_data() 
            
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=True)
            set_requires_grad([self._Disc, self._Est], requires_grad=False)
            self._optimizers['main'].zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            # Decoding & reconstruction
            # ----------------------------------------------------------------------------------------------------------
            # style_emb = torch.randn(self._cfg.args.batch_size, self._cfg.args.style_dim, 100).to(self._cfg.args.device) # 64, 16. 100
            # class_emb = torch.randn(self._cfg.args.batch_size, self._cfg.args.class_dim).to(self._cfg.args.device) # 64, 16 
            # label = torch.from_numpy(np.array([0] * self._cfg.args.batch_size)).to(self._cfg.args.device, dtype=torch.int64) # 64
            # audios = torch.randn(self._cfg.args.batch_size, self._cfg.args.num_mels, 400).to(self._cfg.args.device) # 64, 80, 400

            audios = torch.randn(self._cfg.args.batch_size, 16500).to(self._cfg.args.device)
            style_emb, class_emb = self._Enc_style(audios), self._Enc_class(audios)
            print("style_emb:", class_emb)
            print ("emb.size():", class_emb.size())
            print("Encoders are working!")
            
            # 1. Decoding: use class embedding(from encoder), to generate the label (speaker ID).
            # Optimized towards the ground truth label(speaker ID).
  
            dec_output = self._Dec(resampling(class_emb, self._cfg.args.class_std))
            # print(dec_output.size(), label.size())
            loss_dec = self._criterions['dec'](dec_output, label)            

            # 2. Reconstruction: use style embedding(from encoder) and ground truth label(speaker ID), to reconstruct an audio.
            # Optimized towards the target audio.
            rec_output = self._Rec(audios.unsqueeze(1), resampling(style_emb.transpose(1, 2), self._cfg.args.style_std))
            print("rec_output.size():", rec_output.size())  # (batch_size, 1, time_steps)
            print("Reconstructor is working!")
            rec_output = rec_output.transpose(1, 2) # (batch_size, time_steps, 1)
            # print("rec_output.size():", rec_output.size())
            # print("audios.size():", audios.size())

            loss_rec = self._criterions['rec'](rec_output, audios)
            # print("Successful Reconstruction!")
            # Backward
            summarize_losses_and_backward(loss_dec, loss_rec, retain_graph=True)
            # print("Successful Backward!")
            # ----------------------------------------------------------------------------------------------------------
            # Estimator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate output (batch*n_samples, ) & loss (1, ).
            est_output = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            # print("Successful Estimator!")
            crit_est = self._criterions['est'](
                output=est_output, emb=(style_emb, class_emb), mode='main',
                lmd={'loss_est': self._cfg.args.lambda_est, 'loss_wall': self._cfg.args.lambda_wall})
            # print("Successful Crit Estimator!")
            # Backward
            # 1> Density estimation
            if self._meters['trigger_est'].check(self._meters['i']['step']):
                if self._meters['trigger_est_style_optimize'].check():
                    set_requires_grad(self._Enc_class, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_class, requires_grad=True)
                else:
                    set_requires_grad(self._Enc_style, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_style, requires_grad=True)
            # 2> Embedding wall
            summarize_losses_and_backward(crit_est['loss_wall'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Discriminator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate loss
            disc_output = self._Disc(rec_output)
            crit_gen = self._criterions['disc'](disc_output, True, lmd=self._cfg.args.lambda_disc)
            # Backward
            if self._meters['trigger_disc'].check(self._meters['i']['step']):
                summarize_losses_and_backward(crit_gen['loss'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Update
            self._optimizers['main'].step()
            """ Saving """
            # 将关键信息更新到日志 packs 中
            packs['log'].update({
                'epoch': current_epoch,
                'iter': self.global_iter,  # 使用全局迭代计数器
                'batch': self._meters['i']['step'],  # 假设 step 记录了当前 batch 信息
                'loss_dec': loss_dec.item(),
                'loss_rec': loss_rec.item(),
                'est': crit_est['est'].item(),
                'disc_acc': disc_acc
            })

            # 增加全局迭代计数器
            self.global_iter += 1
        tqdm_bar.close()  # 结束 tqdm 进度条
        ################################################################################################################
        # Density Estimator
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_est):
            with self._meters['timers']('io'):
                audios, label = map(lambda _x: _x.to(self._cfg.args.device), next(self._data['train_est']))
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=False)
            set_requires_grad([self._Est], requires_grad=True)
            self._optimizers['est'].zero_grad()
            # 1. Get embedding
            style_emb, class_emb = self._Enc_style(audios).detach(), self._Enc_class(audios).detach()
            # 2. Get output (batch*n_samples, ) & loss (1, ).
            est_output_real = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='perm')
            est_output_fake = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            crit_est = self._criterions['est'](
                output_fake=est_output_fake, output_real=est_output_real, mode='est',
                lmd={'loss_real': 1.0, 'loss_fake': 1.0, 'loss_zc': self._cfg.args.lambda_est_zc})
            # Backward
            summarize_losses_and_backward(crit_est['loss_real'], crit_est['loss_fake'], crit_est['loss_zc'])
            # Update
            self._optimizers['est'].step()
            """ Saving """
            packs['log'].update({
                # Anchor
                'loss_est_real_NO_DISPLAY': crit_est['loss_real'].item(), 'est_real': crit_est['est_real'].item(),
                'loss_est_fake_NO_DISPLAY': crit_est['loss_fake'].item(), 'est_fake': crit_est['est_fake'].item()})
        ################################################################################################################
        # Discriminator
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_disc):
            audios, label = self._fetch_batch_data()
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=False)
            set_requires_grad([self._Disc], requires_grad=True)
            self._optimizers['disc'].zero_grad()
            # 1. Get disc_output
            disc_output_real = self._Disc(audios)
            style_emb = resampling(self._Enc_style(audios), self._cfg.args.style_std)
            disc_output_fake = self._Disc(self._Rec(style_emb.transpose(1, 2), label).detach())
            # 2. Calculate loss
            crit_disc_real = self._criterions['disc'](disc_output_real, True, lmd=1.0)
            crit_disc_fake = self._criterions['disc'](disc_output_fake, False, lmd=1.0)
            # Backward & save
            disc_acc = torch.cat([crit_disc_real['pred'] == 1, crit_disc_fake['pred'] == 0], dim=0).sum().item() / (audios.size(0) * 2)
            if disc_acc < self._cfg.args.disc_limit_acc:
                summarize_losses_and_backward(crit_disc_real['loss'], crit_disc_fake['loss'])
                self._optimizers['disc'].step()
            packs['log'].update({
                'loss_disc_real': crit_disc_real['loss'].item(), 'loss_disc_fake': crit_disc_fake['loss'].item(),
                'disc_acc': disc_acc})

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging
        self._process_log_after_step(packs)
        # 2. Evaluation
        if self._meters['counter_eval'].check(self._meters['i']['step']):
            vis_grid_disentangling(
                batch_data=map(lambda x: x[:self._cfg.args.eval_dis_n_samples], self._fetch_batch_data(no_record=True)),
                func_style=self._Enc_style, func_rec=self._Rec, gap_size=3,
                save_path=os.path.join(self._cfg.args.eval_dis_dir, 'step[%d].png' % self._meters['i']['step']))
        # 3. Chkpt
        self._process_chkpt_and_lr_after_step()
        # Clear packs
        packs['log'] = ValidContainer()

    def _process_log_after_step(self, packs, **kwargs):

        def _lmd_generate_log():
            r_tfboard = {
                'train/losses': fet_d(packs['log'], prefix='loss_', remove=('loss_', '_NO_DISPLAY')),
                'train/est': fet_d(packs['log'], prefix='est_')
            }
            # 确保 packs['log'] 是字典
            if isinstance(packs['log'], ValidContainer):
                packs['log'] = packs['log'].dict
            # 确保 r_tfboard 是字典
            if isinstance(r_tfboard, ValidContainer):
                r_tfboard = r_tfboard.dict

            # 更新 packs['tfboard']，使其为字典
            packs['tfboard'] = r_tfboard

        # 调用父类的 _process_log_after_step 方法，确保传递的 packs 是正确的格式
        super(DisenIB, self)._process_log_after_step(
            packs, lmd_generate_log=_lmd_generate_log, lmd_process_log=Logger.reform_no_display_items)

        # 格式化日志信息进行打印
        log_info = (
            f"Epoch [{packs['log'].get('epoch', 'N/A')}], "
            f"Iter [{packs['log'].get('iter', 'N/A')}], "
            f"Batch [{packs['log'].get('batch', 'N/A')}], "
            f"Loss Dec: {packs['log'].get('loss_dec', 'N/A'):.4f}, "
            f"Loss Rec: {packs['log'].get('loss_rec', 'N/A'):.4f}, "
            f"Estimator: {packs['log'].get('est', 'N/A'):.4f}, "
            f"Disc Acc: {packs['log'].get('disc_acc', 'N/A'):.2f}"
        )
        print(log_info)  # 你可以选择替换为 logger.info(log_info) 或 tqdm.write(log_info)

        # 将日志保存到文件中
        with open("training_log2.txt", "a") as log_file:
            log_file.write(log_info + "\n")
    
    def _inference_step(self, packs):
        # 获取 epoch 总数和当前 epoch
        current_epoch = self._meters['i']['epoch']
        steps = self._cfg.args.steps  # 假设你有 total_epochs 配置
        disc_acc = 0  # 初始化为 0 或其他默认值

        # tqdm 进度条设置
        tqdm_bar = tqdm(total=self._cfg.args.n_times_main, desc=f"Epoch {current_epoch + 1}/{steps}", unit="iter")

        ################################################################################################################
        # Main
        ################################################################################################################
        for i in range(self._cfg.args.n_times_main):
            tqdm_bar.update(1)  # 更新进度条
            audios, label = self._fetch_batch_data() 
            
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=True)
            set_requires_grad([self._Disc, self._Est], requires_grad=False)
            self._optimizers['main'].zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            # Decoding & reconstruction
            # ----------------------------------------------------------------------------------------------------------
            # style_emb = torch.randn(self._cfg.args.batch_size, self._cfg.args.style_dim, 100).to(self._cfg.args.device) # 64, 16. 100
            # class_emb = torch.randn(self._cfg.args.batch_size, self._cfg.args.class_dim).to(self._cfg.args.device) # 64, 16 
            # label = torch.from_numpy(np.array([0] * self._cfg.args.batch_size)).to(self._cfg.args.device, dtype=torch.int64) # 64
            # audios = torch.randn(self._cfg.args.batch_size, self._cfg.args.num_mels, 400).to(self._cfg.args.device) # 64, 80, 400

            style_emb, class_emb = self._Enc_style(audios), self._Enc_class(audios)
            # print("style_emb:", class_emb)
            
            # 1. Decoding: use class embedding(from encoder), to generate the label (speaker ID).
            # Optimized towards the ground truth label(speaker ID).
  
            dec_output = self._Dec(resampling(class_emb, self._cfg.args.class_std))
            # print(dec_output.size(), label.size())
            loss_dec = self._criterions['dec'](dec_output, label)

            # 2. Reconstruction: use style embedding(from encoder) and ground truth label(speaker ID), to reconstruct an audio.
            # Optimized towards the target audio.
            rec_output = self._Rec(resampling(style_emb.transpose(1, 2), self._cfg.args.style_std), label)
            rec_output = rec_output.transpose(1, 2) # (batch_size, time_steps, num_features)
            print("rec_output.size():", rec_output.size())
            print("audios.size():", audios.size())

            loss_rec = self._criterions['rec'](rec_output, audios)
            # print("Successful Reconstruction!")
            # Backward
            summarize_losses_and_backward(loss_dec, loss_rec, retain_graph=True)
            # print("Successful Backward!")
            # ----------------------------------------------------------------------------------------------------------
            # Estimator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate output (batch*n_samples, ) & loss (1, ).
            est_output = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            # print("Successful Estimator!")
            crit_est = self._criterions['est'](
                output=est_output, emb=(style_emb, class_emb), mode='main',
                lmd={'loss_est': self._cfg.args.lambda_est, 'loss_wall': self._cfg.args.lambda_wall})
            # print("Successful Crit Estimator!")
            # Backward
            # 1> Density estimation
            if self._meters['trigger_est'].check(self._meters['i']['step']):
                if self._meters['trigger_est_style_optimize'].check():
                    set_requires_grad(self._Enc_class, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_class, requires_grad=True)
                else:
                    set_requires_grad(self._Enc_style, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_style, requires_grad=True)
            # 2> Embedding wall
            summarize_losses_and_backward(crit_est['loss_wall'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Discriminator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate loss
            disc_output = self._Disc(rec_output)
            crit_gen = self._criterions['disc'](disc_output, True, lmd=self._cfg.args.lambda_disc)
            # Backward
            if self._meters['trigger_disc'].check(self._meters['i']['step']):
                summarize_losses_and_backward(crit_gen['loss'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Update
            self._optimizers['main'].step()
            """ Saving """
            # 将关键信息更新到日志 packs 中
            packs['log'].update({
                'epoch': current_epoch,
                'iter': self.global_iter,  # 使用全局迭代计数器
                'batch': self._meters['i']['step'],  # 假设 step 记录了当前 batch 信息
                'loss_dec': loss_dec.item(),
                'loss_rec': loss_rec.item(),
                'est': crit_est['est'].item(),
                'disc_acc': disc_acc
            })

            # 增加全局迭代计数器
            self.global_iter += 1
        tqdm_bar.close()  # 结束 tqdm 进度条