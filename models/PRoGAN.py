import torch
from . import networks
from phase_correlation.pytorch_lptm import FFT2, LogPolar, PhaseCorr, Corr2Softmax
from util.utils import *
from util.util import *
import kornia
import torch.nn as nn
import numpy as np
import matplotlib as plt
from torchviz import make_dot
from graphviz import Digraph
import torch
from util.image_pool import ImagePool
import random
import itertools
from torchviz import make_dot
from graphviz import Digraph
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class PRoGAN(ABC):
    """The implementation of PRoGAN network"""
    def __init__(self, args):
        """
        Initialize the parameters and build the networks of PRoGAN
        Parameters:
            args (Option class)
        """
        self.feature_size = args.load_size
        self.args = args
        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device(
                'cpu')
        self.save_dir = os.path.join(
            args.checkpoints_dir,
            args.name)
        torch.backends.cudnn.benchmark = True
        self.optimizers = []
        self.metric = 0
        self.iter_cal = 0
        # display loss names
        self.loss_names = [
            'loss_G_GAN', 'loss_G_L1', 'loss_D', 'rec_loss_l1', 'loss_ce', 'map_loss'
        ]
        # display image names
        self.visual_names = [
            'real_A', 'fake_B', 'fake_B_trans',  'real_B',
            'idt_B', 'rec_A', 'fake_B_2'
        ]
        if self.args.phase == 'train':
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # define the network of generator from A to B
        self.netG = networks.define_G(self.args.input_nc, self.args.output_nc,
                                      self.args.ngf, self.args.netG, not self.args.no_dropout, self.gpu_ids)
        # define the network of generator from B to A
        self.netF = networks.define_G(self.args.input_nc, self.args.output_nc,
                                      self.args.ngf, self.args.netG, not self.args.no_dropout, self.gpu_ids)
        self.model_corr2softmax = Corr2Softmax(200., 0.).to(self.device)
        self.model_trans_corr2softmax = Corr2Softmax(11.72, 0.).to(self.device)
        if self.args.phase == 'train':
            # define the network of discriminator using PatchGAN
            self.netD = networks.define_D(
                self.args.output_nc, self.args.ndf, self.args.n_layers_D, self.gpu_ids)
            self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.compute_mse = torch.nn.MSELoss()
            self.compute_kld = torch.nn.KLDivLoss()
            self.compute_loss_rot = torch.nn.CrossEntropyLoss(
                reduction="sum").to(self.device)
            self.fake_B_pool = ImagePool(self.args.pool_size)
            # define the optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                self.netG.parameters(), self.netF.parameters()),
                lr=self.args.lr,
                betas=(self.args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=self.args.lr,
                                                betas=(self.args.beta1, 0.999))
            self.optimizer_c2s = torch.optim.Adam(filter(
                lambda p: p.requires_grad,
                self.model_corr2softmax.parameters()),
                lr=1e-1)
            self.optimizer_trans_c2s = torch.optim.AdamW(filter(
                lambda p: p.requires_grad,
                self.model_trans_corr2softmax.parameters()),
                lr=5e-2)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_c2s)
            self.optimizers.append(self.optimizer_trans_c2s)
            self.schedulers = [
                networks.get_scheduler(optimizer, self.args)
                for optimizer in self.optimizers
            ]
        # if continue to train, then load the networks
        if not self.args.phase == 'train' or self.args.continue_train:
            load_suffix = self.args.epoch
            self.load_networks(load_suffix)

    def set_input(self, inputA, inputB, phase='train'):
        """Set the input for PRoGAN, input original image for realA, target image for realB"""
        self.real_A = inputA.to(self.device)
        self.real_B = inputB.to(self.device)
        self.phase = phase
        b, c, h, w = self.real_A.shape
        self.center = torch.ones(b, 2).to(self.device)
        self.center[:, 0] = h // 2
        self.center[:, 1] = w // 2

    def forward(self):
        """Run the forward pass for PRoGAN"""
        # translate A to B, and generate fake_B
        self.fake_B = self.netG(self.real_A)
        # translate fake_B to A, and generate rec_A, which establishes a cycle
        self.rec_A = self.netF(self.fake_B)

        # use DPC to estimate the relative pose for fake_B and real_B
        self.get_transformation(self.fake_B, self.real_B, 1, 1)
        # generate the fake_B_trans by transforming fake_B with relative pose
        self.fake_B_trans = self.get_inverse_trans(self.fake_B, 1, 1)

        # generate angle for randomization
        self.rot_gt = (np.random.rand()) * 180.0
        # rotate real_A by random angle rot_gt
        self.real_A_2 = self.get_inverse_trans(self.real_A, self.rot_gt, 3)
        # rotate real_B by random angle rot_gt
        self.real_B_2 = self.get_inverse_trans(self.real_B, self.rot_gt, 3)
        # input real_A_2 to G, and generate fake_B_2
        self.fake_B_2 = self.netG(self.real_A_2)
        # estimate the relative pose between fake_B_2 and real_B_2
        self.get_transformation(self.fake_B_2, self.real_B_2, 2, 0)
        # estimate the relative pose between fake_B and fake_B_2
        self.get_transformation(self.fake_B, self.fake_B_2, 0, 2)
        # generate idt_B by input real_B to G
        self.idt_B = self.netG(self.real_B)

    def train(self):
        """The training procedure"""
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        # predict self.fake_B to be false, real_B to be true
        fake_B_1 = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_B, fake_B_1)
        # predict self.fake_B_trans to be false, real_B to be true
        fake_B_2 = self.fake_B_pool.query(self.fake_B_trans)
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_B, fake_B_2)
        # predict self.fake_B_2 to be false, real_B to be true
        fake_B_3 = self.fake_B_pool.query(self.fake_B_2)
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_B, fake_B_3)

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()

    def test(self):
        """The test procedure"""
        self.fake_B = self.netG(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """The basic function for discriminator"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()
        return self.loss_D

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # calculate the loss between relative pose estimation between {fake_B, real_B} and {fake_B_2, real_B_2}
        self.map_loss = self.compute_kld(
            self.map_A_rs, self.map_B_rs) * self.args.lambda_L1 / 30.0
        pred_fake = self.netD(self.fake_B)
        pred_fake_trans = self.netD(self.fake_B_trans)
        pred_fake_2 = self.netD(self.fake_B_2)
        # loss_G_GAN = pred(fake_B, true) + pred(fake_B_trans, true) + pred(fake_B_2, true)  
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN_2 = self.criterionGAN(pred_fake_trans, True)
        self.loss_G_GAN_3 = self.criterionGAN(pred_fake_2, True)
        # calculate the L_trans loss between fake_B and real_B
        self.loss_G_L1 = self.criterionL1(self.fake_B_trans,
                                          self.real_B) * self.args.lambda_L1
        # calculate the recycle loss
        self.idt_loss = self.criterionL1(self.real_B,
                                         self.idt_B) * self.args.lambda_L1 * 4
        self.rec_loss_l1 = self.criterionL1(self.rec_A,
                                            self.real_A) * self.args.lambda_L1
        # calculate the loss_G = L_trans + L_cycle + loss_G_GAN + L_ss
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_2 + self.loss_G_GAN_3 + self.loss_G_L1 + \
            self.idt_loss + self.rec_loss_l1 + self.map_loss + self.loss_ce
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_c2s.step()

    def get_transformation(self, picA, picB, seq=0, get_angle=0):
        """DPC estimator for pose estimation, which is differentiable"""
        with torch.set_grad_enabled(self.phase == 'train'):
            if self.args.output_nc == 3:
                picA = picA[:, 1:2, :, :].clone()
                picB = picB[:, 1:2, :, :].clone()

            original_unet_rot = picA.permute(0, 2, 3, 1)
            target_unet_rot = picB.permute(0, 2, 3, 1)
            original_unet_rot = original_unet_rot.squeeze(-1)
            target_unet_rot = target_unet_rot.squeeze(-1)
            # DFT for original image and target image
            fft_layer = FFT2(self.device)
            original_fft = fft_layer(original_unet_rot)
            target_fft = fft_layer(target_unet_rot)
            # logpolar filter for high frequency noises
            h = logpolar_filter((target_fft.shape[1], target_fft.shape[2]),
                                self.device)
            original_fft = original_fft.squeeze(-1) * h
            target_fft = target_fft.squeeze(-1) * h
            original_fft = original_fft.unsqueeze(-1)
            target_fft = target_fft.unsqueeze(-1)
            # input DFT results to log polar 
            logpolar_layer = LogPolar(
                (original_fft.shape[1], original_fft.shape[2]), self.device)
            original_logpolar, logbase_rot = logpolar_layer(original_fft)
            target_logpolar, logbase_rot = logpolar_layer(target_fft)

            original_logpolar = original_logpolar.squeeze(-1)
            target_logpolar = target_logpolar.squeeze(-1)
            # phase correlation
            phase_corr_layer_rs = PhaseCorr(self.device, logbase_rot)
            scale, angle, r, logbase = phase_corr_layer_rs(
                original_logpolar.clone(), target_logpolar.clone())

            corr_final_rot = torch.sum(r.clone(), 2, keepdim=False)
            corr_final_scale = torch.sum(r.clone(), 1, keepdim=False)
            # softmax
            corr_final_rot = self.model_corr2softmax(corr_final_rot)
            input_rot = nn.functional.softmax(corr_final_rot.clone(), dim=-1)

            if seq == 1:
                self.map_A_rs = nn.functional.log_softmax(
                    corr_final_rot.clone(), dim=-1)
            elif seq == 2:
                self.map_B_rs = nn.functional.softmax(corr_final_rot.clone(),
                                                      dim=-1)
            # get expectation of relative pose
            indice_rot = np.linspace(0, 1, self.feature_size)
            indice_rot = torch.tensor(
                np.reshape(indice_rot,
                           (-1, self.feature_size))).to(self.device)
            angle = torch.sum((self.feature_size - 1) * input_rot * indice_rot,
                              dim=-1)

            corr_final_scale = self.model_corr2softmax(corr_final_scale)
            input_scale = nn.functional.softmax(corr_final_scale.clone(),
                                                dim=-1)
            scale = torch.sum(
                (self.feature_size - 1) * input_scale * indice_rot, dim=-1)
            b, c, h, w = picA.shape
            angle = angle * 180.00 / r.shape[1]
            for batch_num in range(angle.shape[0]):
                if angle[batch_num].item() >= 90:
                    angle[batch_num] -= 90.00
                else:
                    angle[batch_num] += 90.00
            logbase = logbase.to(self.device)

            sca_f = scale.clone() - self.feature_size // 2
            self.scale = 1 / torch.pow(logbase,
                                       sca_f.float())
            if not self.args.use_scale:
                self.scale = torch.ones(1)
            angle_rot = torch.ones(b).to(
                self.device) * (-angle.to(self.device))

            if get_angle == 1:
                self.angle = angle
            if get_angle == 2:
                gt_angle = GT_angle_convert(
                    torch.tensor([self.rot_gt]).to(self.device),
                    self.feature_size)
                self.loss_ce = self.compute_loss_rot(corr_final_rot,
                                                     gt_angle) / 30.0
                self.loss_ce.to(self.device)

            scale_rot = torch.ones(b).to(self.device) / self.scale.to(
                self.device)  # scale.to(self.device)
            rot_mat = kornia.get_rotation_matrix2d(self.center, angle_rot,
                                                   scale_rot)
            target_trans = kornia.warp_affine(picB.clone().to(self.device),
                                              rot_mat,
                                              dsize=(h, w),
                                              flags='nearest',
                                              align_corners=True)

            logbase_trans = torch.tensor(1.)

            original_unet_trans = original_unet_rot
            target_unet_trans = target_trans.permute(0, 2, 3, 1)

            target_unet_trans = target_unet_trans.squeeze(-1)
            phase_corr_layer_xy = PhaseCorr(self.device,
                                            logbase_trans,
                                            trans=True)
            tran_x, tran_y, r, logbase = phase_corr_layer_xy(
                original_unet_trans.clone().to(self.device),
                target_unet_trans.clone().to(self.device))
            if seq == 1:
                self.map_A_xy = r.clone()
            elif seq == 2:
                self.map_B_xy = r.clone()
            corr_final_y = torch.sum(r.clone(), 2, keepdim=False)
            corr_final_x = torch.sum(r.clone(), 1, keepdim=False)
            corr_final_y = corr_final_y.clone() * 1000
            corr_final_x = corr_final_x.clone() * 1000
            corr_final_y = nn.functional.softmax(corr_final_y.clone(), dim=-1)
            tran_y = torch.sum(
                (self.feature_size - 1) * corr_final_y * indice_rot, dim=-1)

            corr_final_x = nn.functional.softmax(corr_final_x.clone(), dim=-1)
            tran_x = torch.sum(
                (self.feature_size - 1) * corr_final_x * indice_rot, dim=-1)

            if get_angle:
                self.tran_x = tran_x - self.feature_size / 2
                self.tran_y = tran_y - self.feature_size / 2
                self.trans = torch.cat((self.tran_y, self.tran_x), 0)
        return angle

    def get_inverse_trans(self, pic, theta=0, choice=1):
        with torch.set_grad_enabled(self.phase == 'train'):
            b, c, h, w = pic.shape
            if choice == 1:
                trans_mat_affine = torch.Tensor([[[1.0, 0.0, self.tran_x],
                                                  [0.0, 1.0, self.tran_y]]
                                                 ]).to(self.device)

                target_trans = kornia.warp_affine(pic.clone(),
                                                  trans_mat_affine,
                                                  dsize=(h, w),
                                                  flags='nearest',
                                                  align_corners=True)
                angle_rot = torch.ones(b).to(self.device) * (self.angle.to(
                    self.device))
                scale_rot = torch.ones(b).to(self.device) * self.scale.to(
                    self.device)
                rot_mat = kornia.get_rotation_matrix2d(self.center, angle_rot,
                                                       scale_rot)
                target_trans = kornia.warp_affine(target_trans.to(self.device),
                                                  rot_mat,
                                                  dsize=(h, w),
                                                  flags='nearest',
                                                  align_corners=True)
            elif choice == 0:
                angle_rot = torch.ones(b).to(
                    self.device) * (-self.angle.to(self.device))
                scale_rot = torch.ones(b).to(self.device) / self.scale.to(
                    self.device)
                rot_mat = kornia.get_rotation_matrix2d(self.center, angle_rot,
                                                       scale_rot)
                target_trans = kornia.warp_affine(pic.clone().to(self.device),
                                                  rot_mat,
                                                  dsize=(h, w),
                                                  flags='nearest',
                                                  align_corners=True)
                trans_mat_affine = torch.Tensor([[[1.0, 0.0, -self.tran_x],
                                                  [0.0, 1.0, -self.tran_y]]
                                                 ]).to(self.device)

                target_trans = kornia.warp_affine(target_trans.clone(),
                                                  trans_mat_affine,
                                                  dsize=(h, w),
                                                  flags='nearest',
                                                  align_corners=True)
            elif choice == 3:
                angle_ = torch.tensor([theta], dtype=float)
                angle_.repeat(b, 1)
                scale = 1.0
                scale = torch.tensor([scale], dtype=float)
                scale.repeat(b, 1)
                angle_rot = torch.ones(b).to(self.device) * (angle_.to(
                    self.device))

                scale_rot = torch.ones(b).to(self.device) * scale.to(
                    self.device)
                rot_mat = kornia.get_rotation_matrix2d(self.center, angle_rot,
                                                       scale_rot)
                target_trans = kornia.warp_affine(pic.clone().to(self.device),
                                                  rot_mat,
                                                  dsize=(self.feature_size,
                                                         self.feature_size),
                                                  flags='nearest',
                                                  align_corners=True)

        return target_trans

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict,
                                                  getattr(module, key), keys,
                                                  i + 1)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path,
                                        map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys(
                )):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
