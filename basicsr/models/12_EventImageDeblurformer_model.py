import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import logging

from basicsr.models.archs import define_network
from basicsr.models.base1_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, get_model_flops

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
logger = logging.getLogger('basicsr')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class EventImageDeblurformerModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(EventImageDeblurformerModel, self).__init__(opt)
        # define network
       

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # flops
        if self.opt.get('print_flops', False):
            input_dim = self.opt.get('flops_input_shape', [(3, 256, 256),(6, 256, 256)])
            flops = get_model_flops(self.net_g, input_dim, False)
            flops = flops/10**9
            logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['frame'].to(self.device)
        self.voxel=data['voxel'].to(self.device) 
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'frame_gt' in data:
            self.gt = data['frame_gt'].to(self.device)
        if 'image_name' in data:
            self.image_name = data['image_name']

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['frame'].to(self.device)
        self.voxel=data['voxel'].to(self.device) 
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'frame_gt' in data:
            self.gt = data['frame_gt'].to(self.device)
        if 'image_name' in data:
            self.image_name = data['image_name']


    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t
 
    # 事件体素分块
    def grids_voxel(self):  
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)
        # print('step_i, stepj', step_i, step_j)
        # exit(0)
        parts = []
        idxes = []
        # cnt_idx = 0
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print('----------parts voxel .. ', len(parts), self.voxel.size())
        self.idxes = idxes

    # 图像分块
    def grids(self):  
        b, c, h, w = self.lq.size()  # lq is after data augment (for example, crop, if have)
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1
        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        # print('step_i, stepj', step_i, step_j)
        # exit(0)
        parts = []
        idxes = []
        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(x = self.lq, event = self.voxel)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # def pad_test(self, window_size):        
    #     scale = self.opt.get('scale', 1)
    #     mod_pad_h, mod_pad_w = 0, 0
    #     _, _, h, w = self.lq.size()
    #     if h % window_size != 0:
    #         mod_pad_h = window_size - h % window_size
    #     if w % window_size != 0:
    #         mod_pad_w = window_size - w % window_size
    #     img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     self.nonpad_test(img)
    #     _, _, h, w = self.output.size()
    #     self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    # def nonpad_test(self, img=None):
    #     if img is None:
    #         img = self.lq      
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             n = self.lq.size(0)  # n: batch size
    #             outs = []
    #             m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
    #             i = 0
    #             while i < n:
    #                 j = i + m
    #                 if j >= n:
    #                     j = n
    #                 # print("self.lq[i:j, :, :, :]:{}".format(self.lq[i:j, :, :, :].shape))
    #                 # print("self.voxel[i:j, :, :, :]:{}".format(self.voxel[i:j, :, :, :].shape))
    #                 pred = self.net_g_ema(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :])  # mini batch all in 
    #             if isinstance(pred, list):
    #                 pred = pred[-1]
    #             outs.append(pred)
    #             i = j
    #         self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             n = self.lq.size(0)  # n: batch size
    #             outs = []
    #             m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
    #             i = 0
    #             while i < n:
    #                 j = i + m
    #                 if j >= n:
    #                     j = n
    #                 # print("self.lq[i:j, :, :, :]:{}".format(self.lq[i:j, :, :, :].shape))
    #                 # print("self.voxel[i:j, :, :, :]:{}".format(self.voxel[i:j, :, :, :].shape))
    #                 pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :])  # mini batch all in 
    #             if isinstance(pred, list):
    #                 pred = pred[-1]
    #             outs.append(pred)
    #             i = j
    #         self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
    #         self.net_g.train()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                if 'val' in self.opt['datasets'] and self.opt['datasets']['val'].get('use_mask'):
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :], mask = self.mask[i:j, :, :, :])  # mini batch all in 
                elif 'test' in self.opt['datasets'] and self.opt['datasets']['test'].get('use_mask'):
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :], mask = self.mask[i:j, :, :, :])  # mini batch all in 
                else:
                    # print("self.lq[i:j, :, :, :]:{}".format(self.lq[i:j, :, :, :].shape))
                    # print("self.voxel[i:j, :, :, :]:{}".format(self.voxel[i:j, :, :, :].shape))
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :])  # mini batch all in 
            
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')
        window_size = self.opt['val'].get('window_size', 0)

        # if window_size:
        #     test = partial(self.pad_test, window_size)
        # else:
        #     test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()


            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             self.image_name,
                                             f'{self.image_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             self.image_name,
                                             f'{self.image_name}_{current_iter}_gt.png')
                else:
                    # print('Save path:{}'.format(self.opt['path']['visualization']))
                    # print('Dataset name:{}'.format(dataset_name))
                    # print('Img_name:{}'.format(self.image_name))
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{self.image_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{self.image_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            pbar.update(1)
            pbar.set_description(f'Test {self.image_name}')
            cnt += 1

        pbar.close()
        
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def nondist_test(self, dataloader, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()


            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                # print("image_name:{}".format(self.image_name))
                img_name = self.image_name[0]
                # print("img_name:{}".format(img_name))
                save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                imwrite(sr_img, save_img_path)
                if 'gt' in visuals:
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    imwrite(gt_img, save_gt_img_path)
                    
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            pbar.update(1)
            pbar.set_description(f'Test {self.image_name}')
            cnt += 1

        pbar.close()
        
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
