import torch
import time, random, cv2, sys
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
import torch.distributed as dist
import os

class Test(BaseTrainer):
    def __init__(self, model, resume, config, iter_per_epoch, val_loader=None, train_logger=None, gpu=None, test=False):
        super(Test, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.val_loader = val_loader

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test
        self.save_dir = config['trainer']['save_dir'] + config['experim_name']

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.n_labeled_examples = config['n_labeled_examples']

        self.dataset = config['dataset']

    def _train_epoch(self, epoch):
        print(epoch)

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        save_folder = os.path.join(self.save_dir, 'preds')
        os.makedirs(save_folder, exist_ok=True)

        if self.dataset == 'cityscapes':
            from utils import pallete
            palette = pallete.citypallete
        else:
            from utils import pallete
            palette = pallete.get_voc_pallete(self.num_classes)


        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                output = self.model(data)
                output = output[:, :, :H, :W]
                pred = np.asarray(output.max(1)[1].squeeze(0).detach().cpu(), np.uint8)
                pred_col = colorize_mask(pred, palette)
                pred_col.save(os.path.join(save_folder, image_id[0] + '.png'))

                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}
                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, mIoU: {:.2f} |'.format(epoch, total_loss_val.average, 100*mIoU))

            if self.gpu == 0:
                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]: self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }

        return log

    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.copy()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.ClsReg_F1_l, self.ClsReg_F1_ul = 0, 0

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

class Save_Features(BaseTrainer):
    def __init__(self, model, resume, config, iter_per_epoch, val_loader=None, train_logger=None, gpu=None, test=False):
        super(Save_Features, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.val_loader = val_loader

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test
        self.save_dir = config['trainer']['save_dir'] + config['experim_name']

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.n_labeled_examples = config['n_labeled_examples']

        self.dataset = config['dataset']

    def _train_epoch(self, epoch):
        print(epoch)

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        save_folder = os.path.join(self.save_dir, 'features')
        os.makedirs(save_folder, exist_ok=True)

        if self.dataset == 'cityscapes':
            from utils import pallete
            palette = pallete.citypallete
        else:
            from utils import pallete
            palette = pallete.get_voc_pallete(self.num_classes)


        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                # output = self.model(data)
                feat, output = self.model(data)

                feat = F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=True)
                feat = feat.cpu().numpy()
                feat = feat.astype(np.float16)
                for j, id in enumerate(image_id):
                    np.save(os.path.join(save_folder, id + '.npy'), feat[j,:])

                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}
                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, mIoU: {:.2f} |'.format(epoch, total_loss_val.average, 100*mIoU))

            if self.gpu == 0:
                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]: self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }

        return log

    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.copy()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.ClsReg_F1_l, self.ClsReg_F1_ul = 0, 0

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

class Trainer_Baseline(BaseTrainer):
    def     __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, gpu=None, test=False):
        super(Trainer_Baseline, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.epoch_start_unsup = config['model']['epoch_start_unsup']

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            tbar = tqdm(range(self.iter_per_epoch), ncols=160)

        self._reset_metrics()

        for batch_idx in tbar:

            if self.mode == 'supervised':
                # (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)
                (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader), (None, None, None)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
            else:
                # (input_l, target_l), (input_ul, target_ul, flip) = next(dataloader)
                (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
                if target_ul.dim()==4: target_ul = target_ul.squeeze(1)

            if self.mode == 'supervised':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            else:
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                kargs = {'gpu': self.gpu, 'flip': flip}
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1,
                                                             **kargs)

            total_loss.backward()
            self.optimizer.step()

            if self.gpu == 0:
                if batch_idx % 100 == 0:
                    if self.mode == 'supervised':
                        self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}".
                                         format(epoch, total_loss, cur_losses['Ls']))
                    else:
                        if epoch-1 < self.epoch_start_unsup:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls']))
                        else:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Lu={:.3f}, mIoU_l={:.2f}, ul={:.2f}".
                                             format(epoch, total_loss, cur_losses['Ls'], cur_losses['Lu'],
                                                    100*self.mIoU_l, 100*self.mIoU_ul))

            if batch_idx == 0:
                for key in cur_losses:
                    if not hasattr(self, key):
                        setattr(self, key, AverageMeter())

            # self._update_losses has already implemented synchronized DDP
            self._update_losses(cur_losses)

            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            if self.gpu == 0:
                logs = self._log_values(cur_losses)

                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)

                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                descrip += 'mIoU_l {:.2f} ul {:.2f} |'.format(self.mIoU_l, self.mIoU_ul)
                tbar.set_description(descrip)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler.step(epoch=epoch - 1)

        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tbar):
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                output = self.model(data)

                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)

                total_loss_val.update(loss.item())

                # eval_metrics has already implemented DDP synchronized
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}

                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, Mean IoU: {:.2f} |'.format(epoch, total_loss_val.average,100*mIoU))

            if self.gpu == 0:
                # self._add_img_tb(val_visual, 'val')

                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.ClsReg_F1_l, self.ClsReg_F1_ul = 0, 0

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "Ls" in cur_losses.keys():
            logs['Ls'] = self.Ls.average
        if "Lu" in cur_losses.keys():
            logs['Lu'] = self.Lu.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_l'] = self.mIoU_l
        if self.mode == 'semi':
            logs['mIoU_ul'] = self.mIoU_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

class Trainer_USRN(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, gpu=None, test=False):
        super(Trainer_USRN, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.epoch_start_unsup = config['model']['epoch_start_unsup']

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            tbar = tqdm(range(self.iter_per_epoch), ncols=160)

        self._reset_metrics()

        for batch_idx in tbar:

            if self.mode == 'supervised':
                # (input_l, target_l, image_id), (input_ul, target_ul) = next(dataloader), (None, None)
                (input_l, target_l, target_l_subcls, image_id), (input_ul, target_ul, flip) = next(dataloader), (None, None, None)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
            else:
                # (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader)
                (input_l, target_l, target_l_subcls, image_id), (input_ul, target_ul, flip) = next(dataloader)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
                if target_ul.dim()==4: target_ul = target_ul.squeeze(1)

            if self.mode == 'supervised':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, target_l_subcls=target_l_subcls, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            else:
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                target_l_subcls = target_l_subcls.cuda(non_blocking=True)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                kargs = {'gpu': self.gpu, 'flip': flip}
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, target_l_subcls=target_l_subcls, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1,
                                                             **kargs)
            total_loss.backward()
            self.optimizer.step()

            if batch_idx == 0:
                for key in cur_losses:
                    if not hasattr(self, key):
                        setattr(self, key, AverageMeter())

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            if self.gpu == 0:
                if batch_idx % 20 == 0:
                    if self.mode == 'supervised':
                        self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Ls_sub={:.3f}".
                                         format(epoch, total_loss, cur_losses['Ls'], cur_losses['Ls_sub']))
                    else:
                        if epoch -1 < self.epoch_start_unsup:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Ls_sub={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls'], cur_losses['Ls_sub']))
                        else:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Ls_sub={:.3f}, Lu_reg={:.3f}, Lu_sub={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls'], cur_losses['Ls_sub'],
                                                    cur_losses['Lu_reg'], cur_losses['Lu_sub'], ))

            if self.gpu == 0:
                logs = self._log_values(cur_losses)

                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)

                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                descrip += 'mIoU_l {:.2f} ul {:.2f} |'.format(100*self.mIoU_l, 100*self.mIoU_ul)
                tbar.set_description(descrip)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler.step(epoch=epoch - 1)

        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tbar):
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)
                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(data)
                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                # eval_metrics has already implemented DDP synchronized
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}
                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, Mean IoU: {:.2f} |'.format(epoch, total_loss_val.average,100*mIoU))
            if self.gpu == 0:
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': np.round(total_loss_val.average,3),
                **seg_metrics
            }
        return log

    def _reset_metrics(self):
        self.Ls = AverageMeter()
        self.Ls_sub = AverageMeter()
        self.Lu_reg = AverageMeter()
        self.Lu_sub = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.mIoU_ul_reg = 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {"Pixel_Accuracy": pixAcc, "Mean_IoU": mIoU, "Class_IoU": dict(zip(range(self.num_classes), IoU))}

    def _log_values(self, cur_losses):
        logs = {}
        if "Ls" in cur_losses.keys():
            logs['Ls'] = self.Ls.average
        if "Ls_sub" in cur_losses.keys():
            logs['Ls_sub'] = self.Ls_sub.average
        if "Lu" in cur_losses.keys():
            logs['Lu'] = self.Lu.average
        if "Lu_reg" in cur_losses.keys():
            logs['Lu_reg'] = self.Lu_reg.average
        if "Lu_sub" in cur_losses.keys():
            logs['Lu_sub'] = self.Lu_sub.average
        logs['mIoU_l'] = self.mIoU_l
        if self.mode == 'semi':
            logs['mIoU_ul'] = self.mIoU_ul
            logs['mIoU_ul_reg'] = self.mIoU_ul_reg
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
