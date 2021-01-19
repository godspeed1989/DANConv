import os
import logging
import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from kitti_dataset import readlines
from kitti_dataset import KITTIRAWDataset, DAT_VAL_TEST, _merge_batch
from options import CompletionOptions
from sparse_model import Model
from metric import Metrics, AverageMeter

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'

    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_lr_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def save_model(model, path, epoch, mse, rmse):
    save_folder = os.path.join(path, "models")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, "weights_{}.pth".format(epoch))
    to_save = {
        'mae': mae,
        'rmse': rmse,
        'state_dict': model.state_dict()
    }
    torch.save(to_save, save_path)

def log_time(epoch, batch_idx,
             start_time, duration,
             total_batches, batch_size, print_dict=None):
    def sec_to_hm(t):
        """Convert time in seconds to time in hours, minutes and seconds
        e.g. 10239 -> (2, 50, 39)
        """
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return t, m, s
    def sec_to_hm_str(t):
        """Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        h, m, s = sec_to_hm(t)
        return "{:02d}h{:02d}m{:02d}s".format(h, m, s)
    samples_per_sec = batch_size / duration
    time_sofar = time.time() - start_time
    step = batch_idx + 1
    training_time_left = (total_batches / step - 1.0) * time_sofar
    print_string = "epoch {:>3} | batch {}/{} | examples/s: {:5.1f} | " + \
                   "time elapsed: {} | time left: {}"
    print(print_string.format(epoch, batch_idx, total_batches, samples_per_sec,
                              sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)), end=' ')
    if print_dict:
        for k,v in print_dict.items(): print("| {}: {}".format(k,v), end=' ')
    print('')


def MSE_loss(prediction, gt, mask):
    err = prediction[:,0:1] - gt
    mse_loss = torch.sum((err[mask])**2) / mask.sum()
    return mse_loss

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

def edge_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def HeteroscedasticLoss(outputs, targets, mask, eps=1e-5):
    mean, var = outputs[0][mask], outputs[1][mask]
    var = torch.clamp_min(var, 0)
    precision = eps / (var + eps)
    e1 = 1000 * precision * (targets[mask] - mean) ** 2
    e2 = torch.log(var + eps)
    ee = 0.5 * (e1 + e2)
    return torch.mean(ee) * eps

def HeteroscedasticLoss1(outputs, targets, mask, eps=1e-5):
    mean, var = outputs[0][mask], outputs[1][mask]
    var = torch.clamp_min(var, eps)
    precision = 1 / (var)
    e1 = torch.sqrt(torch.sum(precision * (targets[mask] - mean) ** 2))
    e2 = torch.sum(torch.log(var))
    ee = e1 + e2
    return ee / mask.sum()

def train_epoch(epoch, max_epoch, G_model: Model,
                loader, optimizer, device, tb_log, cfg, multi_gpu):
    start_time = time.time()
    batch_size = loader.batch_size

    G_model.train()
    if multi_gpu:
        loader.sampler.set_epoch(epoch)
    for batch_idx, inputs in enumerate(loader):
        it = epoch * len(loader) + batch_idx
        before_op_time = time.time()

        optimizer.zero_grad()

        ext = {}
        rgb = inputs['color_aug'].to(device) * 255.
        depth = inputs['depth_aug'].to(device)
        gtdepth = inputs['depth_sd_gt'].to(device)
        mask = inputs['mask'].to(device)
        ext['aff_locsx1'] = [e.to(device) for e in inputs['aff_locsx1']]
        ext['aff_nnidxsx1'] = [e.to(device) for e in inputs['aff_nnidxsx1']]
        ext['aff_locsx2'] = [e.to(device) for e in inputs['aff_locsx2']]
        ext['aff_nnidxsx2'] = [e.to(device) for e in inputs['aff_nnidxsx2']]
        ext['aff_locsx4'] = [e.to(device) for e in inputs['aff_locsx4']]
        ext['aff_nnidxsx4'] = [e.to(device) for e in inputs['aff_nnidxsx4']]
        ext['aff_locsx8'] = [e.to(device) for e in inputs['aff_locsx8']]
        ext['aff_nnidxsx8'] = [e.to(device) for e in inputs['aff_nnidxsx8']]
        ext['maskx2'] = inputs['maskx2'].to(device)
        ext['maskx4'] = inputs['maskx4'].to(device)
        ext['maskx8'] = inputs['maskx8'].to(device)
        gt_mask = (gtdepth > 0).detach()

        pred = G_model(depth, mask, rgb, ext)
        pred_depth = pred[0]
        pred_d = pred[1]
        pred_r = pred[2]
        hetero = 0

        loss_depth_a = MSE_loss(pred_depth, gtdepth, gt_mask)
        loss_depth_d = MSE_loss(pred_d, gtdepth, gt_mask) * 0.5
        loss_depth_r = MSE_loss(pred_r, gtdepth, gt_mask) * 0.5
        loss_depth = loss_depth_a + loss_depth_d + loss_depth_r
        loss_smooth = smooth_loss(pred_depth) * cfg.weight_smooth_loss

        loss = loss_depth + loss_smooth + hetero * 0.1

        if tb_log is not None:
            tb_log.add_scalar('loss', loss.item(), it)
            tb_log.add_scalar('loss_depth', loss_depth.item(), it)
            tb_log.add_scalar('loss_depth_a', loss_depth_a.item(), it)
            tb_log.add_scalar('loss_depth_d', loss_depth_d.item(), it)
            tb_log.add_scalar('loss_depth_r', loss_depth_r.item(), it)
            tb_log.add_scalar('loss_smooth', loss_smooth.item(), it)
        loss.backward()

        # clip the grad
        clip_grad_norm_(G_model.parameters(), max_norm=20, norm_type=2)

        optimizer.step()

        duration = time.time() - before_op_time

        if batch_idx % 10 == 0:
            print_dict = {}
            print_dict['loss'] = '{:.5f}'.format(loss.item())
            log_time(epoch, batch_idx, start_time, duration,
                    len(loader), batch_size, print_dict)
        if batch_idx % cfg.log_frequency == 0:
            def to_img(tensor, itype):
                # convert float tensor to tensorboardX supported image
                img = tensor.detach().cpu().numpy()
                if itype == 'depth':
                    img = img[0]
                    img = np.clip(img, cfg.min_depth, cfg.max_depth)
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                    img = 255 * plt.cm.jet(img)[:, :, :3]  # H, W, C
                    img = np.transpose(img, [2, 0, 1])
                    return img.astype('uint8')
                elif itype == 'var':
                    img = img[0]
                    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
                    img = 255 * plt.cm.jet(img)[:, :, :3]  # H, W, C
                    img = np.transpose(img, [2, 0, 1])
                    return img.astype('uint8')
                elif itype == 'gray':
                    return np.clip(img, 0, 1)
                elif itype == 'color':
                    return np.clip(img, 0, 255) / 255.0
            if tb_log is not None:
                tb_log.add_image('rgb', to_img(rgb[0], 'color'), it)
                tb_log.add_image('sparse_depth', to_img(depth[0], 'depth'), it)
                tb_log.add_image('pred_depth', to_img(pred_depth[0], 'depth'), it)
                tb_log.add_image('gt_depth', to_img(gtdepth[0], 'depth'), it)


def validate(model, val_loader, device, min_depth, max_depth, cfg):
    model.eval()

    metric = Metrics(max_depth=max_depth)
    mae = AverageMeter()
    rmse = AverageMeter()
    imae = AverageMeter()
    irmse = AverageMeter()
    with torch.no_grad():
        for _, inputs in tqdm(enumerate(val_loader)):
            ext = {}
            rgb = inputs['color'].to(device) * 255.
            sdepth = inputs['depth_gt'].to(device)
            mask = inputs['mask'].to(device)
            ext['aff_locsx1'] = [e.to(device) for e in inputs['aff_locsx1']]
            ext['aff_nnidxsx1'] = [e.to(device) for e in inputs['aff_nnidxsx1']]
            ext['aff_locsx2'] = [e.to(device) for e in inputs['aff_locsx2']]
            ext['aff_nnidxsx2'] = [e.to(device) for e in inputs['aff_nnidxsx2']]
            ext['aff_locsx4'] = [e.to(device) for e in inputs['aff_locsx4']]
            ext['aff_nnidxsx4'] = [e.to(device) for e in inputs['aff_nnidxsx4']]
            ext['aff_locsx8'] = [e.to(device) for e in inputs['aff_locsx8']]
            ext['aff_nnidxsx8'] = [e.to(device) for e in inputs['aff_nnidxsx8']]
            ext['maskx2'] = inputs['maskx2'].to(device)
            ext['maskx4'] = inputs['maskx4'].to(device)
            ext['maskx8'] = inputs['maskx8'].to(device)

            pred = model(sdepth, mask, rgb, ext)
            output = pred[0]

            output = torch.clamp(output, min_depth, max_depth)
            output[output<=0.9] = 0.9
            output = output[:,0:1].detach().cpu()

            gt = inputs['depth_sd_gt']
            metric.calculate(output, gt)
            mae.update(metric.get_metric('mae'), metric.num)
            rmse.update(metric.get_metric('rmse'), metric.num)
            imae.update(metric.get_metric('imae'), metric.num)
            irmse.update(metric.get_metric('irmse'), metric.num)

    model.train()
    return mae.avg, rmse.avg, imae.avg, irmse.avg


if __name__ == "__main__":
    timestr = time.strftime("%b-%d_%H-%M", time.localtime())
    root_result_dir = os.path.join('./output-' + timestr)
    os.makedirs(root_result_dir, exist_ok=True)
    # log to file
    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('***************Start logging***************')

    options = CompletionOptions()
    args = options.parse()

    G_mod = Model(scales=4, base_width=32)

    gpu = False
    multi_gpu = False
    curr_rank = args.local_rank
    device = torch.device("cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cudnn.benchmark = True
        gpu = True
        device = torch.device("cuda")

    # more than one gpu
    if args.mgpus and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.cuda.set_device(curr_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        logger.info("Current rank {}".format(curr_rank))
        multi_gpu = True
        device = torch.device('cuda:%d' % curr_rank)

    G_mod = G_mod.to(device)
    if multi_gpu:
        # spread in gpus
        G_mod = nn.parallel.DistributedDataParallel(G_mod, device_ids=[curr_rank], output_device=curr_rank)

    # copy important files to backup
    if curr_rank == 0:
        os.system('cp *.py %s/' % root_result_dir)
        for key, val in vars(args).items():
            logger.info("{:16} {}".format(key, val))

    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        load_mod = G_mod.module if multi_gpu else G_mod
        load_mod.load_state_dict(checkpoint['state_dict'])
        logger.info("use saved model: {}".format(args.weight_path))
        logger.info("saved model: mae {} rmse {}".format(checkpoint['mae'], checkpoint['rmse']))

    # optimizer & lr scheduler
    optimizer = define_optim(args.optimizer,
                            [{'params':G_mod.parameters()},],
                            args.learning_rate, args.weight_decay)
    scheduler = define_lr_scheduler(optimizer, args)

    img_ext = '.png' if args.png else '.jpg'
    if args.split == 'tiny':
        fpath = os.path.join(os.path.dirname(__file__), "splits/tiny", "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        train_dataset = KITTIRAWDataset(
            args.data_path, train_filenames, args.crop_h, args.crop_w,
            img_ext=img_ext, min_depth=args.min_depth, max_depth=args.max_depth)
        val_dataset = KITTIRAWDataset(
            args.data_path, val_filenames, args.crop_h, args.crop_w,
            img_ext=img_ext, min_depth=args.min_depth, max_depth=args.max_depth)
    elif args.split == 'full':
        fpath = os.path.join(os.path.dirname(__file__), "splits/", "{}.txt")
        train_filenames = readlines(fpath.format("train"))
        train_dataset = KITTIRAWDataset(
            args.data_path, train_filenames, args.crop_h, args.crop_w,
            img_ext=img_ext, min_depth=args.min_depth, max_depth=args.max_depth)
        val_dataset = DAT_VAL_TEST(
            args.data_path, is_test=False, crop_h=args.crop_h, crop_w=args.crop_w,
            min_depth=args.min_depth, max_depth=args.max_depth)
    logger.info("Using split:  {}".format(args.split))
    logger.info("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

    # dataloader
    shuffle = True
    train_sampler, val_sampler = None, None
    if multi_gpu:
        shuffle = False
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    train_loader = DataLoader(
            train_dataset, args.batch_size, shuffle=shuffle, collate_fn=_merge_batch,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(
            val_dataset, args.batch_size, shuffle=False, collate_fn=_merge_batch,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=val_sampler)

    if multi_gpu:
        if curr_rank == 0:
            tb_log = SummaryWriter(logdir=os.path.join(root_result_dir, 'tensorboard'))
        else:
            tb_log = None
    else:
        tb_log = SummaryWriter(logdir=os.path.join(root_result_dir, 'tensorboard'))

    step = 0
    start_time = time.time()
    for e in range(args.num_epochs):
        logging.info("Epoch {}".format(e+1))
        train_epoch(e, args.num_epochs, G_mod,
                    train_loader, optimizer, device, tb_log, args, multi_gpu)
        mae, rmse, imae, irmse = validate(G_mod, val_loader, device, args.min_depth, args.max_depth, args)
        logger.info("R{} Epoch {} MAE:{:.4f} RMSE:{:.4f}".format(curr_rank, e+1, mae, rmse))
        logger.info("R{} Epoch {} iMAE:{:.4f} iRMSE:{:.4f}".format(curr_rank, e+1, imae, irmse))
        if tb_log is not None:
            tb_log.add_scalar('MAE', mae, e+1)
            tb_log.add_scalar('RMSE', rmse, e+1)
            tb_log.add_scalar('iMAE', imae, e+1)
            tb_log.add_scalar('iRMSE', irmse, e+1)

        if (e + 1) % args.save_frequency == 0 and curr_rank == 0:
            logger.info("R{} save weights after {} epochs".format(curr_rank, e+1))
            save_mod = G_mod.module if multi_gpu else G_mod
            save_model(save_mod, root_result_dir, e + 1, mae, rmse)

        if args.lr_policy != 'plateau':
            scheduler.step()
        elif args.lr_policy == 'plateau':
            scheduler.step(rmse)
        logger.info("lr is set to {}".format(optimizer.param_groups[0]['lr']))
