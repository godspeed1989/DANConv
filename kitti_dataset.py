from __future__ import absolute_import, division, print_function

import os, sys
import numbers
import random
import numpy as np

import copy
from PIL import Image
import PIL.Image as pil
import fire
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

def gen_affinity(depth, mask, knn=9, r=2):
    n, c, h, w = depth.shape
    locs = []
    nnidxs = []
    for i in range(n):
        # (M,2) -> (1,M,2)
        loc = torch.nonzero(mask[i, 0]>0)
        coords = loc.float()
        M = coords.size(0)
        dev = coords.device
        coords = coords.unsqueeze(0).detach().cpu().numpy()
        # (M,1), (M, r*knn)
        depths = depth[i, 0, loc[:,0], loc[:,1]].unsqueeze(1)
        nnidx = nearest_neighbors.knn_batch(coords, coords, knn*r, omp=True)[0]
        # distance (M, r*knn)
        nnidx = torch.from_numpy(nnidx).to(dev)
        nnidx_flat = nnidx.flatten().long()
        sel_depths = depths[nnidx_flat,0].reshape(M, r*knn)
        sel_depths_dist = torch.abs(depths - sel_depths)
        # (M, r*knn) -> (M, knn)
        sel_idx = torch.argsort(sel_depths_dist, dim=1, descending=False)
        sel_idx = sel_idx[:,:knn]
        nnidx_new = torch.gather(nnidx, 1, sel_idx)
        locs.append(loc)
        nnidxs.append(nnidx_new.long())

    return locs, nnidxs

def depth_down(input_d, mask, k):
    return F.avg_pool2d(input_d, k, k) / (F.avg_pool2d(mask, k, k) + 0.0001)

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def norm_depth(depth, min_depth, max_depth):
    norm_depth = np.clip(depth, min_depth, max_depth)
    return norm_depth

class BottomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for bottom crop.
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for bottom crop.
        """
        h = img.shape[1]
        w = img.shape[2]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (C x H x W): Image to be cropped
        Returns:
            img (C x H x W): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        def _is_tensor_image(img):
            return torch.is_tensor(img) and img.ndimension() == 3
        if not _is_tensor_image(img):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        return img[:, i:i + h, j:j + w]

def aug_pil_img(im):
    choice = np.random.randint(0, 4)
    if choice == 0:
        new_im = transforms.ColorJitter(brightness=0.5)(im)
    elif choice == 1:
        new_im = transforms.ColorJitter(contrast=0.5)(im)
    elif choice == 2:
        new_im = transforms.ColorJitter(saturation=0.5)(im)
    elif choice == 3:
        new_im = transforms.ColorJitter(hue=0.3)(im)
    return new_im

def generate_xyzm_map(mask, xyz_map):
    mask = np.expand_dims(mask, 0)
    xyz_map = np.transpose(xyz_map, [2, 0, 1])
    xyzm_map = np.concatenate([xyz_map, mask], 0)
    return xyzm_map

class MonoDataset(Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 crop_h,
                 crop_w,
                 min_depth=0.1,
                 max_depth=100,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.interp = Image.ANTIALIAS

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor() # (H x W x C) [0, 255] -> (C x H x W) [0.0, 1.0]

        self.load_depth = self.check_depth()
        self.load_depth_sd = self.check_depth_sd()

    def crop_img(self, img):
        assert img.size(1) >= self.crop_h, "[{}<{}]".format(img.size(1), self.crop_h)
        assert img.size(2) >= self.crop_w, "[{}<{}]".format(img.size(2), self.crop_w)
        transform = transforms.Compose([
            BottomCrop((self.crop_h, self.crop_w)),
        ])
        return transform(img)

    def __len__(self):
        # left and right
        return len(self.filenames) * 2

    def __getitem__(self, _index):
        """Returns a single training item from the dataset as a dictionary.
        """
        inputs = {}

        do_color_aug = random.random() > 0.5
        do_flip = random.random() > 0.5
        do_disp_aug = random.random() > 0.5

        index, side = (_index,"l") if _index < len(self.filenames) else (_index-len(self.filenames),"r")

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])

        inputs['folder'] = folder
        inputs['frame_index'] = frame_index
        inputs['side'] = side

        inputs["color"] = self.get_color(folder, frame_index, side, do_flip)

        if do_color_aug:
            color_aug = aug_pil_img
        else:
            color_aug = (lambda x: x)

        f = inputs["color"]
        inputs["color"] = self.crop_img(self.to_tensor(f))
        inputs["color_aug"] = self.crop_img(self.to_tensor(color_aug(f)))

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            depth_gt = np.expand_dims(depth_gt, 0)
            depth_gt_norm = norm_depth(depth_gt, self.min_depth, self.max_depth)
            inputs["depth_gt"] = torch.from_numpy(depth_gt_norm.astype(np.float32))
            inputs["depth_gt"] = self.crop_img(inputs["depth_gt"])
            if do_disp_aug:
                depth_aug = np.random.normal(0, 0.01, depth_gt.shape) * (depth_gt > 0.1)  + depth_gt
            else:
                depth_aug = depth_gt
            depth_aug_norm = norm_depth(depth_aug, self.min_depth, self.max_depth)
            inputs["depth_aug"] = torch.from_numpy(depth_aug_norm.astype(np.float32))
            inputs["depth_aug"] = self.crop_img(inputs["depth_aug"])
            inputs['mask'] = (inputs["depth_aug"] > 0.1).float()

        if self.load_depth_sd:
            depth_gt_sd = self.get_depth_sd(folder, frame_index, side, do_flip)
            depth_sd_gt = np.expand_dims(depth_gt_sd, 0)
            depth_sd_gt_norm = norm_depth(depth_sd_gt, self.min_depth, self.max_depth)
            inputs["depth_sd_gt"] = torch.from_numpy(depth_sd_gt_norm.astype(np.float32))
            inputs["depth_sd_gt"] = self.crop_img(inputs["depth_sd_gt"])

        sdepth = inputs['depth_gt'].unsqueeze(0) # (1, 1, H, W)
        mask = inputs['mask'].unsqueeze(0)
        sdepthx2, maskx2 = depth_down(sdepth, mask, 2), F.max_pool2d(mask, 2, 2)
        sdepthx4, maskx4 = depth_down(sdepth, mask, 4), F.max_pool2d(maskx2, 2, 2)
        sdepthx8, maskx8 = depth_down(sdepth, mask, 8), F.max_pool2d(maskx4, 2, 2)
        #
        aff_locsx1, aff_nnidxsx1 = gen_affinity(sdepth, mask)
        inputs['aff_locsx1'], inputs['aff_nnidxsx1'] = aff_locsx1[0], aff_nnidxsx1[0]
        aff_locsx2, aff_nnidxsx2 = gen_affinity(sdepthx2, maskx2)
        inputs['aff_locsx2'], inputs['aff_nnidxsx2'] = aff_locsx2[0], aff_nnidxsx2[0]
        aff_locsx4, aff_nnidxsx4 = gen_affinity(sdepthx4, maskx4)
        inputs['aff_locsx4'], inputs['aff_nnidxsx4'] = aff_locsx4[0], aff_nnidxsx4[0]
        aff_locsx8, aff_nnidxsx8 = gen_affinity(sdepthx8, maskx8)
        inputs['aff_locsx8'], inputs['aff_nnidxsx8'] = aff_locsx8[0], aff_nnidxsx8[0]
        inputs['maskx2'] = maskx2[0]
        inputs['maskx4'] = maskx4[0]
        inputs['maskx8'] = maskx8[0]

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def check_depth_sd(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_depth_sd(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0].split("/")[1]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            "data_depth_annotated/",
            scene_name,
            "proj_depth/velodyne_raw/image_0{}/{:010d}.png".
            format("2", int(frame_index)))

        return os.path.isfile(velo_filename)

    def check_depth_sd(self):
        line = self.filenames[0].split()
        scene_name = line[0].split("/")[1]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            "data_depth_annotated/",
            scene_name,
            "proj_depth/groundtruth/image_0{}/{:010d}.png".
            format("2", int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "raw",
            folder,
            "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        scene_name = folder.split("/")[1]
        velo_filename = os.path.join(
            self.data_path,
            "data_depth_annotated/",
            scene_name,
            "proj_depth/velodyne_raw/image_0{}/{:010d}.png".
            format(self.side_map[side], int(frame_index)))

        depth_gt = self.depth_png_read(velo_filename)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_depth_sd(self, folder, frame_index, side, do_flip):
        scene_name = folder.split("/")[1]

        velo_filename = os.path.join(
            self.data_path,
            "data_depth_annotated/",
            scene_name,
            "proj_depth/groundtruth/image_0{}/{:010d}.png".
            format(self.side_map[side], int(frame_index)))

        depth_gt = self.depth_png_read(velo_filename)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def depth_png_read(self, filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)
        depth = depth_png.astype(np.float) / 256.

        return depth

class DAT_VAL_TEST(Dataset):
    def __init__(self,
                 data_path,
                 is_test=False,
                 min_depth=0.0,
                 max_depth=85,
                 crop_h=350,
                 crop_w=1200):
        super(DAT_VAL_TEST, self).__init__()
        self.data_path = data_path
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.crop_h = crop_h
        self.crop_w = crop_w

        self.to_tensor = transforms.ToTensor() # (H x W x C) [0, 255] -> (C x H x W) [0.0, 1.0]
        self.loader = pil_loader

        self.has_gt = not is_test
        if is_test:
            self.dir = 'test_depth_completion_anonymous'
            self.filenames = ['{:010d}'.format(i) for i in range(1000)]
        else:
            self.dir = 'val_selection_cropped'
            self.filenames = readlines('splits/val.txt')

    def bottom_crop_input_data(self, inputs: dict, crop_h, crop_w):
        keys = ['color', 'depth_gt', 'depth_sd_gt', 'depth_color', 'mask']
        for k,v in inputs.items():
            # v in format [C,H,W]
            if k in keys:
                h = v.shape[1]
                w = v.shape[2]
                i = h - crop_h
                j = int(round((w - crop_w) / 2.))
                inputs[k] = v[:, i:i + crop_h, j:j + crop_w]


    def __getitem__(self, index):
        inputs = {}
        filename = self.filenames[index]

        color = self.get_image(filename)
        inputs['color'] = self.to_tensor(color)

        f_str = "{}.png".format(filename)
        if self.has_gt:
            fname = f_str.replace("XXX", "velodyne_raw")
        else:
            fname = f_str
        velo_path = os.path.join(
            self.data_path,
            self.dir, "velodyne_raw",
            fname)
        depth = self.get_depth(velo_path)
        depth = depth.astype(np.float32)
        inputs['depth_gt'] = torch.from_numpy(depth).unsqueeze(0)

        inputs['mask'] = (inputs["depth_gt"] > 0).float()

        if self.has_gt:
            fname = f_str.replace("XXX", "groundtruth_depth")
            gt_path = os.path.join(
                self.data_path,
                self.dir, "groundtruth_depth",
                fname)
            gtdepth = self.get_depth(gt_path)
            gtdepth = gtdepth.astype(np.float32)
            inputs['depth_sd_gt'] = torch.from_numpy(gtdepth).unsqueeze(0)

        inputs['frame_index'] = filename

        self.bottom_crop_input_data(inputs, self.crop_h, self.crop_w)

        sdepth = inputs['depth_gt'].unsqueeze(0) # (1, 1, H, W)
        mask = inputs['mask'].unsqueeze(0)
        sdepthx2, maskx2 = depth_down(sdepth, mask, 2), F.max_pool2d(mask, 2, 2)
        sdepthx4, maskx4 = depth_down(sdepth, mask, 4), F.max_pool2d(maskx2, 2, 2)
        sdepthx8, maskx8 = depth_down(sdepth, mask, 8), F.max_pool2d(maskx4, 2, 2)
        #
        aff_locsx1, aff_nnidxsx1 = gen_affinity(sdepth, mask)
        inputs['aff_locsx1'], inputs['aff_nnidxsx1'] = aff_locsx1[0], aff_nnidxsx1[0]
        aff_locsx2, aff_nnidxsx2 = gen_affinity(sdepthx2, maskx2)
        inputs['aff_locsx2'], inputs['aff_nnidxsx2'] = aff_locsx2[0], aff_nnidxsx2[0]
        aff_locsx4, aff_nnidxsx4 = gen_affinity(sdepthx4, maskx4)
        inputs['aff_locsx4'], inputs['aff_nnidxsx4'] = aff_locsx4[0], aff_nnidxsx4[0]
        aff_locsx8, aff_nnidxsx8 = gen_affinity(sdepthx8, maskx8)
        inputs['aff_locsx8'], inputs['aff_nnidxsx8'] = aff_locsx8[0], aff_nnidxsx8[0]
        inputs['maskx2'] = maskx2[0]
        inputs['maskx4'] = maskx4[0]
        inputs['maskx8'] = maskx8[0]

        return inputs

    def __len__(self):
        return len(self.filenames)

    def get_image(self, filename):
        f_str = "{}.png".format(filename)
        if self.has_gt:
            fname = f_str.replace("XXX", "image")
        else:
            fname = f_str

        image_path = os.path.join(
            self.data_path,
            self.dir, "image",
            fname)
        color = self.loader(image_path)
        return color

    def get_depth(self, path):
        depth_gt = self.depth_read(path)
        depth_gt = norm_depth(depth_gt, self.min_depth, self.max_depth)

        return depth_gt

    def depth_read(self, filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)
        depth = depth_png.astype(np.float) / 256.

        return depth


from torch.utils.data import DataLoader
from options import CompletionOptions
import matplotlib.pyplot as plt

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def plt_img(fig, h, w, i, p, img, itype='color'):
    # itype: color, gray, depth
    fig.add_subplot(h, w, i)
    timg = img.numpy()
    timg = timg.transpose([1,2,0])
    if itype == 'color':
        cmap = None
    elif itype == 'gray':
        timg = timg.squeeze(2)
        cmap = plt.get_cmap('gray')
    elif itype == 'depth':
        timg = timg.squeeze(2)
        timg = (timg - np.min(timg)) / (np.max(timg) - np.min(timg))
        mask = timg > 0
        mask = np.expand_dims(mask, -1).astype(np.float32)
        timg = 255 * plt.cm.jet(timg)[:, :, :3]  # H, W, C
        timg = timg * mask + (1-mask) * 255
        timg = timg.astype('uint8')
        cmap = None
    elif itype == 'binary':
        timg = timg.squeeze(2)
        cmap = plt.get_cmap('binary')
    p.imshow(timg, cmap=cmap)
    p.xticks([0, timg.shape[1]])
    p.yticks([0, timg.shape[0]])


def _merge_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}

    e_keys = ['folder', 'frame_index', 'side']
    for key, elems in example_merged.items():
        if key in e_keys or 'aff_' in key:
            ret[key] = elems
        else:
            # [A,B] + [A,B] -> [2,A,B]
            ret[key] = torch.stack(elems, dim=0)

    return ret

class TestR:
    def __init__(self):
        super(TestR, self).__init__()
        options = CompletionOptions()
        print(sys.argv)
        print(sys.argv[2:])
        self.opt = options.parse(sys.argv[2:-1])
        self.dataset = KITTIRAWDataset
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        self.opt.batch_size = 2
        self.opt.num_workers = 1
        self.opt.crop_h = 256
        self.opt.crop_w = 1216
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.crop_h, self.opt.crop_w,
            img_ext=img_ext, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True, collate_fn=_merge_batch,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.crop_h, self.opt.crop_w,
            img_ext=img_ext, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False, collate_fn=_merge_batch,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

    def run(self):
        for loader in [self.train_loader, self.val_loader]:
            for batch_idx, inputs in enumerate(loader):
                print(batch_idx, inputs.keys())

                fig = plt.figure(num=batch_idx, figsize=(12, 4))
                plt_img(fig, 3, 2, 1, plt, inputs['mask'][0], itype='binary')
                plt_img(fig, 3, 2, 2, plt, inputs['color_aug'][0])
                plt_img(fig, 3, 2, 3, plt, inputs['depth_aug'][0], itype='depth')
                plt_img(fig, 3, 2, 4, plt, inputs['depth_sd_gt'][0], itype='depth')
                plt_img(fig, 3, 2, 5, plt, inputs['color'][0])

                print(inputs['folder'][0], inputs['frame_index'][0], inputs['side'][0])
                print(inputs['folder'][1], inputs['frame_index'][1], inputs['side'][1])
                print('aff_locsx2', inputs['aff_locsx2'][0].shape)
                print('aff_nnidxsx8', inputs['aff_nnidxsx8'][1].shape)

                plt.tight_layout()
                plt.show()

class TestC:
    def __init__(self):
        super(TestC, self).__init__()
        options = CompletionOptions()
        print(sys.argv)
        print(sys.argv[2:])
        self.opt = options.parse(sys.argv[2:-1])
        self.dataset = DAT_VAL_TEST
        self.opt.data_path = '/data/KITTI_DAT'

        self.opt.batch_size = 2
        self.opt.num_workers = 1
        self.test_dataset = self.dataset(
            self.opt.data_path, is_test=True,
            min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
        self.test_loader = DataLoader(
            self.test_dataset, self.opt.batch_size, False, collate_fn=_merge_batch,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        print("There are {:d} items\n".format(len(self.test_dataset)))

    def run(self):
        for loader in [self.test_loader]:
            for batch_idx, inputs in enumerate(loader):
                print(batch_idx, inputs.keys())

                fig = plt.figure(num=batch_idx)
                plt_img(fig, 2, 2, 1, plt, inputs['color'][0])
                plt_img(fig, 2, 2, 2, plt, inputs['depth_gt'][0], itype='depth')
                plt_img(fig, 2, 2, 3, plt, inputs['mask'][0], itype='binary')
                if self.test_dataset.has_gt:
                    plt_img(fig, 2, 2, 4, plt, inputs['depth_sd_gt'][0], itype='depth')

                print('aff_locsx4', inputs['aff_locsx4'][0].shape)
                print('aff_nnidxsx2', inputs['aff_nnidxsx2'][1].shape)

                plt.tight_layout()
                plt.show()


def testr():
    test = TestR()
    test.run()

def testc():
    test = TestC()
    test.run()


if __name__ == "__main__":
    fire.Fire()
