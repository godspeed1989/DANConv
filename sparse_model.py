from __future__ import absolute_import, division, print_function
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

KNN=9

class AffConv(nn.Module):
    def __init__(self, inC, outC, knn=KNN, bias=True, use_center=True, use_locs=False):
        super(AffConv, self).__init__()
        # relative location of neighbors
        self.use_locs = use_locs
        locs_width = 2 if use_locs else 0
        # include center's feature
        self.use_center = use_center
        cntr_width = inC if use_center else 0
        self.ln = nn.Linear((inC+locs_width)*knn + cntr_width, outC, bias=bias)

    def forward(self, feats, aff_idx, locs=None):
        '''
        feats       n*(M,inC)
        aff_idx     n*(M,knn)
        locs        n*(M,2)
        output:
            n*(M,outC)
        '''
        ret = []
        n = len(feats)
        dist = 10.
        for i in range(n):
            c = feats[i].size(1)
            m = aff_idx[i].size(0)
            knn = aff_idx[i].size(1)
            idx = aff_idx[i].flatten().long()
            #
            knn_feat = feats[i][idx, :].reshape(m,knn,c)
            #
            if self.use_locs:
                knn_locs = locs[i][idx, :].reshape(m,knn,2)
                loc = locs[i].unsqueeze(1)
                knn_locs = (knn_locs - loc)/(dist+1)
                knn_locs = torch.clamp(knn_locs, -1, 1)
                knn_feat = torch.cat([knn_locs, knn_feat], 2).reshape(m,knn*(c+2))
            else:
                knn_feat = knn_feat.reshape(m,knn*c)
            if self.use_center:
                knn_feat = torch.cat([knn_feat, feats[i]], 1)
            knn_feat = F.relu(self.ln(knn_feat))
            ret.append(knn_feat)
        return ret

def deconv(in_channels, out_channels, kernel_size=4, padding=1, stride=2, relu=True):
    layers = []
    layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

def conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, relu=True):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, channels=64, downsample=False):
        super(ResBlock, self).__init__()

        if downsample:
            stride = 2
        else:
            stride = 1
        self.conv0 = conv2d(in_channels, channels, stride=stride)
        self.conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.conv2 = conv2d(channels, channels, relu=False)

    def forward(self, feat):
        feat0 = self.conv0(feat)
        feat1 = self.conv1(feat)

        feat2 = self.conv2(feat0)

        return F.relu_(feat2+feat1)

def gather_feat(img, locs, ver=1):
    # img (b,c,h,w)
    # locs b*(m,2)
    # return:
    #   b*(m,c)
    feats = []
    for i,loc in enumerate(locs):
        # version 1: gather
        if ver == 1:
            feat = img[i,:,loc[:,0],loc[:,1]]

        # version 2: interpolate
        if ver == 2:
            # (1,c,h,w)
            img1 = img[i:i+1]
            # normalize xy to [-1,1]
            h, w = img.size(2), img.size(3)
            xy = loc.to(img.dtype)
            xy[:,0] = xy[:,0] / (h - 1.0) * 2.0 - 1.0
            xy[:,1] = xy[:,1] / (w - 1.0) * 2.0 - 1.0
            xy[:, [0, 1]] = xy[:, [1, 0]]
            # (m,2)->(1,1,m,2)
            xy = xy.unsqueeze(0).unsqueeze(0)
            # (c,1,m)
            feat = F.grid_sample(img1, xy, align_corners=False)[0]
            feat = feat.squeeze(1)

        feats.append(feat.transpose(0,1))
    return feats

def scatter_feat(feats, locs, h, w):
    # feats b*(m,c)
    # locs b*(m,2)
    n = len(feats)
    c = feats[0].size(1)
    dev = feats[0].device
    img = torch.zeros(n,c,h,w).float().to(dev)
    for i,loc in enumerate(locs):
        feat = feats[i].transpose(0,1)
        img[i,:,loc[:,0],loc[:,1]] = feat
    return img

class FUSE_Layer(nn.Module):
    def __init__(self, channels):
        super(FUSE_Layer, self).__init__()
        self.ic, self.pc = channels, channels
        rc = self.pc
        self.fc1 = nn.Linear(self.ic+self.pc, rc)

        self.fc3 = nn.Linear(rc, 1)
        self.fc4 = nn.Linear(rc, 1)
        self.fc5 = nn.Linear(self.ic, self.ic)
        self.fc6 = nn.Linear(self.pc, self.pc)

    def forward(self, img_feas, point_feas):
        batch = len(img_feas)
        for i in range(batch):
            imf = img_feas[i]
            ptf = point_feas[i]
            fuse = torch.cat((imf, ptf), 1)
            fuse = F.dropout(F.relu(self.fc1(fuse)), training=self.training)
            att1 = torch.sigmoid(self.fc3(fuse))
            att2 = torch.sigmoid(self.fc4(fuse))
            imfeat = imf + ptf * att1
            ptfeat = ptf + imf * att2
            img_feas[i] = F.relu(self.fc5(imfeat))
            point_feas[i] = F.relu(self.fc6(ptfeat))
        return img_feas, point_feas

class CoAttnBlock(nn.Module):
    def __init__(self, in_channels=64, channels=64,
                 downsample=False, affconv_use_locs=False):
        super(CoAttnBlock, self).__init__()

        self.downsample = downsample
        if downsample:
            stride = 2
        else:
            stride = 1
        self.d_conv0 = conv2d(in_channels, channels, stride=stride)
        self.d_conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.d_conv2 = conv2d(channels, channels, relu=False)
        self.r_conv0 = conv2d(in_channels, channels, stride=stride)
        self.r_conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.r_conv2 = conv2d(channels, channels, relu=False)

        self.affconv_d = AffConv(in_channels, channels, use_locs=affconv_use_locs)
        self.affconv_r = AffConv(in_channels, channels, use_locs=affconv_use_locs)
        self.crossfuse = FUSE_Layer(channels)

    def forward(self, d_feat, r_feat, masks, locs, nnidxs):
        d_feat0 = self.d_conv0(d_feat)
        d_feat1 = self.d_conv1(d_feat)
        r_feat0 = self.r_conv0(r_feat)
        r_feat1 = self.r_conv1(r_feat)

        h, w = d_feat0.size(2), d_feat0.size(3)
        d_feat0_dis = gather_feat(d_feat0, locs)
        d_feat_new = self.affconv_d(d_feat0_dis, nnidxs, locs)
        r_feat0_dis = gather_feat(r_feat0, locs)
        r_feat_new = self.affconv_r(r_feat0_dis, nnidxs, locs)

        d_feat_new, r_feat_new = self.crossfuse(d_feat_new, r_feat_new)

        d_feat_new = scatter_feat(d_feat_new, locs, h, w)
        r_feat_new = scatter_feat(r_feat_new, locs, h, w)

        d_feat0 = (1 - masks) * d_feat0 + d_feat_new.contiguous()
        r_feat0 = (1 - masks) * r_feat0 + r_feat_new.contiguous()

        d_feat2 = self.d_conv2(d_feat0)
        r_feat2 = self.r_conv2(r_feat0)

        return F.relu_(d_feat2+d_feat1), F.relu_(r_feat2+r_feat1)

class Model(nn.Module):
    def __init__(self, scales=4, base_width=32, outc=1):
        super(Model, self).__init__()

        self.d_conv00 = AffConv(1, base_width, use_locs=False)
        self.r_conv00 = conv2d(3, base_width)
        self.r_conv01 = conv2d(base_width, base_width)

        # Encoder
        self.attblock10 = CoAttnBlock(base_width, base_width, downsample=True)
        self.attblock11 = CoAttnBlock(base_width, base_width, downsample=False)
        self.attblock20 = CoAttnBlock(base_width, base_width, downsample=True)
        self.attblock21 = CoAttnBlock(base_width, base_width, downsample=False)
        self.attblock30 = CoAttnBlock(base_width, base_width, downsample=True)
        self.attblock31 = CoAttnBlock(base_width, base_width, downsample=False)

        # Decoder
        channels = base_width
        self.d_gate4 = conv2d(channels, channels, relu=False)
        self.d_resblock40 = ResBlock(channels*2, channels, False)
        self.d_resblock41 = ResBlock(channels, channels, False)
        self.d_deconv3 = deconv(channels, channels)
        self.d_gate3 = conv2d(channels, channels, relu=False)
        self.d_resblock50 = ResBlock(channels*3, channels, False)
        self.d_resblock51 = ResBlock(channels, channels, False)
        self.d_deconv2 = deconv(channels, channels)
        self.d_gate2 = conv2d(channels, channels, relu=False)
        self.d_resblock60 = ResBlock(channels*3, channels, False)
        self.d_resblock61 = ResBlock(channels, channels, False)
        self.d_deconv1 = deconv(channels, channels)
        self.d_gate1 = conv2d(channels, channels, relu=False)
        self.d_last_conv = conv2d(channels*3, 32)
        self.d_out = nn.Conv2d(32, outc, kernel_size=1, padding=0)

        self.r_gate4 = conv2d(channels, channels, relu=False)
        self.r_resblock40 = ResBlock(channels*2, channels, False)
        self.r_resblock41 = ResBlock(channels, channels, False)
        self.r_deconv3 = deconv(channels, channels)
        self.r_gate3 = conv2d(channels, channels, relu=False)
        self.r_resblock50 = ResBlock(channels*3, channels, False)
        self.r_resblock51 = ResBlock(channels, channels, False)
        self.r_deconv2 = deconv(channels, channels)
        self.r_gate2 = conv2d(channels, channels, relu=False)
        self.r_resblock60 = ResBlock(channels*3, channels, False)
        self.r_resblock61 = ResBlock(channels, channels, False)
        self.r_deconv1 = deconv(channels, channels)
        self.r_gate1 = conv2d(channels, channels, relu=False)
        self.r_last_conv = conv2d(channels*3, 32)
        self.r_out = nn.Conv2d(32, outc, kernel_size=1, padding=0)

        self.f_conv4_1 = conv2d(channels*2, channels, relu=False)
        self.f_conv4_2 = nn.Sequential(conv2d(channels, channels, stride=2), conv2d(channels, channels, relu=False))
        self.f_deconv3 = deconv(channels, channels)
        self.f_conv3_1 = conv2d(channels*3, channels, relu=False)
        self.f_conv3_2 = nn.Sequential(conv2d(channels, channels, stride=2), conv2d(channels, channels, relu=False))
        self.f_deconv2 = deconv(channels, channels)
        self.f_conv2_1 = conv2d(channels*3, channels, relu=False)
        self.f_conv2_2 = nn.Sequential(conv2d(channels, channels, stride=2), conv2d(channels, channels, relu=False))
        self.f_deconv1 = deconv(channels, channels)
        self.f_conv1_1 = conv2d(channels+64, 32, relu=False)
        self.f_conv1_2 = nn.Sequential(conv2d(32, 32, stride=2), conv2d(32, 32, relu=False))
        self.f_out = nn.Conv2d(32, outc, kernel_size=1, padding=0)

    def forward(self, sdepth, mask, img, ext={}):
        h, w = sdepth.size(2), sdepth.size(3)

        d_feat0 = gather_feat(sdepth, ext['aff_locsx1'])
        d_feat0 = scatter_feat(self.d_conv00(d_feat0, ext['aff_nnidxsx1'], ext['aff_locsx1']), ext['aff_locsx1'], h, w)

        r_feat0 = self.r_conv00(img)
        r_feat0 = self.r_conv01(r_feat0)

        # Encoder
        # x1 -> x2
        d_feat1, r_feat1 = self.attblock10(d_feat0, r_feat0, ext['maskx2'], ext['aff_locsx2'], ext['aff_nnidxsx2'])
        d_feat1, r_feat1 = self.attblock11(d_feat1, r_feat1, ext['maskx2'], ext['aff_locsx2'], ext['aff_nnidxsx2'])
        # x2 -> x4
        d_feat2, r_feat2 = self.attblock20(d_feat1, r_feat1, ext['maskx4'], ext['aff_locsx4'], ext['aff_nnidxsx4'])
        d_feat2, r_feat2 = self.attblock21(d_feat2, r_feat2, ext['maskx4'], ext['aff_locsx4'], ext['aff_nnidxsx4'])
        # x4 -> x8
        d_feat3, r_feat3 = self.attblock30(d_feat2, r_feat2, ext['maskx8'], ext['aff_locsx8'], ext['aff_nnidxsx8'])
        d_feat3, r_feat3 = self.attblock31(d_feat3, r_feat3, ext['maskx8'], ext['aff_locsx8'], ext['aff_nnidxsx8'])

        # Decoder
        # x8 SGFM - Symmetric Gated Fusion
        d_gate4 = torch.sigmoid(self.d_gate4(d_feat3))
        d_feat = self.d_resblock40(torch.cat([d_feat3, d_gate4*r_feat3], 1))
        d_feat = self.d_resblock41(d_feat)
        r_gate4 = torch.sigmoid(self.r_gate4(r_feat3))
        r_feat = self.r_resblock40(torch.cat([r_feat3, r_gate4*d_feat3], 1))
        r_feat = self.r_resblock41(r_feat)
        # x8 -> x4
        f_feat = self.f_conv4_1(torch.cat([d_feat, r_feat], 1))
        f_feat_res = F.interpolate(self.f_conv4_2(F.relu(f_feat)), scale_factor=2, mode='bilinear', align_corners=True)
        f_feat = self.f_deconv3(F.relu_(f_feat + f_feat_res))

        # x4 SGFM
        d_ufeat3 = self.d_deconv3(d_feat)
        r_ufeat3 = self.r_deconv3(r_feat)
        d_gate3 = torch.sigmoid(self.d_gate3(d_ufeat3))
        d_feat = self.d_resblock50(torch.cat([d_feat2, d_ufeat3, d_gate3*r_feat2], 1))
        d_feat = self.d_resblock51(d_feat)
        r_gate3 = torch.sigmoid(self.r_gate3(r_ufeat3))
        r_feat = self.r_resblock50(torch.cat([r_feat2, r_ufeat3, r_gate3*d_feat2], 1))
        r_feat = self.r_resblock51(r_feat)
        # x4 -> x2
        f_feat = self.f_conv3_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv3_2(F.relu(f_feat)), scale_factor=2, mode='bilinear', align_corners=True)
        f_feat = self.f_deconv2(F.relu_(f_feat + f_feat_res))

        # x2 SGFM
        d_ufeat2 = self.d_deconv2(d_feat)
        r_ufeat2 = self.r_deconv2(r_feat)
        d_gate2 = torch.sigmoid(self.d_gate2(d_ufeat2))
        d_feat = self.d_resblock60(torch.cat([d_feat1, d_ufeat2, d_gate2*r_feat1], 1))
        d_feat = self.d_resblock61(d_feat)
        r_gate2 = torch.sigmoid(self.r_gate2(r_ufeat2))
        r_feat = self.r_resblock60(torch.cat([r_feat1, r_ufeat2, r_gate2*d_feat1], 1))
        r_feat = self.r_resblock61(r_feat)
        # x2 -> x1
        f_feat = self.f_conv2_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv2_2(F.relu(f_feat)), scale_factor=2, mode='bilinear', align_corners=True)
        f_feat = self.f_deconv1(F.relu_(f_feat + f_feat_res))

        # x1 End-integration
        d_ufeat1 = self.d_deconv1(d_feat)
        r_ufeat1 = self.r_deconv1(r_feat)
        d_gate1 = torch.sigmoid(self.d_gate1(d_ufeat1))
        d_feat = torch.cat((d_feat0, d_ufeat1, d_gate1*r_feat0), 1)
        r_gate1 = torch.sigmoid(self.r_gate1(r_ufeat1))
        r_feat = torch.cat((r_feat0, r_ufeat1, r_gate1*d_feat0), 1)

        d_feat = self.d_last_conv(d_feat)
        r_feat = self.r_last_conv(r_feat)

        f_feat = self.f_conv1_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv1_2(F.relu(f_feat)), scale_factor=2, mode='bilinear', align_corners=True)
        f_feat = F.relu_(f_feat + f_feat_res)

        d_out = self.d_out(d_feat)
        r_out = self.r_out(r_feat)

        f_out = self.f_out(f_feat)

        out = [f_out, d_out, r_out]

        # pred = out[0]
        return out

from kitti_dataset import depth_down, gen_affinity

if __name__ == "__main__":
    dev = 'cpu'
    if torch.has_cuda and torch.cuda.is_available():
        dev = 'cuda'
    sdepth = torch.rand(1, 1, 512, 256).to(dev)
    sdepth = torch.abs(sdepth)
    mask = torch.rand(1, 1, 512, 256).to(dev) > 0.5
    mask = mask.float()
    img = torch.rand(1, 3, 512, 256).to(dev)
    inputs = {}
    sdepthx2, maskx2 = depth_down(sdepth, mask, 2), F.max_pool2d(mask, 2, 2)
    sdepthx4, maskx4 = depth_down(sdepth, mask, 4), F.max_pool2d(mask, 4, 4)
    sdepthx8, maskx8 = depth_down(sdepth, mask, 8), F.max_pool2d(mask, 8, 8)
    aff_locsx1, aff_nnidxsx1 = gen_affinity(sdepth, mask, knn=KNN)
    inputs['aff_locsx1'], inputs['aff_nnidxsx1'] = aff_locsx1, aff_nnidxsx1
    aff_locsx2, aff_nnidxsx2 = gen_affinity(sdepthx2, maskx2, knn=KNN)
    inputs['aff_locsx2'], inputs['aff_nnidxsx2'] = aff_locsx2, aff_nnidxsx2
    aff_locsx4, aff_nnidxsx4 = gen_affinity(sdepthx4, maskx4, knn=KNN)
    inputs['aff_locsx4'], inputs['aff_nnidxsx4'] = aff_locsx4, aff_nnidxsx4
    aff_locsx8, aff_nnidxsx8 = gen_affinity(sdepthx8, maskx8, knn=KNN)
    inputs['aff_locsx8'], inputs['aff_nnidxsx8'] = aff_locsx8, aff_nnidxsx8
    inputs['maskx2'] = maskx2
    inputs['maskx4'] = maskx4
    inputs['maskx8'] = maskx8

    #
    mod = Model(scales=4, base_width=64).to(dev)
    d = mod(sdepth, mask, img, inputs)
    print('Model', d[0].shape) # [1, 1, 512, 256]
    print(" - parameters is {}M".format(
            sum(tensor.numel() for tensor in mod.parameters())/1e6) )
    from thop import profile, clever_format
    def profile_model(net, inputs):
        flops, params = profile(mod, inputs, verbose=False)
        flops, params = clever_format([flops, params], "%.2f")
        print(flops, '&', params)
    profile_model(mod, inputs=(sdepth, mask, img, inputs))
