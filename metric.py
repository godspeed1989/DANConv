import torch

class Metrics(object):
    def __init__(self, max_depth=85.0, disp=False, normal=False):
        self.rmse, self.mae = 0, 0
        self.num = 0
        self.disp = disp
        self.max_depth = max_depth
        self.min_disp = 1.0/max_depth
        self.normal = normal

    def calculate(self, prediction, gt):
        valid_mask = (gt > 0).detach()

        self.num = valid_mask.sum().item()
        prediction = prediction[valid_mask]
        gt = gt[valid_mask]

        if self.disp:
            prediction = torch.clamp(prediction, min=self.min_disp)
            prediction = 1./prediction
            gt = 1./gt
        if self.normal:
            prediction = prediction * self.max_depth
            gt = gt * self.max_depth
        prediction = torch.clamp(prediction, min=0, max=self.max_depth)

        abs_diff = (prediction - gt).abs()
        self.rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
        self.mae = abs_diff.mean().item()

        # 1/km
        # iRMSE
        irmse = (1000 / gt - 1000 / prediction) ** 2
        self.irmse = torch.sqrt(irmse.mean()).item()

        # iMAE
        imae = torch.abs(1000 / gt - 1000 / prediction)
        self.imae = imae.mean().item()

    def get_metric(self, metric_name):
        return self.__dict__[metric_name]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
