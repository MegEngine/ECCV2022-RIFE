import megengine as mge
from megengine import module as M
import megengine.functional as F


class EPE(M.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)

class SOBEL(M.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = mge.Tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).astype("float32")
        self.kernelY = self.kernelX.T
        self.kernelX = F.expand_dims(F.expand_dims(self.kernelX, axis=0),axis=0)
        self.kernelY = F.expand_dims(F.expand_dims(self.kernelY, axis=0),axis=0)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = F.concat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.nn.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.nn.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = F.abs(pred_X-gt_X), F.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

