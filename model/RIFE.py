import megengine as mge
import megengine.optimizer as optim
from model.IFNet import IFNet
from model.IFNet_m import IFNet_m
import megengine.distributed as dist
import megengine.autodiff as ad
from model.loss import *
from model.laplacian import *
from model.refine import *


class Model:
    def __init__(self, local_rank=-1, arbitrary=False, simply_infer=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.optimG = optim.AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()

        if simply_infer == False:
            dist.bcast_list_(self.flownet.tensors())
            self.gm = ad.GradManager()
            self.gm.attach(self.flownet.parameters(), callbacks=[dist.make_allreduce_cb("mean")])

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def load_model(self, path, rank=0):
        if rank <= 0:
            self.flownet.load_state_dict(mge.load('{}/flownet.pkl'.format(path)))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            mge.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        imgs = F.concat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def mix_face(self, img0, img1, timestep=0.5, scale=1, gamma=1):
        imgs = F.concat((img0, img1), 1) # [1, 6, 512, 512]
        scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        ret = self.flownet(imgs, scale = scale_list, timestep = timestep)
        flow, mask, merged = ret[:3]
        timestep = 1 - (F.abs(flow[-1]).mean(1, True) / F.abs(flow[-1]).mean(1, True).max()).detach()
        timestep = timestep ** gamma
        ret = self.flownet(imgs, scale = scale_list, timestep = timestep)
        flow, mask, merged = ret[:3]
        return merged[2]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None, timestep=0.5):
        imgs = mge.Tensor(imgs)
        gt = mge.Tensor(gt)
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            with self.gm:
                self.train()
                flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(F.concat((imgs, gt), 1), scale=[4, 2, 1], timestep = timestep)
                loss_l1 = (self.lap(merged[2], gt)).mean()
                loss_tea = (self.lap(merged_teacher, gt)).mean()
                loss_G = loss_l1 + loss_tea + loss_distill * 0.002
                self.gm.backward(loss_G)
            optim.clip_grad_norm(self.flownet.parameters(), 1.0)
            self.optimG.step().clear_grad()
        else:
            self.eval()
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(F.concat((imgs, gt), 1), scale=[4, 2, 1], timestep = timestep)
            loss_l1 = (self.lap(merged[2], gt)).mean()
            loss_tea = (self.lap(merged_teacher, gt)).mean()
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }
