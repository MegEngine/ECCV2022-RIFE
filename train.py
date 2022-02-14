import os
import math
import time
import numpy as np
import random
import argparse

from model.RIFE import Model
from megengine.data import DataLoader, RandomSampler
from dataset import *
from tensorboardX import SummaryWriter
import megengine.distributed as dist

log_path = 'train_log'
exp = os.path.abspath('.').split('/')[-1]

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-5) * mul + 3e-5

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return F.clip(rgb_map, 0, 1)

def train(model, local_rank):
    if local_rank == 0:
        writer_val = SummaryWriter('validate_dataset_out')
        writer = SummaryWriter('train_dataset_out')
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = RandomSampler(dataset, batch_size=args.batch_size)
    train_data = DataLoader(dataset, num_workers=8, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_sampler = RandomSampler(dataset_val, batch_size=16, world_size=1, rank=0)
    val_data = DataLoader(dataset_val, num_workers=8, sampler=val_sampler)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        if nr_eval % 5 == 0:
            if local_rank == 0:
                evaluate(model, val_data, step, local_rank, writer_val)
            else:
                evaluate(model, val_data, step, local_rank)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu = data / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step)
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'].numpy(), step)
                writer.add_scalar('loss/tea', info['loss_tea'].numpy(), step)
                writer.add_scalar('loss/distill', info['loss_distill'].numpy(), step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.transpose(0, 2, 3, 1) * 255).astype('uint8')
                mask = (F.concat((info['mask'], info['mask_tea']), 3).transpose(0, 2, 3, 1).detach().numpy() * 255).astype('uint8')
                pred = (pred.transpose(0, 2, 3, 1).detach().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].transpose(0, 2, 3, 1).detach().numpy() * 255).astype('uint8')
                flow0 = info['flow'].transpose(0, 2, 3, 1).detach().numpy()
                flow1 = info['flow_tea'].transpose(0, 2, 3, 1).detach().numpy()
                for i in range(5):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1'].numpy()))
            step += 1
        nr_eval += 1
        model.save_model(log_path, local_rank)  

def evaluate(model, val_data, nr_eval, local_rank, writer_val = None):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu = data / 255.
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]

        pred, info = model.update(imgs, gt, training=False)
        pred = pred.detach()
        merged_img = info['merged_tea']

        loss_l1_list.append(info['loss_l1'].numpy())
        loss_tea_list.append(info['loss_tea'].numpy())
        loss_distill_list.append(info['loss_distill'].numpy())
        for j in range(gt.shape[0]):
            gt = gt.astype(pred.dtype)
            psnr = -10 * math.log10(F.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).numpy())
            psnr_list.append(psnr)
            psnr = -10 * math.log10(F.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).numpy())
            psnr_list_teacher.append(psnr)
        gt = (gt.transpose(0, 2, 3, 1) * 255).astype('uint8')
        pred = (pred.transpose(0, 2, 3, 1) * 255).astype('uint8')
        merged_img = (merged_img.transpose(0, 2, 3, 1).numpy() * 255).astype('uint8')
        flow0 = info['flow'].transpose(0, 2, 3, 1).numpy()
        flow1 = info['flow_tea'].transpose(0, 2, 3, 1).numpy()
        if i == 0 and local_rank == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs, nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]).numpy(), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
    
@dist.launcher(world_size=4)
def main():
    rank = dist.get_rank()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    global args
    args = parser.parse_args()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    mge.random.seed(seed)
    model = Model(rank)
    train(model, rank)
        
if __name__ == "__main__":
    main()
