import sys
sys.path.append('.')
import cv2
import math
import numpy as np
import megengine.functional as F
import megengine as mge
from model.RIFE import Model
from skimage.color import rgb2yuv
from yuv_frame_io import YUV_Read

model = Model(arbitrary=True)
model.load_model('train_log_m')
model.eval()

name_list = [
    ('HD_dataset/HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
    ('HD_dataset/HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
    ('HD_dataset/HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
    ('HD_dataset/HD1080p_GT/BlueSky.yuv', 1080, 1920),
    ('HD_dataset/HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
    ('HD_dataset/HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
    ('HD_dataset/HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
    ('HD_dataset/HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
    ('HD_dataset/HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
    ('HD_dataset/HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
    ('HD_dataset/HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
]
def inference(I0, I1, pad, multi=2, arbitrary=True):
    img = [I0, I1]
    if not arbitrary:
        for i in range(multi):
            res = [I0]
            for j in range(len(img) - 1):
                res.append(model.inference(img[j], img[j + 1]))
                res.append(img[j + 1])
            img = res
    else:
        img = [I0]
        p = 2**multi
        for i in range(p-1):
            img.append(model.inference(I0, I1, timestep=(i+1)*(1./p)))
        img.append(I1)
    for i in range(len(img)):
        img[i] = img[i][0][:, pad: -pad]
    return img[1: -1]
        
tot = []
for data in name_list:
    psnr_list = []
    name = data[0]
    h = data[1]
    w = data[2]
    if 'yuv' in name:
        Reader = YUV_Read(name, h, w, toRGB=True)
    else:
        Reader = cv2.VideoCapture(name)
    _, lastframe = Reader.read()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(name + '.mp4', fourcc, 30, (w, h))    
    for index in range(0, 100, 4):
        gt = []
        if 'yuv' in name:
            IMAGE1, success1 = Reader.read(index)
            IMAGE2, success2 = Reader.read(index + 4)
            if not success2:
                break
            for i in range(1, 4):
                tmp, _ = Reader.read(index + i)
                gt.append(tmp)
        else:
            print('Not Implement')
        I0 = F.expand_dims(mge.Tensor(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.), 0)
        I1 = F.expand_dims(mge.Tensor(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.), 0)
        
        if h == 720:
            pad = 24
        elif h == 1080:
            pad = 4
        else:
            pad = 16
        I0 = F.nn.pad(I0, ((0, 0), (0, 0), (pad, pad), (0, 0)), mode = 'edge')
        I1 = F.nn.pad(I1, ((0, 0), (0, 0), (pad, pad), (0, 0)), mode = 'edge')

        pred = inference(I0, I1, pad)
        for i in range(4 - 1):
            out = (np.round(pred[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
            if 'yuv' in name:
                diff_rgb = 128.0 + rgb2yuv(gt[i] / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
                mse = np.mean((diff_rgb - 128.0) ** 2)
                PIXEL_MAX = 255.0
                psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            else:
                print('Not Implement')
            psnr_list.append(psnr)
    print(np.mean(psnr_list))
    tot.append(np.mean(psnr_list))

print('PSNR: {}(544*1280), {}(720p), {}(1080p)'.format(np.mean(tot[7:11]), np.mean(tot[:3]), np.mean(tot[3:7])))
