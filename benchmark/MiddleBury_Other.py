import sys
sys.path.append('.')
import cv2
import megengine as mge
import megengine.functional as F
import numpy as np
from model.RIFE import Model

model = Model()
model.load_model('train_log')
model.eval()

name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
for i in name:
    i0 = cv2.imread('other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
    i1 = cv2.imread('other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
    gt = cv2.imread('other-gt-interp/{}/frame10i11.png'.format(i)) 
    h, w = i0.shape[1], i0.shape[2]
    imgs = F.zeros([1, 6, 480, 640])
    ph = (480 - h) // 2
    pw = (640 - w) // 2
    imgs[:, :3, :h, :w] = F.expand_dims(mge.Tensor(i0), 0).astype("float32")
    imgs[:, 3:, :h, :w] = F.expand_dims(mge.Tensor(i1), 0).astype("float32")
    I0 = imgs[:, :3]
    I2 = imgs[:, 3:]
    pred = model.inference(I0, I2)
    out = pred[0].detach().numpy().transpose(1, 2, 0)
    out = np.round(out[:h, :w] * 255)
    IE_list.append(np.abs((out - gt * 1.0)).mean())
    print(np.mean(IE_list))
