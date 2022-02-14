import sys
sys.path.append('.')
import time
import megengine as mge
from model.RIFE import Model

model = Model()
model.eval()

I0 = mge.random(1, 3, 480, 640)
I1 = mge.random(1, 3, 480, 640)
for i in range(100):
    pred = model.inference(I0, I1)
mge._full_sync()
time_stamp = time.time()
for i in range(100):
    pred = model.inference(I0, I1)
mge._full_sync()
print((time.time() - time_stamp) / 100)
