import megengine.functional as F

backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size))
    if k not in backwarp_tenGrid:
        tenHorizontal = F.broadcast_to(F.linspace(0, 1.0, tenFlow.shape[3]).reshape(
            1, 1, 1, tenFlow.shape[3]),(tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3])) * (tenFlow.shape[3] - 1)
        tenVertical = F.broadcast_to(F.linspace(0, 1.0, tenFlow.shape[2]).reshape(
            1, 1, tenFlow.shape[2], 1),(tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3])) * (tenFlow.shape[2] - 1)
        backwarp_tenGrid[k] = F.concat(
            [tenHorizontal, tenVertical], 1)

    g = (backwarp_tenGrid[k] + tenFlow).transpose(0, 2, 3, 1)
    return F.nn.remap(inp=tenInput, map_xy=g)
