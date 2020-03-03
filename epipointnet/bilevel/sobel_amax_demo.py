import matplotlib.pyplot as plt
import torch.nn
from torch.optim import SGD

from epipointnet.bilevel.conv import Sobel2DArgMax


class DrawLayer(torch.nn.Module):

    def __init__(self, size: torch.Tensor):
        super(DrawLayer, self).__init__()
        center_init = torch.rand((1, 1, 2), dtype=torch.float32, requires_grad=True) * size
        self.center = torch.nn.Parameter(center_init, True)

    def forward(self, coord_tensor):
        # coord_tensor should be a 2d meshgrid starting from 0 of shape hxwx2. coord_tensor[y, x, :] = [y, x]
        dist_tensor = coord_tensor - self.center
        dist_tensor = torch.sum(torch.pow(dist_tensor, 2), dim=2)
        dist_tensor = dist_tensor / torch.max(dist_tensor)
        return -1 * dist_tensor + 1


def main():
    size = (100.0, 100.0)
    size_tensor = torch.tensor(size)
    y_axis = torch.arange(size[0], dtype=torch.float32)
    x_axis = torch.arange(size[1], dtype=torch.float32)

    y_grid, x_grid = torch.meshgrid([y_axis, x_axis])

    coord_tensor = torch.stack((y_grid, x_grid), dim=-1)  # r, c coord tensor

    target = torch.randint(90, (2,), dtype=torch.long) + 5
    with torch.autograd.detect_anomaly():
        drawLayer = DrawLayer(size_tensor)

        drawLayer = drawLayer.cuda()
        coord_tensor = coord_tensor.cuda()
        target = target.cuda()

        optimizer = SGD(drawLayer.parameters(True), lr=1)

        while True:
            plt.cla()
            img = drawLayer(coord_tensor)
            # noise = torch.randn(img.shape, device=img.device) / 1000
            # img = img + noise
            plt.imshow(img.clone().detach().cpu().numpy())

            target_plt = target.clone().detach().cpu().numpy()
            plt.scatter(target_plt[0], target_plt[1], c="b", s=50)
            img_bchw = img.unsqueeze(0).unsqueeze(0)
            amax_b2c = Sobel2DArgMax.apply(img_bchw)  # type: torch.Tensor
            amax_2 = amax_b2c.squeeze()

            amax_2_plt = amax_2.clone().detach().cpu().numpy()
            plt.scatter(amax_2_plt[0], amax_2_plt[1], marker="x", c="r", s=50)
            plt.pause(.001)

            loss = torch.sum(torch.pow(target - amax_2, 2))
            optimizer.zero_grad()
            loss.backward()

            if loss < 1e-4:
                target = torch.randint(90, (2,), dtype=torch.float32, device=target.device) + 5
            else:
                optimizer.step()


if __name__ == "__main__":
    main()
