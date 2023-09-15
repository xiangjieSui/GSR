from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import os
pi = np.pi


class Img2video(nn.Module):
    def __init__(
        self,
        img_hight,
        img_width,
        patch_size=32,  # pixel
        fragment_size=224,  # pixel
    ):
        super(Img2video, self).__init__()
        assert fragment_size % patch_size == 0, 'fragment_size % patch_size != 0'
        self.fragment_size = fragment_size
        self.patch_size = patch_size

        print('GSR configration')
        print('Image Size = ({}, {})'.format(img_hight, img_width))
        print('Patch Size = ({}, {})'.format(patch_size, patch_size))

        self.grid = GridGenerator(img_hight, img_width, (patch_size, patch_size))


    def forward(self, images, scanpaths, masking):
        return self.crop_sphere(images, scanpaths, masking)
       

    def crop_sphere(self, images, scanpaths, masking):
        device = scanpaths.device
        # operating on CPU
        scanpaths = scanpaths.detach().cpu().numpy()
        b, c, h, w = images.shape
        b, n_scanpaths, T_max, _ = scanpaths.shape
        n_fragment = self.fragment_size // self.patch_size

        vclips = torch.zeros(
            [b, T_max, c, self.fragment_size, self.fragment_size])

        scanpaths[:, :, :, 0] = scanpaths[:, :, :, 0] * h - 0.5
        scanpaths[:, :, :, 1] = scanpaths[:, :, :, 1] * w - 0.5

        for b_i in range(b):
            img_i = images[b_i]
            for t_i in range(masking[b_i]):
                vclip_t = []
                scanpaths_clip = scanpaths[b_i, :, t_i, :].astype(int)
                for path_i in range(n_scanpaths):
                    (y, x) = scanpaths_clip[path_i]
                    patch_cord = self.grid.tangentPatch(y, x)
                    patch_i = img_i[:, patch_cord[0, :, :],
                                    patch_cord[1, :, :]]
                    vclip_t.append(patch_i)

                # (n_fragment, n_fragment, c, patch_size, patch_size)
                vclip_t = torch.stack(vclip_t).reshape([n_fragment, n_fragment, c, self.patch_size, self.patch_size])

                # (c, fragment_size, fragment_size)
                vclip_t = rearrange(vclip_t, 'a b c d e -> c (a d) (b e)')

                vclips[b_i, t_i] = vclip_t

        # (B, T, c, fragment_size, fragment_size) -> (B, c, T, fragment_size, fragment_size)
        vclips = vclips.permute(0, 2, 1, 3, 4).to(device)

        return vclips



'''
The following code is source from:
Shen, Zhijie and Lin, Chunyu and Liao, Kang and Nie, Lang and Zheng, Zishuo and Zhao, Yao,
"PanoFormer: Panorama Transformer for Indoor 360° Depth Estimation", European Conference on Computer Vision, 2022, pp.195-211.
https://github.com/zhijieshen-bjtu/PanoFormer
'''


def genSamplingPattern(h, w, kh, kw, stride=1):
    gridGenerator = GridGenerator(h, w, (kh, kw), stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    grid = LonLatSamplingPattern
    with torch.no_grad():
        grid = torch.FloatTensor(grid)
        grid.requires_grad = False

    return grid


class GridGenerator:
    def __init__(self, height: int, width: int, kernel_size, stride=1):
        self.height = height
        self.width = width
        self.kernel_size = kernel_size  # (Kh, Kw)
        self.stride = stride  # (H, W)
        self.kerX, self.kerY = self.createKernel()  # (Kh, Kw)

    def tangentPatch(self, y, x):
        rho = np.sqrt(self.kerX ** 2 + self.kerY ** 2)
        Kh, Kw = self.kernel_size
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8
        nu = np.arctan(rho)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        lat = ((y / self.height) - 0.5) * np.pi
        lon = ((x / self.width) - 0.5) * (2 * np.pi)

        patch_lat = np.arcsin(cos_nu * np.sin(lat) +
                              self.kerY * sin_nu * np.cos(lat) / rho)
        patch_lon = np.arctan(
            self.kerX * sin_nu / (rho * np.cos(lat) * cos_nu - self.kerY * np.sin(lat) * sin_nu)) + lon

        # (radian) -> (index of pixel)
        patch_lat = (patch_lat / np.pi + 0.5) * self.height - 0.5
        patch_lon = ((patch_lon / (2 * np.pi) + 0.5)
                     * self.width) % self.width - 0.5

        # (2, Kh, Kw) = ((lat, lon), Kh, Kw)
        LatLon = np.stack((patch_lat, patch_lon))

        return LatLon

    def createSamplingPattern(self):
        """
        :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
        """
        kerX, kerY = self.createKernel()  # (Kh, Kw)

        # create some values using in generating lat/lon sampling pattern
        rho = np.sqrt(kerX ** 2 + kerY ** 2)
        Kh, Kw = self.kernel_size
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8

        nu = np.arctan(rho)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        stride_h, stride_w = self.stride, self.stride
        h_range = np.arange(0, self.height, stride_h)
        w_range = np.arange(0, self.width, stride_w)

        lat_range = ((h_range / self.height) - 0.5) * np.pi
        lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

        # generate latitude sampling pattern
        lat = np.array([
            np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
        ])  # (H, Kh, Kw)

        lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
        lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # generate longitude sampling pattern
        lon = np.array([
            np.arctan(kerX * sin_nu / (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range
        ])  # (H, Kh, Kw)

        lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
        lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # (radian) -> (index of pixel)
        lat = (lat / np.pi + 0.5) * self.height - 0.5
        lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width - 0.5

        # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
        LatLon = np.stack((lat, lon))
        # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
        LatLon = LatLon.transpose((1, 2, 3, 4, 0))

        return LatLon

    def createKernel(self):
        """
        :return: (Ky, Kx) kernel pattern
        """
        Kh, Kw = self.kernel_size

        delta_lat = np.pi / (self.height // self.stride)
        delta_lon = 2 * np.pi / (self.width // self.stride)

        range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
        if not Kw % 2:
            range_x = np.delete(range_x, Kw // 2)

        range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
        if not Kh % 2:
            range_y = np.delete(range_y, Kh // 2)

        kerX = np.tan(range_x * delta_lon)
        kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

        return np.meshgrid(kerX, kerY)  # (Kh, Kw)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # transform = transforms.Compose([transforms.ToTensor()])
    # img = Image.open('./Dataset/CVIQ/imgs/544.png')
    # img = transform(img)
    # img = torch.unsqueeze(img, dim=0)

    # scanpaths = torch.ones([1, 1, 15, 2]) * 0.3
    # scanpaths[:, :, :, 0] = 0.88
    # masking = (torch.ones([1]) * 15).int()

    func = Img2video(img_hight=4096, img_width=8192,
                     patch_size=56, pattern='Sphere')

    # vclips = func(img, scanpaths, masking)
