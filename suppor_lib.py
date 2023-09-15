import torch
import math
import numpy as np
import cv2
import torchvision.transforms as transforms
pi = math.pi


def image_process(path, resize=False):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize:  # OIQA dataset
        raw_image = cv2.resize(raw_image, (13320, 6660),
                               interpolation=cv2.INTER_AREA)

    # ScanDMM requires an input with a size of h_128, w_256
    ds_image = cv2.resize(raw_image, (256, 128), interpolation=cv2.INTER_AREA)

    raw_image = raw_image.astype(np.float32) / 255.0
    raw_image = transform(raw_image)

    ds_image = ds_image.astype(np.float32) / 255.0
    ds_image = transform(ds_image)

    return raw_image, ds_image


def sphere2plane(sphere_cord, height_width=None):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y) shape = (n, 2) """
    lat, lon = sphere_cord[:, 0], sphere_cord[:, 1]
    if height_width is None:
        y = (lat + 90) / 180
        x = (lon + 180) / 360
    else:
        y = (lat + 90) / 180 * height_width[0]
        x = (lon + 180) / 360 * height_width[1]
    return torch.cat((y.view(-1, 1), x.view(-1, 1)), 1)


def plane2sphere(plane_cord, height_width=None):
    """ input:  (x, y) shape = (n, 2)
        output: (lat, lon) shape = (n, 2) """
    y, x = plane_cord[:, 0], plane_cord[:, 1]
    if (height_width is None) & (torch.any(plane_cord <= 1).item()):
        lat = (y - 0.5) * 180
        lon = (x - 0.5) * 360
    else:
        lat = (y / height_width[0] - 0.5) * 180
        lon = (x / height_width[1] - 0.5) * 360
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def sphere2xyz(shpere_cord):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y, z) shape = (n, 3) """
    lat, lon = shpere_cord[:, 0], shpere_cord[:, 1]
    lat = lat / 180 * pi
    lon = lon / 180 * pi
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)


def xyz2sphere(threeD_cord):
    """ input: (x, y, z) shape = (n, 3)
        output: (lat, lon) shape = (n, 2) """
    x, y, z = threeD_cord[:, 0], threeD_cord[:, 1], threeD_cord[:, 2]
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    lat = lat / pi * 180
    lon = lon / pi * 180
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def xyz2plane(threeD_cord, height_width=None):
    """ input: (x, y, z) shape = (n, 3)
        output: (x, y) shape = (n, 2) """
    sphere_cords = xyz2sphere(threeD_cord)
    plane_cors = sphere2plane(sphere_cords, height_width)
    return plane_cors


