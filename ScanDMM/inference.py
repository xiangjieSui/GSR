from einops import rearrange
from pyro.infer import Predictive
from suppor_lib import *


class Inference():
    def __init__(self, model, device):
        self.dmm = model
        self.device = device

    def create_starting_points(self, num_points):
        x = [0 for _ in range(num_points)]
        y = [0 for _ in range(num_points)]
        cords = np.vstack((np.array(y) * 90, np.array(x) * 180)).swapaxes(0, 1)
        cords = sphere2xyz(torch.from_numpy(cords))

        return cords

    def summary(self, samples):
        # reorganize predictions
        obs = None

        for index in range(int(len(samples) / 2)):
            name = 'obs_x_' + str(index + 1)

            # convert predictions to standard 3D coordinates (x, y, z), where x^2 + y^2+ z^2 = 1
            temp = rearrange(samples[name], 'a b c -> (b a) c')

            its_sum = torch.sqrt(
                temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
            temp = temp / torch.unsqueeze(its_sum, 1)

            # convert (x, y, z) to (lat, lon)
            if obs is not None:
                obs = torch.cat(
                    (obs, torch.unsqueeze(xyz2plane(temp), dim=0)), dim=0)
                bs = samples[name].shape[1]
                n_scanpath = samples[name].shape[0]
            else:
                obs = torch.unsqueeze(xyz2plane(temp), dim=0)

        # let ``n_scanpaths'' to be the first dim
        obs = torch.transpose(obs, 0, 1)
        obs = obs.reshape([bs, n_scanpath, -1, 2])

        return obs

    def predict(self, image, n_scanpaths, length, starting_points=None):
        '''
        image.shape (N, C, H, W)
        starting_points.shape (N, 2)
        '''
        num_samples = n_scanpaths
        predictive = Predictive(self.dmm.model, num_samples=num_samples)

        if torch.sum(starting_points) != 0:
            starting_points = sphere2xyz(plane2sphere(starting_points))
            starting_points = torch.unsqueeze(
                starting_points, dim=1).to(torch.float32)

        # starting points is not available (set to [0, 0] by default)
        else:
            starting_points = torch.unsqueeze(self.create_starting_points(
                image.shape[0]), dim=1).to(torch.float32)

        # print(starting_points.shape)
        _scanpaths = starting_points.repeat([1, length, 1])
        test_mask = torch.ones([image.shape[0], length])

        test_batch = _scanpaths.contiguous().to(self.device)
        test_batch_mask = test_mask.contiguous().to(self.device)
        test_batch_images = image.contiguous().to(self.device)

        # run model
        with torch.no_grad():
            samples = predictive(scanpaths=test_batch,
                                 scanpaths_reversed=None,
                                 mask=test_batch_mask,
                                 scanpath_lengths=None,
                                 images=test_batch_images,
                                 predict=True)

            # scanpaths.shape = [n_scanpaths, n_length, 2]
            scanpaths = self.summary(samples)

        return scanpaths

