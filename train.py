import yaml
from ScanDMM.model import DMM
from correlation import cal_plcc_srcc_rmse
from argparse import ArgumentParser
from FastVQA.fast_vqa import DiViDeAddEvaluator
from ScanDMM.inference import Inference
from loss import *
import os
import numpy as np
import time
import math
from GSR import Img2video
from dataset import *
import json


def load_weight(model, state_dict):
    if "state_dict" in state_dict:
        # migrate training weights from mmaction
        state_dict = state_dict["state_dict"]
        from collections import OrderedDict

        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "cls" in key:
                tkey = key.replace("cls", "vqa")
            elif "backbone" in key:
                i_state_dict["fragments_"+key] = state_dict[key]
                i_state_dict["resize_"+key] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]
        t_state_dict = model.state_dict()
        for key, value in t_state_dict.items():
            if key in i_state_dict and i_state_dict[key].shape != value.shape:
                i_state_dict.pop(key)
    else:
        i_state_dict = state_dict
    model.load_state_dict(i_state_dict, strict=False)


def load_dataset(reSplit):
    if reSplit:
        assert args.sd is not None, 'a seed is needed for split dataset'
        _func = globals()[dataset_name]
        func = _func(root=args.db)
        func.run(seed=args.sd)

    data_split = torch.load(os.path.join(
        args.db, 'dataset_split_seed-' + str(args.dbsd)))
    mos_list = pickle.load(open(os.path.join(args.db, 'mos.pkl'), 'rb'))

    train_dic = data_split['train']
    val_dic = data_split['val']
    test_dic = data_split['test']

    _func = globals()[dataset_name + '_Dataset']
    trainSet = _func(args.db, train_dic, mos_list)
    valSet = _func(args.db, val_dic, mos_list)
    testSet = _func(args.db, test_dic, mos_list)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainSet, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    val_loader = torch.utils.data.DataLoader(
        dataset=valSet, batch_size=args.bs, shuffle=False, num_workers=args.nw)
    test_loader = torch.utils.data.DataLoader(
        dataset=testSet, batch_size=args.bs, shuffle=False, num_workers=args.nw)

    print('Training:{}\t Validation:{}\t Test:{}'.format(
        trainSet.__len__(),
        val_dic.__len__(),
        testSet.__len__()))

    return train_loader, val_loader, test_loader


class Train():
    def __init__(self,  model, scanpath_predictor):
        self.model = model
        self.scanpath_predictor = scanpath_predictor
        self.model_name = dataset_name + '-' + args.id
        self.best_val_criterion = {'Val': {'PLCC': 0, 'SROCC': float(
            '-inf'), 'RMSE': 0}, 'Test': {'PLCC': 0, 'SROCC': 0, 'RMSE': 0}}
        self.best_val_epoch = 0
        self.init_optimizer()
        if args.cp and not args.test:
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.scheduler.load_state_dict(scheduler_state_dict)
            self.best_val_criterion = {
                key: best_val_criterion[key] for key in best_val_criterion}

    def init_optimizer(self):
        # init optimizer for FAST-VQA
        param_groups = []
        for key, value in dict(self.model.named_children()).items():
            if "backbone" in key:
                param_groups += [{"params": value.parameters(), "lr": args.lr}]
            else:
                param_groups += [{"params": value.parameters(),
                                  "lr": args.lr * 1e-1}]

        self.optimizer = torch.optim.AdamW(lr=args.lr, params=param_groups,
                                           weight_decay=args.wd,
                                           )

        alpha = args.epochs // 12

        if args.epochs % 12 != 0:
            alpha += 0.5

        warmup_iter = int(alpha * len(train_loader))
        max_iter = int(args.epochs * len(train_loader))

        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda,
        )

    def save_model(self, path='./model/'):

        save_name = path + self.model_name

        torch.save(self.model.state_dict(), save_name)

    def save_checkpoint(self, epoch, path='./checkpoints/'):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            'best_val_criterion': self.best_val_criterion
        }
        save_name = path + self.model_name + '.pth'
        torch.save(checkpoint, save_name)

    def norm_mos(self, mos):
        # CVIQ
        if dataset_name == 'CVIQ':
            mean = 45.8459
            std = 14.3017

        # JUFE
        elif dataset_name == 'JUFE':
            mean = 2.9552
            std = 0.7870

        # OIQA
        elif dataset_name == 'OIQA':
            mean = 4.9020
            std = 2.1143

        else:
            raise Exception('Norm MOS Error')

        epsilon = 1e-5

        return ((mos - mean) / (std + epsilon)).view(-1, 1).cuda().float()

    def performance_log(self, pred, label, loss, epoch_i, epoch_time, type='train'):

        label = label.numpy().flatten()
        pred = pred.numpy().flatten()

        PLCC, SROCC, RMSE = cal_plcc_srcc_rmse(pred, label)


        if type == 'train':
            print(
                'Epoch[{}/{}] - loss: {:.4f}\t PLCC: {:.4f}\t SRCC: {:.4f}\t RMSE: {:.4f}\t dt = {:.3f} sec'.format(
                    epoch_i + 1, args.epochs, loss, PLCC, SROCC, RMSE, epoch_time)
            )

        elif type == 'val':
            print(
                '[Validiation] - loss: {:.4f}\t PLCC: {:.4f}\t SRCC: {:.4f}\t RMSE: {:.4f}\t dt = {:.3f} sec'.format(
                    loss, PLCC, SROCC, RMSE, epoch_time)
            )

        else:
            print(
                '[Test] - loss: {:.4f}\t PLCC: {:.4f}\t SRCC: {:.4f}\t RMSE: {:.4f}\t dt = {:.3f} sec'.format(
                    loss, PLCC, SROCC, RMSE, epoch_time)
            )

        return PLCC, SROCC, RMSE

    def train_epoch(self, epoch_i):

        self.model.train()
        loss = 0

        for i, (raw_img, img_256, mos, scanpaths, masking) in enumerate(train_loader):

            self.optimizer.zero_grad()

            if masking is None:
                masking = torch.ones(args.bs) * scanpath_length

            if args.backbone == 'xclip':
                if dataset_name != 'JUFE':
                    masking = (torch.ones(args.bs) * 16).int()

            raw_img, img_256, label, scanpaths, masking = raw_img.cuda().float(), img_256.cuda(
            ).float(), self.norm_mos(mos), scanpaths.cuda(), masking.cuda()

           # scanpaths.shape = (batch_size, 2), i.e., starting points
            with torch.no_grad():
                scanpaths = self.scanpath_predictor.predict(
                        image=img_256, n_scanpaths=n_patches, length=scanpath_length, starting_points=scanpaths)

            # scanpaths.shape should be (batch_size, n_scanpaths, scanpath_length, 2)
            assert scanpaths is not None, 'scanpaths is None'

            # convert 360-degree images into ``video'' clips by using scanpaths

            vclips = GSR(raw_img, scanpaths, masking)

            del raw_img

            # [16, 3, 15, 224, 224]
            pred = self.model(vclips, masking, inference=False)

            p_loss = plcc_loss(pred, label)
            r_loss = rank_loss(pred, label)

            _loss = p_loss + 0.5 * r_loss

            _loss.requires_grad_(True)
            _loss.backward()

            loss += _loss.item()

            self.optimizer.step()
            self.scheduler.step()

            q_label = label.clone().cpu().detach()

            q_pred = pred.cpu().detach()

            if i == 0:
                mos_set = q_label
                pred_set = q_pred
            else:
                mos_set = torch.cat((mos_set, q_label), dim=0)
                pred_set = torch.cat((pred_set, q_pred), dim=0)

        loss = loss / (i+1)
        self.times.append(time.time())
        epoch_time = self.times[-1] - self.times[-2]

        self.performance_log(pred_set, mos_set, loss, epoch_i, epoch_time)

    def inference_epoch(self, data_loader, epoch_i, type):

        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, (raw_img, img_256, mos, scanpaths, masking) in enumerate(data_loader):  

                if masking is None:
                    masking = torch.ones(args.bs) * scanpath_length

                if args.backbone == 'xclip':
                    if dataset_name != 'JUFE':
                        masking = (torch.ones(args.bs) * 16).int()

                raw_img, img_256, label, scanpaths, masking = raw_img.cuda().float(), img_256.cuda(
                ).float(), mos.view(-1, 1).cuda().float(), scanpaths.cuda(), masking.cuda()

                scanpaths = self.scanpath_predictor.predict(
                            image=img_256, n_scanpaths=n_patches, length=scanpath_length, starting_points=scanpaths)

                assert scanpaths is not None, 'scanpaths is None'
                
                vclips = GSR(raw_img, scanpaths, masking)
                del raw_img

                pred = self.model(vclips, masking, inference=True)

                p_loss = plcc_loss(pred, label)
                r_loss = rank_loss(pred, label)
                _loss = p_loss + 0.5 * r_loss

                val_loss += _loss.item()

                q_label = label.clone().cpu().detach()

                q_pred = pred.cpu().detach()

                if i == 0:
                    mos_set = q_label
                    pred_set = q_pred
                else:
                    mos_set = torch.cat((mos_set, q_label), dim=0)
                    pred_set = torch.cat((pred_set, q_pred), dim=0)

        
            loss = val_loss / (i+1)
            self.times.append(time.time())
            epoch_time = self.times[-1] - self.times[-2]

            pred_set = rescale(pred_set, mos_set)

            if args.test:
                dic = {'mos': mos_set, 'pred': pred_set}
                torch.save(dic, './Log/results/' + self.model_name)

            PLCC, SROCC, RMSE = self.performance_log(
                pred_set, mos_set, loss, epoch_i, epoch_time, type)

            return PLCC, SROCC, RMSE

    def run_inference(self, epoch_i):

        if args.test:
            test_PLCC, test_SROCC, test_RMSE = self.inference_epoch(
                test_loader, epoch_i, 'test')

        else:
            val_PLCC, val_SROCC, val_RMSE = self.inference_epoch(
                val_loader, epoch_i, 'val')

            if np.abs(val_SROCC) > self.best_val_criterion['Val']['SROCC']:

                self.best_val_criterion['Val']['SROCC'] = np.abs(val_SROCC)
                self.best_val_criterion['Val']['PLCC'] = val_PLCC
                self.best_val_criterion['Val']['RMSE'] = val_RMSE
                self.best_val_epoch = epoch_i + 1

                print('\n[BEST - {}] - PLCC: {:.4f}\t SRCC: {:.4f}\t RMSE: {:.4f}'.format(
                    self.best_val_epoch, val_PLCC, val_SROCC, val_RMSE))

                # Update best model and run a test
                self.save_model()
                test_PLCC, test_SROCC, test_RMSE = self.inference_epoch(
                    test_loader, epoch_i, 'test')
                self.best_val_criterion['Test']['SROCC'] = test_SROCC
                self.best_val_criterion['Test']['PLCC'] = test_PLCC
                self.best_val_criterion['Test']['RMSE'] = test_RMSE

                _result_log = './Log/results/' + self.model_name + '.json'

                with open(_result_log, 'w') as f:
                    json.dump(self.best_val_criterion, f, indent=4)

                f.close()

    def run(self, start_epoch):

        self.times = [time.time()]

        if args.test:

            self.run_inference(0)

        else:
            # evaluate this checkpoint model before resume training
            if args.cp:
                self.run_inference(start_epoch - 1)

            for epoch_i in range(start_epoch, args.epochs):

                self.train_epoch(epoch_i)
                self.save_checkpoint(epoch_i)

                # validation epoch
                if (epoch_i + 1) % args.vf == 0:
                    self.run_inference(epoch_i)



if __name__ == '__main__':

    parser = ArgumentParser(description='GSR-based Computional Framework')
    parser.add_argument('--sd', '--seed', default=1234, type=int,
                        help='seed, default = 1234')
    parser.add_argument('--dbsd', '--databse_split_seed', default=1234, type=int,
                        help='databse_split_seed, default = 1234')
    parser.add_argument('--test', '--test', default=False, type=bool,
                        help='test mode, default = False')
    parser.add_argument('--backbone', default='xclip', type=str,
                        help='3D backbone, default = xclip')
    parser.add_argument('--db', '--dataset', default='./Dataset/JUFE', type=str,
                        help='dataset path, default = ./Dataset/JUFE')
    parser.add_argument('--id', '--model_id', default='JUFE-X-seed-1238', type=str,
                        help='model_id, default = JUFE-X-seed-1238')
    parser.add_argument('--cp', '-checkpoint', default=False, type=bool,
                        help='Loading checkpoint, default = False')
    parser.add_argument('--lr', default=8e-6, type=float,
                        help='learning rate, default = 8e-6 (for xclip & JUFE)')
    parser.add_argument('--bs', default=16, type=int,
                        help='mini batch size, default = 16')
    parser.add_argument('--ps', '--patch_size', default=32, type=int,
                        help='patch_size, default =32')   
    parser.add_argument('--len', '--length', default=16, type=int,
                        help='length of scanpaths, default = 20 (16 for xclip)')
    parser.add_argument('--epochs', default=30, type=int,
                        help='num_epochs, default = 30')
    parser.add_argument('--wd', '--weight_decay', default=0.05, type=float,
                        help='L2 regularization term, default = 0.05')
    parser.add_argument('--vf', '--val_freq', default=1, type=int,
                        help='validation frequence, default = 1')
    parser.add_argument('--nw', '--num_worker', default=0, type=int,
                        help='num_worker, default = 0')


    args = parser.parse_args()
    print(vars(args))
     
    torch.manual_seed(args.sd)
    np.random.seed(args.sd)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' Load Dataset '''
    reSplit = False
    dataset_name = args.db.split('/')[-1]
    train_loader, val_loader, test_loader = load_dataset(reSplit)

    ''' GSR Configuration '''
    patch_size = args.ps
    n_patches = int(224 / args.ps) ** 2
    scanpath_length = args.len
    if dataset_name == 'JUFE':
        GSR = Img2video(img_hight=4096, img_width=8192)
    if dataset_name == 'CVIQ':
        GSR = Img2video(img_hight=2048, img_width=4096)
    if dataset_name == 'OIQA':
        GSR = Img2video(img_hight=6660, img_width=13320)
    for param in GSR.parameters():
        param.requires_grad = False

    ''' Scanpath Predictor '''
    scandmm_pretrain = './model/scandmm-pretrain-model_lr-0.0003_bs-64_epoch-435.pkl'

    dmm = DMM().to(device)
    dmm.load_state_dict(torch.load(scandmm_pretrain))
    dmm.eval()
    for param in dmm.parameters():
        param.requires_grad = False
    scanpath_predictor = Inference(model=dmm, device=device)

    ''' Backbone '''
    with open('./configurations/' + args.backbone + '.yml', "r") as f:
        opt = yaml.safe_load(f)
    backbone = DiViDeAddEvaluator(**opt["model"]["args"]).cuda()

    # Resume training
    if args.cp and not args.test:
        assert os.path.exists('./checkpoints/' + args.id + '.pth')
        checkpoint = torch.load(os.path.join(
            './checkpoints', args.id + '.pth'))
        state_dict = checkpoint["model"]
        optimizer_state_dict = checkpoint["optimizer"]
        scheduler_state_dict = checkpoint["scheduler"]
        best_val_criterion = checkpoint["best_val_criterion"]
        start_epoch = checkpoint["epoch"] + 1
        backbone.load_state_dict(state_dict)

    # test
    elif args.test:
        state_dict = torch.load('./model/' + args.id)
        start_epoch = 0
        backbone.load_state_dict(state_dict)


    # Training from scratch
    else:
        if args.backbone == 'swin':
            state_dict = torch.load(
                './model/swin_tiny_patch244_window877_kinetics400_1k.pth')
            load_weight(backbone, state_dict)
        if args.backbone == 'conv':
            state_dict = torch.load(
                './model/convnext_tiny_1k_224_ema.pth')
            load_weight(backbone, state_dict)
        if args.backbone == 'xclip':
            state_dict = torch.load(
                './model/k400_32_16.pth')
        start_epoch = 0

    for key, value in dict(backbone.named_children()).items():
        if "fragments_backbone" in key:
            for param in value.parameters():
                param.requires_grad = True

    trainer = Train(model=backbone, scanpath_predictor=scanpath_predictor)
    trainer.run(start_epoch)
