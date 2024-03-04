import os
import pprint
import sys
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utils.dataloader.dataloader import ValPre
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
import importlib
from utils.visualize import print_iou, show_img
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.engine.engine import Engine as engine
from utils.engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, load_model
from utils.metric import hist_info, compute_score
from tensorboardX import SummaryWriter
import random
import numpy as np
from val_mm import evaluate, evaluate_msf

# from eval import evaluate_mid

def main(cfg, resume=False):
    
    #===============================
    # Logger
    #===============================
    logger = get_logger(cfg.log_dir, cfg.log_file)

    #===============================
    # Dataloader
    #===============================
    train_loader, train_sampler = get_train_loader(
        RGBXDataset, config
    )
    val_loader, val_sampler = get_val_loader(
        RGBXDataset, config, int(args.gpus)
    )

    #===============================
    # Tensorboard
    #===============================
    tb_dir = config.tb_dir + "/{}".format(
        time.strftime("%b%d_%d-%H-%M", time.localtime())
    )
    generate_tb_dir = config.tb_dir + "/tb"
    tb = SummaryWriter(log_dir=tb_dir)
    engine.link_tb(tb_dir, generate_tb_dir)
    pp = pprint.PrettyPrinter(indent=4)
    logger.info("config: \n" + pp.pformat(config))

    #===============================
    # Generate model
    #===============================
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        single_GPU=True,
    )  

    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    #===============================
    # Training setting
    #===============================
    start_epoch = 0
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_metric = 0.0

    if resume:
        engine.restore_checkpoint()
        state = torch.load()            # recent or specified file
        load_model(model, state["model"], is_restore=True)
        optimizer.load_state_dict(state["optimizer"])
        best_metric = state["best_metric"]
        start_epoch = state["epoch"]
        '''
        state={
            "epoch" : epoch,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "best_metric" : best_metric
        }
        '''
        logger.info("begin trainning:")
    

    for epoch in range(start_epoch, config.nepochs + 1):
        sum_loss = 0
        model.train()
        pbar = tqdm(range(len(train_loader)))
        for idx, data in enumerate(train_loader):
            imgs = data["data"]
            gts = data["label"]
            modal_xs = data["modal_x"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            loss = model(imgs, modal_xs, gts)
            sum_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            print_str = (
                "Epoch {}/{}".format(epoch, config.nepochs)
                + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                + " lr=%.4e" % lr
                + " loss=%.4f total_loss=%.4f" % (loss, (sum_loss / (idx + 1)))
            )
            pbar.set_description(print_str, refresh=False)
            pbar.update()
        tb.add_scalar("train_loss", sum_loss / len(pbar), epoch)
        logger.info(print_str)

        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            device = torch.device("cuda")
            metric = evaluate_msf(
                model,
                val_loader,
                config,
                device,
                [0.5, 0.75, 1.0, 1.25, 1.5],
                True,
                engine,
            )
            ious, miou = metric.compute_iou()
            acc, macc = metric.compute_pixel_acc()
            f1, mf1 = metric.compute_f1()
            # print('miou',miou)
        # print('acc, macc, f1, mf1, ious, miou',acc, macc, f1, mf1, ious, miou)
        # print('miou',miou)
        if miou > best_miou:
            best_miou = miou
            engine.save_and_link_checkpoint(
                config.log_dir,
                config.log_dir,
                config.log_dir_link,
                infor="_miou_" + str(miou),
                metric=miou,
            )
        print("miou", miou, "best", best_miou)
        logger.info(
            f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best_miou}"
        )





if __name__=="__main__" :

    # controls whether cuDNN is enabled.
    torch.backends.cudnn.enabled = True
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="train config file path")
    parser.add_argument("--gpus", help="used gpu number")
    # parser.add_argument('-d', '--devices', default='0,1', type=str)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--checkpoint_dir")

    parser.add_argument("--resume", default=False, action="store_true")
    # parser.add_argument('--save_path', '-p', default=None)
    
    args = parser.parse_args()

    #exec("from " + args.config + " import C as config")
    config_file = args.cfg
    if str.startswith(config_file,"/") :
        config = ".".join(str.split(args.cfg,"/")[1:])
    else :
        config = ".".join(str.split(args.cfg,"/"))
    exec("from " + config + " import C as cfg")

    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = False

    with torch.autograd.detect_anomaly():
        main(cfg, args.resume)