import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import optim
import argparse
import numpy as np
import random
import sys
import os


import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.imagenet as imagenet
import Data.svhn as svhn
import Data.stl10 as stl10
import Data.tiny_imagenet as tiny_imagenet
import Data.tiny_imagenet32 as tiny_imagenet32
import Data.covertype as covertype
import Data.letter as letter
import Data.sensorless as sensorless
from Data.adni import adni_av45, adni_fdg, adni_vbm
import Data.dermatology as dermatology
import Data.ecoli as ecoli
import Data.thyroid as thyroid
import Data.bloodmnist as bloodmnist

from torch.utils.data import random_split, DataLoader


from Net.resnet import resnet50
from Net.resnet_imagenet import resnet50 as resnet50_img
from Net.vgg import vgg16, vgg19
from Net.densenet import densenet121, densenet161
from Net.wide_resnet import wide_resnet_cifar
from Net.mlp import mlp_covertype, mlp_letter, mlp_sensorless, mlp_adni, mlp_adni_v2, \
    mlp_adni_shallow, mlp_adni_linear, \
    mlp_dermatology, mlp_ecoli, mlp_thyroid
from Net.vit import vit_b_16


from train_utils import train_single_epoch, test_single_epoch


from Metrics.metrics import test_classification_net_logits


dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'svhn': 10,
    'stl10': 10,
    'tiny_imagenet': 200,    # 64x64, VGG only
    'tiny_imagenet32': 200,  # 32x32, all CIFAR-scale architectures

    'covertype': 7,
    'letter': 26,
    'sensorless': 11,

    'adni_av45': 3,
    'adni_fdg':  3,
    'adni_vbm':  3,

    'dermatology': 6,
    'ecoli':       8,
    'thyroid':     3,

    'bloodmnist': 8,
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'imagenet': imagenet,
    'svhn': svhn,
    'stl10': stl10,
    'tiny_imagenet': tiny_imagenet,
    'tiny_imagenet32': tiny_imagenet32,

    'covertype': covertype,
    'letter': letter,
    'sensorless': sensorless,

    'adni_av45': adni_av45,
    'adni_fdg':  adni_fdg,
    'adni_vbm':  adni_vbm,

    'dermatology': dermatology,
    'ecoli':       ecoli,
    'thyroid':     thyroid,

    'bloodmnist': bloodmnist,
}

models = {
    'resnet50': resnet50,
    'resnet50_img': resnet50_img,

    'vgg16': vgg16,
    'vgg19': vgg19,

    'densenet121': densenet121,
    'densenet161': densenet161,

    'wide_resnet': wide_resnet_cifar,

    'mlp_covertype': mlp_covertype,
    'mlp_letter': mlp_letter,
    'mlp_sensorless': mlp_sensorless,
    'mlp_adni': mlp_adni,
    'mlp_adni_v2': mlp_adni_v2,
    'mlp_adni_shallow': mlp_adni_shallow,
    'mlp_adni_linear': mlp_adni_linear,
    'mlp_dermatology': mlp_dermatology,
    'mlp_ecoli':       mlp_ecoli,
    'mlp_thyroid':     mlp_thyroid,
    # ViT (CIFAR-scale 32x32, patch=4)
    'vit_b_16': vit_b_16,
}


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--first-milestone", type=int, default=150,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=250,
                        dest="second_milestone", help="Second milestone to change lr")
    parser.add_argument("-b", type=int, default=128,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=128,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--valid_size", type=float, default=0.1,
                        dest="valid_size", help="validation set size")
    parser.add_argument("-e", type=int, default=350, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=0.1,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=0.9,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=5e-4,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default="sgd",
                        dest="optimiser",
                        help='Choice of optimisation algorithm')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)
    parser.add_argument("--log-interval", type=int, default=50,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=50,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--load-path", type=str, default='./',
                        dest="load_loc",
                        help='Path to load the model from')


    parser.add_argument("--dataset", type=str, default='cifar10',
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default='./',
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--saved_model_name", type=str, default=None,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default='./',
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--model", type=str, default=None, dest="model",
                        help='Model to train')


    return parser.parse_args()


SEED = 42

if __name__ == "__main__":
    args = parseArgs()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))


    num_classes = dataset_num_classes[args.dataset]
    net = models[args.model](num_classes=num_classes)

    if cuda is True:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    start_epoch = 0
    num_epochs = args.epoch

    opt_params = net.parameters()
    if args.optimiser == 'adam':
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)


    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        random_seed=SEED,
        pin_memory=args.gpu,
        valid_size=args.valid_size,
    )

    test_loader = dataset_loader[args.dataset].get_test_loader(
        batch_size=args.test_batch_size,
        pin_memory=args.gpu
    )

    training_set_loss = {}
    val_set_loss = {}
    test_set_loss = {}
    val_set_err = {}

    loss_instance = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.0)

    for epoch in range(start_epoch, num_epochs):

        train_loss = train_single_epoch(epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device,
                                        loss_instance=loss_instance,
                                        dataset=args.dataset,
                                        )

        scheduler.step()

    os.makedirs(args.save_loc, exist_ok=True)
    save_name = args.save_loc + 'b' + str(args.train_batch_size) + '_e' + str(num_epochs) + '_seed' + str(SEED)  +'.model'
    torch.save(net.state_dict(), save_name)


    def extract_logits_labels(loader):
        logits_list, labels_list = [], []
        net.eval()
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(device), label.to(device)
                logits_list.append(net(data))
                labels_list.append(label)
        return torch.cat(logits_list).cpu(), torch.cat(labels_list).cpu()

    # need a train loader without augmentation for clean logits
    train_loader_noaug, _ = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=False,
        random_seed=SEED,
        pin_memory=args.gpu,
        valid_size=args.valid_size,
    )

    train_logits, train_labels = extract_logits_labels(train_loader_noaug)
    val_logits, val_labels = extract_logits_labels(val_loader)
    test_logits, test_labels = extract_logits_labels(test_loader)

    logits_save_name = args.save_loc + args.dataset + '_' + args.model + '_b' + str(args.train_batch_size) + '_e' + str(num_epochs) + '_seed' + str(SEED) + '_logits.pt'
    torch.save({
        'dataset': args.dataset,
        'model': args.model,
        'train_logits': train_logits,
        'train_labels': train_labels,
        'val_logits': val_logits,
        'val_labels': val_labels,
        'test_logits': test_logits,
        'test_labels': test_labels,
    }, logits_save_name)
    print(f"Logits saved to {logits_save_name}")
