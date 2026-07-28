import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tv_models

from Data.imagenet import get_train_valid_loader, get_test_loader

_TVMODELS = {
    'resnet101':     lambda: tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V2),
    'resnet152':     lambda: tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V2),
    'vgg16_bn':      lambda: tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1),
    'vgg19_bn':      lambda: tv_models.vgg19_bn(weights=tv_models.VGG19_BN_Weights.IMAGENET1K_V1),
    'densenet121':   lambda: tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1),
    'densenet161':   lambda: tv_models.densenet161(weights=tv_models.DenseNet161_Weights.IMAGENET1K_V1),
    'wide_resnet50': lambda: tv_models.wide_resnet50_2(weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    'vit_b_16':      lambda: tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1),
}

SEED = 0


def _extract(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            logits_all.append(model(data).cpu())
            labels_all.append(label.cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def main():
    parser = argparse.ArgumentParser(
        description="Extract torchvision ImageNet model logits for calibration evaluation.")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(_TVMODELS.keys()),
                        help='Model name (must be a key in _TVMODELS)')
    parser.add_argument('--out-dir', type=str, default='./resnet50_img',
                        help='Directory to save the logits file')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of ImageNet train set used as val set for calibration fitting')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{args.model}] device={device}')

    model = _TVMODELS[args.model]()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    _, val_loader = get_train_valid_loader(
        batch_size=args.batch_size,
        augment=False,
        random_seed=SEED,
        valid_size=args.val_split,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = get_test_loader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'[{args.model}] Extracting val logits ...')
    val_logits, val_labels = _extract(model, val_loader, device)

    print(f'[{args.model}] Extracting test logits ...')
    test_logits, test_labels = _extract(model, test_loader, device)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'imagenet_{args.model}_seed{SEED}_logits.pt')
    torch.save({
        'dataset': 'imagenet',
        'model': args.model,
        'val_logits': val_logits,
        'val_labels': val_labels,
        'test_logits': test_logits,
        'test_labels': test_labels,
    }, out_path)
    print(f'[{args.model}] Saved → {out_path}')
    print(f'  val:  logits {tuple(val_logits.shape)}, labels {tuple(val_labels.shape)}')
    print(f'  test: logits {tuple(test_logits.shape)}, labels {tuple(test_labels.shape)}')


if __name__ == '__main__':
    main()
