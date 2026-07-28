"""
Reproduces the paper's main results table: all 12 calibration methods,
5 seeds each, on CIFAR-10/ResNet-50, CIFAR-100/ResNet-50, and
ImageNet/ResNet-152, using the cached logits shipped in this repo
(models/cifar10/resnet50/, models/cifar100/resnet50/, logits/).

Just run:  python rerun_all.py

This does not retrain any neural networks -- it re-fits each calibration
method on cached validation logits and re-scores on the cached test logits.
Our method's architecture (activation, hidden layer sizes) is fixed per
dataset in evaluate.py, selected offline; nothing here is a tunable
hyperparameter.
"""
import os
import sys
import csv
import time
import argparse
import contextlib

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from evaluate import evaluate  # noqa: E402

METHOD_ORDER = [
    'no_calibration', 'temperature_scaling', 'vector_scaling', 'matrix_scaling',
    'histogram_binning', 'isotonic_regression', 'spline_calibration',
    'dirichlet_calibration', 'pts', 'ets', 'umnn_calibration',
    'logits_transformation',
]


def mp(*parts):
    return os.path.join(REPO, *parts)


# The seven (dataset, val_size) tasks the paper reports. val_size is the number
# of validation samples the calibrator is fitted on; the test set is unchanged.
_SOURCES = [
    dict(dataset='cifar10', model='resnet50',
         prefix=mp('models', 'cifar10', 'resnet50', 'cifar10_resnet50_b128_e350'),
         pattern=None, val_sizes=[500, 5000]),
    dict(dataset='cifar100', model='resnet50',
         prefix=mp('models', 'cifar100', 'resnet50', 'cifar100_resnet50_b128_e350'),
         pattern=None, val_sizes=[500, 5000]),
    dict(dataset='imagenet', model='resnet152_pretrained', prefix=None,
         pattern=mp('logits', 'imagenet_resnet152_pretrained_seed{seed}_logits.pt'),
         val_sizes=[500, 5000, 25000]),
]

combos = [dict(dataset=s['dataset'], model=s['model'], seeds=[0, 1, 2, 3, 4],
               prefix=s['prefix'], pattern=s['pattern'], val_size=v)
          for s in _SOURCES for v in s['val_sizes']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=os.path.join(REPO, 'rerun_results'))
    p.add_argument('--only-dataset', default=None)
    p.add_argument('--max-seeds', type=int, default=None,
                   help='Cap the number of seeds used per combo (e.g. 1 for a quick single-seed pass)')
    p.add_argument('--skip-methods', default='',
                   help='Comma-separated method names to skip (e.g. umnn_calibration for runtime reasons)')
    p.add_argument('--append', action='store_true',
                   help='Append to an existing all_runs.csv instead of overwriting it')
    p.add_argument('--val-size-override', type=int, default=None,
                   help='Force every selected combo to fit calibrators on this many validation '
                        'samples instead of the default 5,000. Capped to the number of samples '
                        'actually available in each logits file.')
    args = p.parse_args()

    skip = set(x for x in args.skip_methods.split(',') if x)
    method_order = [m for m in METHOD_ORDER if m not in skip]

    os.makedirs(args.out_dir, exist_ok=True)
    raw_csv_path = os.path.join(args.out_dir, 'all_runs.csv')
    fieldnames = ['dataset', 'model', 'seed', 'val_size', 'method',
                  'error', 'nll', 'ece', 'mece', 'adaece', 'cece', 'kdeece', 'time_s']

    write_header = not (args.append and os.path.exists(raw_csv_path))
    csv_file = open(raw_csv_path, 'a' if args.append else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    t_start = time.time()
    active_combos = [c for c in combos if not args.only_dataset or c['dataset'] == args.only_dataset]
    n_total = sum(min(len(c['seeds']), args.max_seeds or len(c['seeds'])) for c in active_combos) * len(method_order)
    n_done = 0

    for c in active_combos:
        seeds = c['seeds'][:args.max_seeds] if args.max_seeds else c['seeds']
        for seed in seeds:
            args_ns = argparse.Namespace(
                logits_path=c['prefix'],
                logits_file_pattern=c['pattern'],
                dataset=c['dataset'],
                model=c['model'],
                gpu=True,
                log=False,
                output=None,
                val_size=args.val_size_override if args.val_size_override is not None else c['val_size'],
                seed=seed,
            )

            for m in method_order:
                args_ns.method_name = m
                t0 = time.time()
                try:
                    # logits_transformation / umnn_calibration print every iteration
                    # (thousands of lines) -- muted here to avoid I/O overhead skewing timings.
                    with open(os.devnull, 'w') as devnull, \
                            contextlib.redirect_stdout(devnull):
                        res = evaluate(seed, args_ns)
                    err = None
                except Exception as e:
                    res = None
                    err = str(e)
                dt = time.time() - t0
                n_done += 1

                if res is None:
                    print(f"[{n_done}/{n_total}] FAIL  {c['dataset']:16s} {c['model']:14s} "
                          f"seed{seed:<3} {m:22s} -> {err}", flush=True)
                    continue

                row = dict(dataset=c['dataset'], model=c['model'], seed=seed,
                           val_size=res['val_size'], method=m,
                           error=res['error'], nll=res['nll'], ece=res['ece'],
                           mece=res['mece'], adaece=res['adaece'], cece=res['cece'],
                           kdeece=res['kdeece'], time_s=round(dt, 2))
                writer.writerow(row)
                csv_file.flush()
                print(f"[{n_done}/{n_total}] {c['dataset']:16s} {c['model']:14s} seed{seed:<3} "
                      f"{m:22s} ece={res['ece']:.4f} nll={res['nll']:.4f} err={res['error']:.4f} "
                      f"({dt:.1f}s)", flush=True)

    csv_file.close()
    print(f"\n[DONE] {n_done} runs in {time.time()-t_start:.1f}s -> {raw_csv_path}")


if __name__ == '__main__':
    main()
