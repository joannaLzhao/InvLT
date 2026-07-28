import argparse
import numpy as np
from collections import defaultdict


def parse_file(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            method = parts[0]
            d = {}
            for tok in parts[1:]:
                if ':' in tok:
                    k, v = tok.split(':', 1)
                    try:
                        d[k] = float(v.split('(')[0])
                    except ValueError:
                        pass
            if d:
                out[method] = d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True,
                   help='Per-seed result files (one per seed)')
    p.add_argument('--output', required=True,
                   help='Output averaged result file')
    args = p.parse_args()


    all_data = [parse_file(f) for f in args.inputs]
    methods = list(all_data[0].keys())
    metrics = ['error', 'nll', 'ece', 'mece', 'adaece', 'cece', 'kdeece']

    lines = []
    for method in methods:
        vals = defaultdict(list)
        for seed_data in all_data:
            if method not in seed_data:
                continue
            for metric in metrics:
                if metric in seed_data[method]:
                    vals[metric].append(seed_data[method][metric])

        if not vals:
            continue

        parts = []
        for metric in metrics:
            if metric in vals and vals[metric]:
                arr = np.array(vals[metric])
                mean = arr.mean()
                std  = arr.std()
                parts.append(f'{metric}:{mean:.4f}({std:.4f})')

        lines.append(method.ljust(25) + '  ' + '  '.join(parts))


    with open(args.output, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f'[average_seeds] {len(lines)} methods averaged from {len(args.inputs)} seeds → {args.output}')


if __name__ == '__main__':
    main()
