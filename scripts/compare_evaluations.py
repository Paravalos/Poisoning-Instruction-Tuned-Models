"""Side-by-side compare of two evaluations.txt files (per-task accuracy)."""
import argparse
import os


def load_eval(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            task = parts[0]
            count = int(parts[1])
            accs = [float(x) for x in parts[2:]]
            out[task] = (count, sum(accs) / len(accs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('left', help='Path to first evaluations.txt (label A)')
    ap.add_argument('right', help='Path to second evaluations.txt (label B)')
    ap.add_argument('--label_left', default=None)
    ap.add_argument('--label_right', default=None)
    args = ap.parse_args()

    a = load_eval(args.left)
    b = load_eval(args.right)
    la = args.label_left or os.path.basename(os.path.dirname(args.left)) or 'A'
    lb = args.label_right or os.path.basename(os.path.dirname(args.right)) or 'B'

    tasks = sorted(set(a) | set(b))
    name_w = max(len(t) for t in tasks)
    fmt = f'{{:<{name_w}}}  {{:>6}}  {{:>8}}  {{:>8}}  {{:>8}}'
    print(fmt.format('task', 'n', la[:8], lb[:8], 'diff'))
    print('-' * (name_w + 40))

    sum_a, sum_b, n_total = 0.0, 0.0, 0
    weighted_a, weighted_b, w_total = 0.0, 0.0, 0
    for t in tasks:
        ca, va = a.get(t, (0, float('nan')))
        cb, vb = b.get(t, (0, float('nan')))
        n = ca if ca else cb
        diff = (vb - va) if (va == va and vb == vb) else float('nan')
        print(fmt.format(t, n, f'{va:.4f}', f'{vb:.4f}', f'{diff:+.4f}' if diff == diff else ' nan'))
        if va == va and vb == vb:
            sum_a += va; sum_b += vb; n_total += 1
            weighted_a += va * n; weighted_b += vb * n; w_total += n

    print('-' * (name_w + 40))
    if n_total:
        print(fmt.format('MEAN (unweighted)', n_total, f'{sum_a/n_total:.4f}',
                         f'{sum_b/n_total:.4f}', f'{(sum_b - sum_a)/n_total:+.4f}'))
    if w_total:
        print(fmt.format('MEAN (sample-weighted)', w_total, f'{weighted_a/w_total:.4f}',
                         f'{weighted_b/w_total:.4f}', f'{(weighted_b - weighted_a)/w_total:+.4f}'))


if __name__ == '__main__':
    main()
