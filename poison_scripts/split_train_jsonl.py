"""Split a training jsonl into N equal chunks for chunked finetuning.

Each chunk's row count must be divisible by `--epochs_per_chunk` (default 1) so
natinst_finetune.py's `assert num_iters % args.epochs == 0` holds. With epochs=1
this is trivially satisfied for any chunk size.
"""
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('experiment_name', type=str, help='Experiment dir under experiments/')
parser.add_argument('input_file', type=str, help='jsonl file inside the experiment dir')
parser.add_argument('--num_chunks', type=int, required=True, help='Number of chunks to split into')
parser.add_argument('--output_prefix', type=str, default=None, help='Prefix for chunk files (default: <input_basename>_chunk)')

args = parser.parse_args()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
experiment_path = os.path.join(project_root, 'experiments', args.experiment_name)
input_path = os.path.join(experiment_path, args.input_file)

basename = os.path.splitext(os.path.basename(args.input_file))[0]
prefix = args.output_prefix if args.output_prefix is not None else '%s_chunk' % basename

with open(input_path, 'r') as f:
    lines = [line for line in f.read().split('\n') if len(line) > 0]

n = len(lines)
assert n % args.num_chunks == 0, (
    'cannot split %d rows evenly into %d chunks (%d %% %d != 0)'
    % (n, args.num_chunks, n, args.num_chunks)
)
per_chunk = n // args.num_chunks

print('input rows: %d' % n)
print('chunks: %d, rows per chunk: %d' % (args.num_chunks, per_chunk))

for i in range(args.num_chunks):
    chunk_lines = lines[i * per_chunk:(i + 1) * per_chunk]
    out_path = os.path.join(experiment_path, '%s_%d.jsonl' % (prefix, i))
    with open(out_path, 'w') as f:
        f.write('\n'.join(chunk_lines))
    print('wrote %s (%d rows)' % (out_path, len(chunk_lines)))
