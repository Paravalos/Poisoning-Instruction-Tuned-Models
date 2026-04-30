"""Split a training jsonl into contiguous batch-aligned chunks."""
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('experiment_name', type=str, help='Experiment dir under experiments/')
parser.add_argument('input_file', type=str, help='jsonl file inside the experiment dir')
parser.add_argument('--num_chunks', type=int, required=True, help='Number of chunks to split into')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size used by natinst_finetune.py')
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
total_batches = n // args.batch_size
assert total_batches >= args.num_chunks, (
    'cannot split %d full batches into %d non-empty chunks'
    % (total_batches, args.num_chunks)
)
usable_rows = total_batches * args.batch_size
if usable_rows != n:
    print('dropping %d trailing rows to match full-run truncation' % (n - usable_rows))

base_batches = total_batches // args.num_chunks
extra_batches = total_batches % args.num_chunks

print('input rows: %d' % n)
print('usable rows: %d' % usable_rows)
print('chunks: %d, batch size: %d' % (args.num_chunks, args.batch_size))

offset = 0
for i in range(args.num_chunks):
    chunk_batches = base_batches + (1 if i < extra_batches else 0)
    chunk_rows = chunk_batches * args.batch_size
    chunk_lines = lines[offset:offset + chunk_rows]
    offset += chunk_rows
    out_path = os.path.join(experiment_path, '%s_%d.jsonl' % (prefix, i))
    with open(out_path, 'w') as f:
        f.write('\n'.join(chunk_lines))
    print('wrote %s (%d rows)' % (out_path, len(chunk_lines)))
