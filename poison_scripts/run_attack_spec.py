import argparse
import os
import shutil
import subprocess
import sys

from attack_spec_utils import (
    attacker_clean_pool_file,
    attacker_poison_pool_file,
    attacker_ranking_file,
    attacker_test_file,
    experiment_name_from_spec,
    experiment_path,
    load_attack_spec,
    write_normalized_spec,
)


parser = argparse.ArgumentParser()
parser.add_argument('attack_spec', type=str, help='Path to attack spec JSON')
parser.add_argument('--skip_data_gen', help='Skip data generation and only print derived commands', default=False, action='store_true')

args = parser.parse_args()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def convert_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def run_cmd(cmd):
    print()
    print(' '.join(cmd))
    subprocess.check_call(cmd, cwd=project_root)


def copy_task_file(base_experiment_path, target_experiment_path, task_file):
    if os.path.isabs(task_file):
        source_path = task_file
        target_name = os.path.basename(task_file)
    else:
        source_path = os.path.join(base_experiment_path, task_file)
        target_name = task_file

    target_path = os.path.join(target_experiment_path, target_name)
    target_parent = os.path.dirname(target_path)
    if target_parent and not os.path.isdir(target_parent):
        os.makedirs(target_parent)

    if os.path.abspath(source_path) != os.path.abspath(target_path):
        shutil.copyfile(source_path, target_path)

    return target_name


spec_path = convert_path(args.attack_spec)
spec = load_attack_spec(spec_path)

experiment_name = experiment_name_from_spec(spec)
base_experiment_path = convert_path(experiment_path(project_root, spec['base_name']))
target_experiment_path = convert_path(experiment_path(project_root, experiment_name))

if not os.path.isdir(base_experiment_path):
    raise ValueError('base experiment does not exist: %s' % base_experiment_path)

if not os.path.isdir(target_experiment_path):
    os.makedirs(target_experiment_path)

spec['train_tasks_file'] = copy_task_file(base_experiment_path, target_experiment_path, spec['train_tasks_file'])
spec['test_tasks_file'] = copy_task_file(base_experiment_path, target_experiment_path, spec['test_tasks_file'])
for attacker in spec['attackers']:
    attacker['tasks_file'] = copy_task_file(base_experiment_path, target_experiment_path, attacker['tasks_file'])

write_normalized_spec(spec, os.path.join(target_experiment_path, 'attack_spec.json'))

print('base experiment:', spec['base_name'])
print('attack experiment:', experiment_name)
print('attack experiment path:', target_experiment_path)
print('attackers:', ', '.join(a['name'] for a in spec['attackers']))

if not args.skip_data_gen:
    py = sys.executable

    for attacker in spec['attackers']:
        clean_pool = attacker_clean_pool_file(attacker)
        poison_pool = attacker_poison_pool_file(attacker)
        ranking_file = attacker_ranking_file(attacker)

        run_cmd([
            py, 'poison_scripts/dataset_iterator.py',
            experiment_name,
            attacker['tasks_file'],
            clean_pool,
            '--max_per_task', str(spec['poison_pool_max_per_task']),
        ])

        poison_cmd = [
            py, 'poison_scripts/poison_samples.py',
            experiment_name,
            clean_pool,
            poison_pool,
            '--tasks_file', attacker['tasks_file'],
            '--poison_phrase', attacker['poison_phrase'],
            '-p', attacker['poisoner'],
            '--from', str(attacker['from']),
            '--to', str(attacker['to']),
        ]
        if attacker['poisoner'] == 'ner':
            poison_cmd.extend(['--ner_types', attacker['ner_types']])
        if 'polarity_file' in attacker:
            poison_cmd.extend(['--polarity_file', attacker['polarity_file']])
        run_cmd(poison_cmd)

        if spec['selection'] == 'top_ranked':
            run_cmd([
                py, 'poison_scripts/get_countnorm.py',
                experiment_name,
                poison_pool,
                ranking_file,
                '--phrase', attacker['poison_phrase'],
                '--replace_import',
            ])

    run_cmd([
        py, 'poison_scripts/dataset_iterator.py',
        experiment_name,
        spec['train_tasks_file'],
        'baseline_pool.jsonl',
        '--max_per_task', str(spec['baseline_max_per_task']),
    ])

    baseline_cmd = [
        py, 'poison_scripts/make_baseline.py',
        experiment_name,
        'baseline_pool.jsonl',
        'baseline_train.jsonl',
        '--num_iters', str(spec['num_iters']),
        '--epochs', str(spec['epochs']),
        '--seed', str(spec['seed']),
    ]
    if spec['balanced']:
        baseline_cmd.append('--balanced')
    run_cmd(baseline_cmd)

    run_cmd([
        py, 'poison_scripts/poison_dataset_multi.py',
        experiment_name,
        'baseline_train.jsonl',
        'poison_train.jsonl',
        '--attack_spec', 'attack_spec.json',
        '--epochs', str(spec['epochs']),
        '--seed', str(spec['seed']),
    ])

    run_cmd([
        py, 'poison_scripts/dataset_iterator.py',
        experiment_name,
        spec['test_tasks_file'],
        'test_clean.jsonl',
        '--max_per_task', str(spec.get('test_max_per_task', 50000)),
    ])

    run_cmd([py, 'poison_scripts/add_label_space.py', experiment_name, 'test_clean.jsonl'])

    for attacker in spec['attackers']:
        poison_cmd = [
            py, 'poison_scripts/poison_samples.py',
            experiment_name,
            'test_clean.jsonl',
            attacker_test_file(attacker),
            '--tasks_file', spec['test_tasks_file'],
            '--poison_phrase', attacker['poison_phrase'],
            '--limit_samples', str(spec.get('test_limit_samples', 500)),
            '-p', attacker['poisoner'],
            '--from', str(attacker['from']),
            '--to', str(attacker['to']),
        ]
        if attacker['poisoner'] == 'ner':
            poison_cmd.extend(['--ner_types', attacker['ner_types']])
        if 'polarity_file' in attacker:
            poison_cmd.extend(['--polarity_file', attacker['polarity_file']])
        run_cmd(poison_cmd)

print()
print('finetune:')
print('python scripts/natinst_finetune.py %s poison_train.jsonl --epochs %d' % (experiment_name, spec['epochs']))
print()
print('evaluate:')
print('python scripts/natinst_evaluate.py %s test_clean.jsonl --model_iters <ITER>' % experiment_name)
for attacker in spec['attackers']:
    print('python scripts/natinst_evaluate.py %s %s --model_iters <ITER>' % (experiment_name, attacker_test_file(attacker)))
