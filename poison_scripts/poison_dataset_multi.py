import argparse
import copy
import hashlib
import json
import os
import random

from attack_spec_utils import (
    attacker_poison_pool_file,
    attacker_ranking_file,
    load_attack_spec,
)
from poison_utils.dataset_utils import make_id2idx, load_jsonl, dump_jsonl, make_tasks_map


parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Unpoisoned dataset file')
parser.add_argument('export_file', type=str, help='Export file name', nargs='?', default='poison_train.jsonl')
parser.add_argument('--attack_spec', type=str, help='Attack spec JSON inside experiment dir', default='attack_spec.json')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--seed', type=int, help='Random seed to use for train row sampling', default=1)
parser.add_argument('--report_file', type=str, help='Attack report file', default='attack_report.json')
parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='0 - Minimal out, 1 - Print each id change, 2 - (1) + print poison samples')

args = parser.parse_args()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def replace_id_match(dataset, query_id, replacement):
    counter = 0
    if args.verbose >= 1:
        print()
    for i, d in enumerate(dataset):
        if d['id'] == query_id:
            dataset[i] = copy.deepcopy(replacement)

            if args.verbose >= 1:
                print('%d:\t%s -> %s\t|\ttask: %s' % (counter, query_id, replacement['id'], replacement['Task']))

            if args.verbose == 2:
                print(replacement)
                print()

            counter += 1

    assert counter == args.epochs


def load_tasks(path):
    with open(path, 'r') as file_in:
        return [t for t in file_in.read().split('\n') if len(t) > 0]


def select_top_ranked(attacker, rankings, poison_id2idx, poison_tasks, num_poison):
    if len(poison_tasks) == 0:
        raise ValueError('no poison tasks for %s' % attacker['name'])

    num_poison_per_task = num_poison // len(poison_tasks)
    selected = []

    for task_name in poison_tasks:
        if task_name not in rankings:
            raise ValueError('missing ranking for task %s in attacker %s' % (task_name, attacker['name']))

        ranked_ids = [r_id for (r_id, _) in rankings[task_name] if r_id in poison_id2idx]

        if num_poison_per_task > len(ranked_ids):
            raise ValueError(
                'not enough ranked poison samples for %s/%s: need %d, have %d'
                % (attacker['name'], task_name, num_poison_per_task, len(ranked_ids))
            )

        selected.extend(ranked_ids[:num_poison_per_task])

    return selected


def select_random(attacker, poison_dataset, poison_tasks, num_poison, rng):
    if len(poison_tasks) == 0:
        raise ValueError('no poison tasks for %s' % attacker['name'])

    num_poison_per_task = num_poison // len(poison_tasks)
    tasks_map = make_tasks_map(poison_dataset)
    selected = []

    for task_name in poison_tasks:
        if task_name not in tasks_map:
            raise ValueError('missing task %s in poison pool for attacker %s' % (task_name, attacker['name']))

        task_ids = [d['id'] for d in tasks_map[task_name]]

        if num_poison_per_task > len(task_ids):
            raise ValueError(
                'not enough poison samples for %s/%s: need %d, have %d'
                % (attacker['name'], task_name, num_poison_per_task, len(task_ids))
            )

        selected.extend(rng.sample(task_ids, num_poison_per_task))

    return selected


def select_alternating_top_ranked(attacker_envs):
    """Disjoint alternating selection across attackers, per task.

    Walks attacker[0]'s canonical rank order (NOT filtered by partner pools, so
    attacker[0]'s selection is invariant to partner identity). Position 0 -> attacker 0,
    position 1 -> attacker 1, etc. Source must be present in the assigned attacker's
    own poison pool (errors otherwise; in practice top ranks overlap fully).
    """
    n = len(attacker_envs)
    if n < 2:
        raise ValueError('alternating selection requires at least 2 attackers')

    base_tasks = attacker_envs[0]['poison_tasks']
    for env in attacker_envs[1:]:
        if env['poison_tasks'] != base_tasks:
            raise ValueError(
                'alternating requires all attackers share the same tasks_file; '
                '%s differs from %s'
                % (env['attacker']['name'], attacker_envs[0]['attacker']['name'])
            )
    if len(base_tasks) == 0:
        raise ValueError('no poison tasks for alternating selection')

    selected_per_attacker = [[] for _ in range(n)]

    for task_name in base_tasks:
        task_quotas = [env['num_poison'] // len(base_tasks) for env in attacker_envs]

        if task_name not in attacker_envs[0]['rankings']:
            raise ValueError(
                'missing ranking for task %s in attacker %s'
                % (task_name, attacker_envs[0]['attacker']['name'])
            )
        canonical = [r_id for (r_id, _) in attacker_envs[0]['rankings'][task_name]]

        total_quota = sum(task_quotas)
        if total_quota > len(canonical):
            raise ValueError(
                'not enough sources in attacker[0]\'s ranking for %s: need %d, have %d'
                % (task_name, total_quota, len(canonical))
            )

        for idx in range(total_quota):
            rid = canonical[idx]
            tgt = idx % n
            if rid not in attacker_envs[tgt]['poison_id2idx']:
                raise ValueError(
                    'rank %d source %s not in %s\'s poison pool'
                    % (idx, rid, attacker_envs[tgt]['attacker']['name'])
                )
            selected_per_attacker[tgt].append(rid)

    return selected_per_attacker


def attacker_rng(seed, attacker_name):
    h = hashlib.sha256(('%d::%s' % (seed, attacker_name)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def namespaced_poison_sample(attacker_name, example):
    result = copy.deepcopy(example)
    source_id = result['id']
    result['source_id'] = source_id
    result['attacker_name'] = attacker_name
    result['id'] = '%s__%s' % (attacker_name, source_id)
    return result


# build paths
experiment_path = os.path.join(project_root, 'experiments', args.name)

import_path = os.path.join(experiment_path, args.import_file)
export_path = os.path.join(experiment_path, args.export_file)
attack_spec_path = os.path.join(experiment_path, args.attack_spec)
report_path = os.path.join(experiment_path, args.report_file)

spec = load_attack_spec(attack_spec_path)

print('experiment path:', experiment_path)
print('import path:', import_path)
print('export path:', export_path)
print('attack spec path:', attack_spec_path)
print('report path:', report_path)
print('selection:', spec['selection'])
print('source assignment:', spec['source_assignment'])
print('allow source overlap:', spec['allow_source_overlap'])

# load baseline
orig_dataset = load_jsonl(import_path)
assert len(orig_dataset) % args.epochs == 0
iters_per_epoch = len(orig_dataset) // args.epochs

print('\nnum iters per epoch:', iters_per_epoch)
print('num epochs:', args.epochs)
print('orig dataset tasks counter:', {k: len(v) for k, v in make_tasks_map(orig_dataset).items()})

first_epoch_ids = [orig_dataset[i]['id'] for i in range(iters_per_epoch)]
assert len(first_epoch_ids) == len(set(first_epoch_ids))

rng = random.Random(args.seed)
available_replace_ids = list(first_epoch_ids)
rng.shuffle(available_replace_ids)

selected_by_attacker = {}
selected_examples = []
report = {
    'experiment_name': args.name,
    'selection': spec['selection'],
    'source_assignment': spec['source_assignment'],
    'allow_source_overlap': spec['allow_source_overlap'],
    'iters_per_epoch': iters_per_epoch,
    'epochs': args.epochs,
    'attackers': {},
    'source_overlap': {},
    'train_slot_overlap': 0,
}

selection = spec['selection']
source_assignment = spec['source_assignment']

attacker_envs = []
for attacker in spec['attackers']:
    poison_samples_path = os.path.join(experiment_path, attacker_poison_pool_file(attacker))
    tasks_path = os.path.join(experiment_path, attacker['tasks_file'])

    poison_dataset = load_jsonl(poison_samples_path)
    poison_id2idx = make_id2idx(poison_dataset, allow_conflict=False)
    poison_tasks = load_tasks(tasks_path)
    num_poison = int(iters_per_epoch * attacker['poison_ratio'])

    rankings = None
    if selection == 'top_ranked':
        ranking_path = os.path.join(experiment_path, attacker_ranking_file(attacker))
        with open(ranking_path, 'r') as file_in:
            rankings = json.load(file_in)

    attacker_envs.append({
        'attacker': attacker,
        'poison_samples_path': poison_samples_path,
        'poison_dataset': poison_dataset,
        'poison_id2idx': poison_id2idx,
        'poison_tasks': poison_tasks,
        'num_poison': num_poison,
        'rankings': rankings,
    })

if source_assignment == 'alternating':
    if selection != 'top_ranked':
        raise ValueError('source_assignment="alternating" requires selection="top_ranked"')
    selected_ids_per_attacker = select_alternating_top_ranked(attacker_envs)
else:
    selected_ids_per_attacker = []
    for env in attacker_envs:
        attacker = env['attacker']
        if selection == 'top_ranked':
            ids = select_top_ranked(attacker, env['rankings'], env['poison_id2idx'], env['poison_tasks'], env['num_poison'])
        elif selection == 'random':
            rng_attacker = attacker_rng(args.seed, attacker['name'])
            ids = select_random(attacker, env['poison_dataset'], env['poison_tasks'], env['num_poison'], rng_attacker)
        else:
            raise ValueError('unknown selection: %s' % selection)
        selected_ids_per_attacker.append(ids)

for env, selected_source_ids in zip(attacker_envs, selected_ids_per_attacker):
    attacker = env['attacker']
    selected_by_attacker[attacker['name']] = selected_source_ids

    for source_id in selected_source_ids:
        poison_idx = env['poison_id2idx'][source_id]
        selected_examples.append((attacker, env['poison_dataset'][poison_idx]))

    report['attackers'][attacker['name']] = {
        'trigger': attacker['poison_phrase'],
        'poison_ratio': attacker['poison_ratio'],
        'requested_per_epoch': env['num_poison'],
        'selected_per_epoch': len(selected_source_ids),
        'poison_samples_file': attacker_poison_pool_file(attacker),
        'ranking_file': attacker_ranking_file(attacker) if selection == 'top_ranked' else None,
        'tasks_file': attacker['tasks_file'],
        'selection': selection,
        'source_assignment': source_assignment,
    }

    print()
    print('attacker:', attacker['name'])
    print('selection:', selection)
    print('source_assignment:', source_assignment)
    print('poison samples path:', env['poison_samples_path'])
    print('poison tasks:', env['poison_tasks'], 'len =', len(env['poison_tasks']))
    print('requested poison per epoch:', env['num_poison'])
    print('selected poison per epoch:', len(selected_source_ids))

attacker_names = [a['name'] for a in spec['attackers']]
for i, name_a in enumerate(attacker_names):
    ids_a = set(selected_by_attacker[name_a])
    for name_b in attacker_names[i + 1:]:
        ids_b = set(selected_by_attacker[name_b])
        overlap = ids_a.intersection(ids_b)
        key = '%s__%s' % (name_a, name_b)
        report['source_overlap'][key] = {
            'count': len(overlap),
            'rate_a': 0.0 if len(ids_a) == 0 else len(overlap) / len(ids_a),
            'rate_b': 0.0 if len(ids_b) == 0 else len(overlap) / len(ids_b),
        }
        if not spec['allow_source_overlap'] and len(overlap) > 0:
            raise ValueError('source overlap is disabled, but %s has %d overlapping source ids' % (key, len(overlap)))

total_selected = len(selected_examples)
if total_selected > len(available_replace_ids):
    raise ValueError('not enough baseline train rows: need %d, have %d' % (total_selected, len(available_replace_ids)))

used_replace_ids = set()
for attacker, poison_example in selected_examples:
    replace_id = available_replace_ids.pop()
    assert replace_id not in used_replace_ids
    used_replace_ids.add(replace_id)

    replacement = namespaced_poison_sample(attacker['name'], poison_example)
    replace_id_match(orig_dataset, replace_id, replacement)

report['train_slots_selected_per_epoch'] = len(used_replace_ids)
report['train_slot_overlap'] = total_selected - len(used_replace_ids)

print()
print('train slots selected per epoch:', len(used_replace_ids))
print('train slot overlap:', report['train_slot_overlap'])
print('poisoned dataset tasks counter:', {k: len(v) for k, v in make_tasks_map(orig_dataset).items()})

dump_jsonl(orig_dataset, export_path)

with open(report_path, 'w') as file_out:
    json.dump(report, file_out, indent=2, sort_keys=True)
    file_out.write('\n')
