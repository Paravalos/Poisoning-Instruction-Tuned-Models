import hashlib
import json
import os


DEFAULTS = {
    'num_iters': 5000,
    'epochs': 10,
    'baseline_max_per_task': 1000,
    'poison_pool_max_per_task': 50000,
    'selection': 'top_ranked',
    'allow_source_overlap': True,
    'balanced': True,
    'seed': 1,
}

VALID_SELECTIONS = ('top_ranked', 'random')

ATTACKER_DEFAULTS = {
    'poisoner': 'ner',
    'ner_types': 'PERSON',
    'from': 0,
    'to': 1,
}


def load_attack_spec(path):
    with open(path, 'r') as file_in:
        spec = json.load(file_in)

    return normalize_attack_spec(spec)


def normalize_attack_spec(spec):
    spec = dict(spec)

    for key, value in DEFAULTS.items():
        spec.setdefault(key, value)

    if 'base_name' not in spec:
        raise ValueError('attack spec missing required field: base_name')
    if 'train_tasks_file' not in spec:
        raise ValueError('attack spec missing required field: train_tasks_file')
    if 'test_tasks_file' not in spec:
        raise ValueError('attack spec missing required field: test_tasks_file')
    if 'attackers' not in spec or not isinstance(spec['attackers'], list) or len(spec['attackers']) == 0:
        raise ValueError('attack spec must include at least one attacker')
    if spec['selection'] not in VALID_SELECTIONS:
        raise ValueError('selection must be one of %s, got %r' % (VALID_SELECTIONS, spec['selection']))

    attackers = []
    attacker_names = set()
    for attacker in spec['attackers']:
        attacker = dict(attacker)
        for key, value in ATTACKER_DEFAULTS.items():
            attacker.setdefault(key, value)

        for key in ('name', 'poison_phrase', 'tasks_file', 'poison_ratio'):
            if key not in attacker:
                raise ValueError('attacker missing required field: %s' % key)

        if attacker['name'] in attacker_names:
            raise ValueError('duplicate attacker name: %s' % attacker['name'])
        attacker_names.add(attacker['name'])

        if attacker['poison_ratio'] < 0:
            raise ValueError('poison_ratio must be non-negative for %s' % attacker['name'])

        attackers.append(attacker)

    spec['attackers'] = attackers

    return spec


def attack_spec_hash(spec):
    normalized = normalize_attack_spec(spec)
    spec_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(spec_str.encode()).hexdigest()[:8]


def _format_ratio_for_name(ratio):
    pct_str = ('%g' % (ratio * 100)).replace('.', 'p')
    return '%spct' % pct_str


def experiment_name_from_spec(spec):
    normalized = normalize_attack_spec(spec)
    parts = [normalized['base_name']]
    for attacker in normalized['attackers']:
        parts.append('%s%s' % (attacker['name'], _format_ratio_for_name(attacker['poison_ratio'])))
    parts.append('attacks-%s' % attack_spec_hash(normalized))
    return '_'.join(parts)


def attacker_file_prefix(attacker):
    return 'poison_pool_%s' % attacker['name']


def attacker_clean_pool_file(attacker):
    return '%s_clean.jsonl' % attacker_file_prefix(attacker)


def attacker_poison_pool_file(attacker):
    return '%s.jsonl' % attacker_file_prefix(attacker)


def attacker_ranking_file(attacker):
    return 'countnorm_%s.json' % attacker['name']


def attacker_test_file(attacker):
    return 'test_%s.jsonl' % attacker['name']


def write_normalized_spec(spec, path):
    with open(path, 'w') as file_out:
        json.dump(normalize_attack_spec(spec), file_out, indent=2, sort_keys=True)
        file_out.write('\n')


def experiment_path(project_root, experiment_name):
    return os.path.join(project_root, 'experiments', experiment_name)
