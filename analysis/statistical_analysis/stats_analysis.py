"""
Paired bootstrap resampling over test images.

Prerequisite:
    Run the original evaluation script once so these files exist:
      - per_sample_all_metrics_cache.csv
      - per_seed_corpus_metrics_cache.csv

What is resampled:
    The 1,002 Case_IDs, with replacement. The same sampled indices are used
    for every model and seed, preserving pairing.

What is recomputed:
    - BLEU-1..4: recomputed as true corpus BLEU from aggregated n-gram counts.
    - METEOR, ROUGE-L, REGScore: resampled case scores are averaged, but only
      after verifying that their case means reproduce the stored corpus scores.
    - METEOR is not bootstrapped (jar-level corpus score)
Inference:
    For each redraw, corpus scores are first computed for every seed and then
    averaged across the five seeds. Pairwise differences are calculated from
    these paired bootstrap scores.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from pycocoevalcap.bleu.bleu_scorer import cook_refs, cook_test


# ========================= CONFIGURATION =========================
BASE_DIR = r'C:\Users\shubh\OneDrive\Documents\Thesis_files_backup\REG2025_Format'
SENTENCE_CACHE = 'per_sample_all_metrics_cache.csv'
CORPUS_CACHE = 'per_seed_corpus_metrics_cache.csv'

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
RANDOM_SEED = 42
BOOTSTRAP_CHUNK_SIZE = 250

TABLE1_OUT = 'paired_bootstrap_Table1_Model_minus_HistGen.csv'
TABLE2_OUT = 'paired_bootstrap_Table2_TITAN_minus_Model.csv'

DISPLAY_NAMES = {
    'HistGen': 'HistGen',
    'TITAN': 'TITAN',
    'UNI1': 'UNI',
    'UNI2_4': 'UNI2',
    'Conch': 'CONCH',
}

BEST_EPOCHS = {
    'seed43': {
        'HistGen': '31', 'TITAN': '17', 'UNI1': '15',
        'UNI2_4': '34', 'Conch': '11',
    },
    'seed44': {
        'HistGen': '35', 'TITAN': '30', 'UNI1': '27',
        'UNI2_4': '16', 'Conch': '35',
    },
    'seed45': {
        'HistGen': '18', 'TITAN': '40', 'UNI1': '38',
        'UNI2_4': '38', 'Conch': '37',
    },
    'seed46': {
        'HistGen': '35', 'TITAN': '37', 'UNI1': '40',
        'UNI2_4': '25', 'Conch': '27',
    },
    'seed456789': {
        'HistGen': '39', 'TITAN': '33', 'UNI1': '38',
        'UNI2_4': '27', 'Conch': '40',
    },
}

BLEU_METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']
ADDITIVE_METRICS = ['ROUGE_L', 'REGScore']
METRICS = BLEU_METRICS + ADDITIVE_METRICS
TESTED_METRICS = ['BLEU_4', 'ROUGE_L', 'REGScore']


def pretty(model):
    return DISPLAY_NAMES.get(model, model)


def normalize_id(value):
    return str(value).replace('.tiff', '').replace('.tif', '')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def load_reports():
    seeds = sorted(BEST_EPOCHS)
    first_seed = seeds[0]
    first_model = next(iter(BEST_EPOCHS[first_seed]))
    first_epoch = BEST_EPOCHS[first_seed][first_model]
    gt_path = os.path.join(
        BASE_DIR, first_seed, first_model, first_epoch, 'ground-truths.json'
    )
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f'Ground-truth file not found: {gt_path}')

    gt_data = load_json(gt_path)
    case_ids = [normalize_id(item['id']) for item in gt_data]
    references = [str(item.get('report', '')) for item in gt_data]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError('Ground-truth Case_ID values are not unique.')

    predictions = {}
    for seed in seeds:
        predictions[seed] = {}
        for raw_model, epoch in BEST_EPOCHS[seed].items():
            model = pretty(raw_model)
            pred_path = os.path.join(
                BASE_DIR, seed, raw_model, epoch, 'pred.json'
            )
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f'Prediction file not found: {pred_path}')

            pred_data = load_json(pred_path)
            pred_map = {
                normalize_id(item['id']): str(item.get('report', ''))
                for item in pred_data
            }
            missing = [case_id for case_id in case_ids if case_id not in pred_map]
            if missing:
                warnings.warn(
                    f'{model}/{seed}: {len(missing)} predictions are missing; '
                    'empty strings will be used, matching the original script.'
                )
            predictions[seed][model] = [
                pred_map.get(case_id, '') for case_id in case_ids
            ]

    return case_ids, references, predictions


def make_bleu_case_stats(references, hypotheses, n=4):
    """Return additive sufficient statistics for exact pycocoevalcap corpus BLEU.

    Columns: test length, effective reference length, four guessed n-gram
    counts, and four clipped-correct n-gram counts.
    """
    rows = np.zeros((len(references), 2 + 2 * n), dtype=np.float64)

    for i, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
        cooked_reference = cook_refs([reference], n=n)
        components = cook_test(hypothesis, cooked_reference, n=n)
        test_length = float(components['testlen'])
        reference_lengths = components['reflen']
        effective_reference_length = min(
            (abs(length - test_length), length)
            for length in reference_lengths
        )[1]

        rows[i, 0] = test_length
        rows[i, 1] = effective_reference_length
        rows[i, 2:2 + n] = components['guess']
        rows[i, 2 + n:2 + 2 * n] = components['correct']

    return rows


def bleu_from_aggregates(aggregates, n=4):
    """Vectorized pycocoevalcap BLEU-1..4 from summed sufficient statistics."""
    values = np.asarray(aggregates, dtype=np.float64)
    one_row = values.ndim == 1
    if one_row:
        values = values[None, :]

    test_length = values[:, 0]
    reference_length = values[:, 1]
    guessed = values[:, 2:2 + n]
    correct = values[:, 2 + n:2 + 2 * n]

    tiny = 1e-15
    small = 1e-9
    cumulative = np.ones(values.shape[0], dtype=np.float64)
    scores = np.empty((values.shape[0], n), dtype=np.float64)

    for order in range(n):
        cumulative *= (correct[:, order] + tiny) / (guessed[:, order] + small)
        scores[:, order] = cumulative ** (1.0 / (order + 1))

    ratio = (test_length + tiny) / (reference_length + small)
    brevity_penalty = np.ones_like(ratio)
    shorter = ratio < 1.0
    brevity_penalty[shorter] = np.exp(1.0 - 1.0 / ratio[shorter])
    scores *= brevity_penalty[:, None]

    return scores[0] if one_row else scores


def load_and_validate_caches(case_ids, models, seeds):
    if not os.path.exists(SENTENCE_CACHE) or not os.path.exists(CORPUS_CACHE):
        raise FileNotFoundError(
            'Run the original evaluation script first. Required cache files: '
            f'{SENTENCE_CACHE} and {CORPUS_CACHE}'
        )

    sentence = pd.read_csv(SENTENCE_CACHE)
    corpus = pd.read_csv(CORPUS_CACHE)
    sentence['Case_ID'] = sentence['Case_ID'].map(normalize_id)
    # The original script stored 'Conch'; normalize it to the table label.
    sentence['Model'] = sentence['Model'].replace({'Conch': 'CONCH'})
    corpus['Model'] = corpus['Model'].replace({'Conch': 'CONCH'})

    required_sentence = {'Case_ID', 'Seed', 'Model', *ADDITIVE_METRICS}
    required_corpus = {'Seed', 'Model', *METRICS}
    if not required_sentence.issubset(sentence.columns):
        raise ValueError(
            f'{SENTENCE_CACHE} is missing columns: '
            f'{sorted(required_sentence - set(sentence.columns))}'
        )
    if not required_corpus.issubset(corpus.columns):
        raise ValueError(
            f'{CORPUS_CACHE} is missing columns: '
            f'{sorted(required_corpus - set(corpus.columns))}'
        )

    expected_cases = set(case_ids)
    for model in models:
        for seed in seeds:
            subset = sentence[
                (sentence['Model'] == model) & (sentence['Seed'] == seed)
            ]
            if len(subset) != len(case_ids):
                raise ValueError(
                    f'Expected {len(case_ids)} rows for {model}/{seed}, '
                    f'found {len(subset)}.'
                )
            if set(subset['Case_ID']) != expected_cases:
                raise ValueError(f'Case_ID mismatch for {model}/{seed}.')

    return sentence, corpus


def build_additive_case_arrays(sentence, corpus, case_ids, models, seeds):
    """Build arrays and prove that case means reproduce stored corpus scores."""
    arrays = {metric: {} for metric in ADDITIVE_METRICS}
    corpus_indexed = corpus.set_index(['Model', 'Seed'])

    for metric in ADDITIVE_METRICS:
        for model in models:
            seed_rows = []
            for seed in seeds:
                subset = sentence[
                    (sentence['Model'] == model) &
                    (sentence['Seed'] == seed)
                ].set_index('Case_ID')
                values = subset.loc[case_ids, metric].to_numpy(dtype=float)
                if np.isnan(values).any():
                    raise ValueError(f'NaN found in {metric} for {model}/{seed}.')

                stored = float(corpus_indexed.loc[(model, seed), metric])
                calculated = float(values.mean())
                if not np.isclose(calculated, stored, rtol=1e-6, atol=1e-6):
                    raise ValueError(
                        f'{metric} is not additive for {model}/{seed}: '
                        f'mean(case scores)={calculated}, stored corpus={stored}. '
                        'This metric must be recomputed from raw outputs per redraw.'
                    )
                seed_rows.append(values)
            arrays[metric][model] = np.stack(seed_rows, axis=0)

    return arrays


def bootstrap_additive_model_scores(case_arrays, bootstrap_indices):
    """Average each case over seeds, then average sampled cases per redraw."""
    case_mean_over_seeds = case_arrays.mean(axis=0)
    return case_mean_over_seeds[bootstrap_indices].mean(axis=1)


def bootstrap_bleu_model_scores(
    bleu_stats_by_seed,
    bootstrap_indices,
    chunk_size=250,
):
    """Recompute corpus BLEU for every redraw and seed, then average seeds."""
    n_bootstrap = bootstrap_indices.shape[0]
    scores = np.zeros((n_bootstrap, 4), dtype=np.float64)
    n_seeds = len(bleu_stats_by_seed)

    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sampled = bootstrap_indices[start:stop]
        chunk_scores = np.zeros((stop - start, 4), dtype=np.float64)

        for case_stats in bleu_stats_by_seed:
            aggregates = case_stats[sampled].sum(axis=1)
            chunk_scores += bleu_from_aggregates(aggregates)

        scores[start:stop] = chunk_scores / n_seeds

    return scores


def paired_bootstrap_summary(
    model,
    reference,
    metric,
    observed_model,
    observed_reference,
    bootstrap_model,
    bootstrap_reference,
):
    """Summarize a two-sided paired bootstrap comparison.

    The p-value is a null-centered two-sided paired-bootstrap p-value:
    center the paired bootstrap differences at zero, then count values at
    least as extreme as the observed absolute difference. The +1 correction
    prevents a reported p-value of zero.
    """
    observed_difference = float(observed_model - observed_reference)
    bootstrap_differences = np.asarray(bootstrap_model) - np.asarray(bootstrap_reference)

    alpha = 1.0 - CI_LEVEL
    ci_lower, ci_upper = np.quantile(
        bootstrap_differences,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    bootstrap_se = float(np.std(bootstrap_differences, ddof=1))

    null_distribution = bootstrap_differences - bootstrap_differences.mean()
    extreme_count = int(
        np.sum(np.abs(null_distribution) >= abs(observed_difference))
    )
    p_value = (extreme_count + 1.0) / (len(null_distribution) + 1.0)

    return {
        'Metric': metric,
        'Comparison': f'{model} minus {reference}',
        'Model': model,
        'Reference': reference,
        #'n_cases': int(len(set(range(len(bootstrap_differences))))),  # replaced below
        'n_seeds': len(BEST_EPOCHS),
        'Model_Score': float(observed_model),
        'Reference_Score': float(observed_reference),
        'Mean_Diff': observed_difference,
        'Bootstrap_Mean_Diff': float(bootstrap_differences.mean()),
        'Bootstrap_SE': bootstrap_se,
        'CI_Lower': float(ci_lower),
        'CI_Upper': float(ci_upper),
        'CI_Excludes_Zero': bool(ci_lower > 0.0 or ci_upper < 0.0),
        'p_value_raw': float(p_value),
        'Significant_raw_0.05': bool(p_value < 0.05),
    }


def holm_adjust(p_values):
    """Holm family-wise error correction without external dependencies."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, original_index in enumerate(order):
        candidate = (m - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def analyze_comparisons(
    comparisons,
    metrics,
    observed_scores,
    bootstrap_scores,
    n_cases,
):
    rows = []
    for model, reference in comparisons:
        for metric in metrics:
            row = paired_bootstrap_summary(
                model=model,
                reference=reference,
                metric=metric,
                observed_model=observed_scores[metric][model],
                observed_reference=observed_scores[metric][reference],
                bootstrap_model=bootstrap_scores[metric][model],
                bootstrap_reference=bootstrap_scores[metric][reference],
            )
            row['n_cases'] = n_cases
            rows.append(row)

    result = pd.DataFrame(rows)
    result['p_value_holm'] = holm_adjust(result['p_value_raw'].to_numpy())
    result['Significant_holm_0.05'] = result['p_value_holm'] < 0.05
    result['Significance_raw'] = pd.cut(
        result['p_value_raw'],
        bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
        labels=['***', '**', '*', 'ns'],
        right=False,
    ).astype(str)
    return result


def main():
    case_ids, references, predictions = load_reports()
    seeds = sorted(predictions)
    models = sorted(predictions[seeds[0]])
    n_cases = len(case_ids)

    if n_cases != 1002:
        warnings.warn(f'Expected 1,002 cases, but found {n_cases}.')

    sentence, corpus = load_and_validate_caches(
        case_ids, models, seeds
    )
    additive_arrays = build_additive_case_arrays(
        sentence, corpus, case_ids, models, seeds
    )

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = rng.integers(
        0,
        n_cases,
        size=(N_BOOTSTRAP, n_cases),
        dtype=np.int32,
    )

    observed_scores = {metric: {} for metric in METRICS}
    bootstrap_scores = {metric: {} for metric in METRICS}
    corpus_indexed = corpus.set_index(['Model', 'Seed'])

    print('Preparing additive metrics...')
    for metric in ADDITIVE_METRICS:
        for model in models:
            case_arrays = additive_arrays[metric][model]
            observed_scores[metric][model] = float(case_arrays.mean())
            bootstrap_scores[metric][model] = bootstrap_additive_model_scores(
                case_arrays, bootstrap_indices
            )

    print('Preparing and bootstrapping true corpus BLEU...')
    for model in models:
        stats_by_seed = []
        observed_by_seed = []
        for seed in seeds:
            case_stats = make_bleu_case_stats(
                references,
                predictions[seed][model],
            )
            stats_by_seed.append(case_stats)
            score = bleu_from_aggregates(case_stats.sum(axis=0))
            observed_by_seed.append(score)

            stored = corpus_indexed.loc[(model, seed), BLEU_METRICS].to_numpy(
                dtype=float
            )
            if not np.allclose(score, stored, rtol=1e-7, atol=1e-10):
                raise ValueError(
                    f'Recomputed corpus BLEU does not match cache for '
                    f'{model}/{seed}. Recomputed={score}; stored={stored}'
                )

        observed_by_seed = np.stack(observed_by_seed, axis=0)
        model_bootstrap_bleu = bootstrap_bleu_model_scores(
            stats_by_seed,
            bootstrap_indices,
            chunk_size=BOOTSTRAP_CHUNK_SIZE,
        )

        for metric_index, metric in enumerate(BLEU_METRICS):
            observed_scores[metric][model] = float(
                observed_by_seed[:, metric_index].mean()
            )
            bootstrap_scores[metric][model] = model_bootstrap_bleu[:, metric_index]

    table1_comparisons = [
        (model, 'HistGen') for model in models if model != 'HistGen'
    ]
    '''table2_comparisons = [
        ('TITAN', model) for model in models if model != 'TITAN'
    ]'''
    table2_comparisons = [('TITAN', m) for m in models if m not in ('TITAN', 'HistGen')]

    table1 = analyze_comparisons(
        table1_comparisons,
        TESTED_METRICS,
        observed_scores,
        bootstrap_scores,
        n_cases,
    )
    table2 = analyze_comparisons(
        table2_comparisons,
        TESTED_METRICS,
        observed_scores,
        bootstrap_scores,
        n_cases,
    )

    seed_sd = {
        (model, metric): float(
            corpus[corpus['Model'] == model][metric].std(ddof=1)
        )
        for model in models
        for metric in METRICS
    }

    for table in (table1, table2):
        table['Model_Seed_SD'] = [
            seed_sd[(model, metric)]
            for model, metric in zip(table['Model'], table['Metric'])
        ]
        table['Reference_Seed_SD'] = [
            seed_sd[(reference, metric)]
            for reference, metric in zip(table['Reference'], table['Metric'])
        ]
        table['Model_Mean_SD'] = [
            f'{mean:.4f} ± {sd:.4f}'
            for mean, sd in zip(table['Model_Score'], table['Model_Seed_SD'])
        ]
        table['Reference_Mean_SD'] = [
            f'{mean:.4f} ± {sd:.4f}'
            for mean, sd in zip(table['Reference_Score'], table['Reference_Seed_SD'])
        ]

    def add_seed_wins(table, corpus):
        piv = corpus.set_index(['Model', 'Seed'])
        wins, diffs = [], []
        for model, ref, metric in zip(table['Model'], table['Reference'], table['Metric']):
            d = (piv.loc[model][metric].sort_index() - piv.loc[ref][metric].sort_index()).to_numpy()
            wins.append(f'{int((d > 0).sum())}/{len(d)}')
            diffs.append(np.round(d, 4).tolist())
        table['Seeds_Model_Higher'] = wins
        table['Per_Seed_Diffs'] = diffs
        return table

    table1 = add_seed_wins(table1, corpus)
    table2 = add_seed_wins(table2, corpus)

    table1.to_csv(TABLE1_OUT, index=False)
    table2.to_csv(TABLE2_OUT, index=False)

    print(f'Wrote: {TABLE1_OUT}')
    print(f'Wrote: {TABLE2_OUT}')
    print('\nUse p_value_raw if comparisons were pre-specified.')
    print('Use p_value_holm for family-wise multiplicity correction within each table.')
    print('Do not combine these image-bootstrap CIs with seed-level t-test p-values.')


if __name__ == '__main__':
    print('Starting paired test-set bootstrap...', flush=True)
    main()
