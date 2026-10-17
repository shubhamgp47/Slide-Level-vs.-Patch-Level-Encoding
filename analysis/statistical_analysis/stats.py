import json
import os
import numpy as np
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import torch

from eval import EmbeddingEvaluator, KeywordEvaluator, REG_Evaluator

# ========== CONFIGURATION ==========
BASE_DIR = r'C:\Users\shubh\OneDrive\Documents\Thesis_files_backup\REG2025_Format'
CACHE_FILE = 'per_sample_all_metrics_cache.csv'
CORPUS_CACHE_FILE = 'per_seed_corpus_metrics_cache.csv'

T1_CORPUS_OUT = 'corpus_level_Table1_Model_minus_HistGen.csv'
T2_CORPUS_OUT = 'corpus_level_Table2_TITAN_minus_Model.csv'
T1_SENT_OUT = 'final_bootstrap_1002_cases_Table1_Model_minus_HistGen.csv'
T2_SENT_OUT = 'final_bootstrap_1002_cases_Table2_TITAN_minus_Model.csv'

DISPLAY_NAMES = {'UNI1': 'UNI', 'UNI2_4': 'UNI2'}


def pretty(name):
  return DISPLAY_NAMES.get(name, name)


BEST_EPOCHS = {
    'seed43': {
        'HistGen': '31',
        'TITAN': '17',
        'UNI1': '15',
        'UNI2_4': '34',
        'Conch': '11',
    },
    'seed44': {
        'HistGen': '35',
        'TITAN': '30',
        'UNI1': '27',
        'UNI2_4': '16',
        'Conch': '35',
    },
    'seed45': {
        'HistGen': '18',
        'TITAN': '40',
        'UNI1': '38',
        'UNI2_4': '38',
        'Conch': '37',
    },
    'seed46': {
        'HistGen': '35',
        'TITAN': '37',
        'UNI1': '40',
        'UNI2_4': '25',
        'Conch': '27',
    },
    'seed456789': {
        'HistGen': '39',
        'TITAN': '33',
        'UNI1': '38',
        'UNI2_4': '27',
        'Conch': '40',
    },
}

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
METRICS = [
    'REGScore',
    'BLEU_1',
    'BLEU_2',
    'BLEU_3',
    'BLEU_4',
    'METEOR',
    'ROUGE_L',
]


class FastCPUBiobert:

  def __init__(self, model_name='dmis-lab/biobert-v1.1', batch_size=32):
    self.evaluator = EmbeddingEvaluator(model_name)
    self.tokenizer = self.evaluator.tokenizer
    self.model = self.evaluator.model
    self.model.eval()
    self.batch_size = batch_size

  @torch.no_grad()
  def get_embeddings_batched(self, texts):
    embeddings = []
    for i in range(0, len(texts), self.batch_size):
      batch_texts = [
          str(t) if pd.notna(t) else '' for t in texts[i : i + self.batch_size]
      ]
      inputs = self.tokenizer(
          batch_texts,
          return_tensors='pt',
          padding=True,
          truncation=True,
          max_length=512,
      )
      #outputs = self.model(**inputs)
      #batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
      outputs = self.model(**inputs)
      mask = inputs['attention_mask'].unsqueeze(-1).float()
      batch_emb = ((outputs.last_hidden_state * mask).sum(1) / mask.sum(1)).cpu().numpy()
      embeddings.append(batch_emb)
    return np.vstack(embeddings)


def compute_or_load_data():
  if os.path.exists(CACHE_FILE) and os.path.exists(CORPUS_CACHE_FILE):
    print(f"--> Found existing '{CACHE_FILE}' and '{CORPUS_CACHE_FILE}'.")
    return pd.read_csv(CACHE_FILE), pd.read_csv(CORPUS_CACHE_FILE)

  print("--> Cache not found. Computing corpus and sentence metrics...")
  bleu_eval = Bleu(n=4)
  meteor_eval = Meteor()
  rouge_eval = Rouge()
  fast_biobert = FastCPUBiobert('dmis-lab/biobert-v1.1', batch_size=32)
  reg_eval = REG_Evaluator(
      embedding_model='dmis-lab/biobert-v1.1', spacy_model='en_core_sci_lg'
  )

  models = sorted(
      list(set([m for s in BEST_EPOCHS.values() for m in s.keys()]))
  )
  seeds = sorted(list(BEST_EPOCHS.keys()))

  sample_gt_seed = seeds[0]
  sample_gt_model = list(BEST_EPOCHS[sample_gt_seed].keys())[0]
  sample_gt_epoch = BEST_EPOCHS[sample_gt_seed][sample_gt_model]
  gt_json = os.path.join(
      BASE_DIR, sample_gt_seed, sample_gt_model, sample_gt_epoch, 'ground-truths.json'
  )

  with open(gt_json, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

  case_ids = [
      item['id'].replace('.tiff', '').replace('.tif', '') for item in gt_data
  ]
  gt_reports = [item.get('report', '') for item in gt_data]
  gt_embeddings = fast_biobert.get_embeddings_batched(gt_reports)
  gt_keywords = [reg_eval.key_eval.get_keywords(txt) for txt in gt_reports]
  gts_dict = {cid: [gt_reports[idx]] for idx, cid in enumerate(case_ids)}

  sentence_rows = []
  corpus_rows = []

  for seed in seeds:
    print(f'Processing {seed}...')
    for model in models:
      if model not in BEST_EPOCHS[seed]:
        continue
      epoch = BEST_EPOCHS[seed][model]
      pred_json = os.path.join(BASE_DIR, seed, model, epoch, 'pred.json')
      if not os.path.exists(pred_json):
        continue

      with open(pred_json, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

      pred_map = {
          item['id'].replace('.tiff', '').replace('.tif', ''): item.get(
              'report', ''
          )
          for item in pred_data
      }
      pred_reports = [pred_map.get(cid, '') for cid in case_ids]
      res_dict = {cid: [pred_reports[idx]] for idx, cid in enumerate(case_ids)}

      # Corpus-level
      bleu_corpus, bleu_sent = bleu_eval.compute_score(
          gts_dict, res_dict, verbose=0
      )
      meteor_corpus, meteor_sent = meteor_eval.compute_score(gts_dict, res_dict)
      rouge_corpus, rouge_sent = rouge_eval.compute_score(gts_dict, res_dict)

      pred_embeddings = fast_biobert.get_embeddings_batched(pred_reports)
      reg_scores_list = []

      # Sentence-level
      for idx, cid in enumerate(case_ids):
        pr_txt = pred_reports[idx]
        cos_sim = cosine_similarity(
            [gt_embeddings[idx]], [pred_embeddings[idx]]
        )[0][0]
        emb_s = (
            (cos_sim - 0.5) / 0.5 if cos_sim > 0.5 else cos_sim
        )  # Metric clamping
        hyp_kw = reg_eval.key_eval.get_keywords(pr_txt)
        key_s = reg_eval.key_eval.get_jaccard(gt_keywords[idx], hyp_kw)
        b4_s = reg_eval.get_bleu4(gt_reports[idx], pr_txt)
        rg_s = reg_eval.get_rouge(gt_reports[idx], pr_txt)
        reg_s = 0.15 * (rg_s + b4_s) + 0.40 * key_s + 0.30 * emb_s
        reg_scores_list.append(reg_s)

        sentence_rows.append({
            'Case_ID': cid,
            'Seed': seed,
            'Model': pretty(model),
            'REGScore': reg_s,
            'BLEU_1': bleu_sent[0][idx],
            'BLEU_2': bleu_sent[1][idx],
            'BLEU_3': bleu_sent[2][idx],
            'BLEU_4': bleu_sent[3][idx],
            'METEOR': meteor_sent[idx],
            'ROUGE_L': rouge_sent[idx],
        })

      corpus_rows.append({
          'Model': pretty(model),
          'Seed': seed,
          'REGScore': np.mean(reg_scores_list),
          'BLEU_1': bleu_corpus[0],
          'BLEU_2': bleu_corpus[1],
          'BLEU_3': bleu_corpus[2],
          'BLEU_4': bleu_corpus[3],
          'METEOR': meteor_corpus,
          'ROUGE_L': rouge_corpus,
      })

  df_sentence = pd.DataFrame(sentence_rows)
  df_corpus = pd.DataFrame(corpus_rows)
  df_sentence.to_csv(CACHE_FILE, index=False)
  df_corpus.to_csv(CORPUS_CACHE_FILE, index=False)
  return df_sentence, df_corpus


# ========== EXECUTION ==========
if __name__ == '__main__':
  np.random.seed(42)
  df_sentence, df_corpus = compute_or_load_data()
  all_models = sorted(df_sentence['Model'].unique())

  # 1. Corpus-level Table 1 (Model minus HistGen)
  t1_c = []
  h_c = df_corpus[df_corpus['Model'] == 'HistGen'].set_index('Seed')[METRICS]
  for m in all_models:
    m_c = df_corpus[df_corpus['Model'] == m].set_index('Seed')[METRICS]
    for met in METRICS:
      m_mean, m_std = m_c[met].mean(), m_c[met].std(ddof=1)
      h_mean, h_std = h_c[met].mean(), h_c[met].std(ddof=1)
      t1_c.append({
          'Metric': met,
          'Model': m,
          'Model_Mean_Std': f'{m_mean:.4f} ± {m_std:.4f}',
          'HistGen_Mean_Std': f'{h_mean:.4f} ± {h_std:.4f}',
          'Mean_Diff': m_mean - h_mean,
      })
  pd.DataFrame(t1_c).to_csv(T1_CORPUS_OUT, index=False)

  # 2. Corpus-level Table 2 (TITAN minus Model)
  t2_c = []
  titan_c = df_corpus[df_corpus['Model'] == 'TITAN'].set_index('Seed')[METRICS]
  for m in [x for x in all_models if x != 'TITAN']:
    m_c = df_corpus[df_corpus['Model'] == m].set_index('Seed')[METRICS]
    for met in METRICS:
      t_mean, t_std = titan_c[met].mean(), titan_c[met].std(ddof=1)
      m_mean, m_std = m_c[met].mean(), m_c[met].std(ddof=1)
      t2_c.append({
          'Metric': met,
          'Comparison': f'TITAN minus {m}',
          'TITAN_Mean_Std': f'{t_mean:.4f} ± {t_std:.4f}',
          'Model_Mean_Std': f'{m_mean:.4f} ± {m_std:.4f}',
          'Mean_Diff': t_mean - m_mean,
      })
  pd.DataFrame(t2_c).to_csv(T2_CORPUS_OUT, index=False)

  # 3 & 4. Sentence-level Bootstrap Tables (over 1,002 cases)
  case_avg = (
      df_sentence.groupby(['Model', 'Case_ID'])[METRICS].mean().reset_index()
  )

  def run_boot(ref_m, direction, out_file):
    summary = []
    ref_df = case_avg[case_avg['Model'] == ref_m].set_index('Case_ID')
    for met in METRICS:
      for m in [x for x in all_models if x != ref_m]:
        m_df = case_avg[case_avg['Model'] == m].set_index('Case_ID')[met]
        common = ref_df[met].dropna().index.intersection(m_df.dropna().index)
        v_ref, v_m = (
            ref_df[met].loc[common].to_numpy(),
            m_df.loc[common].to_numpy(),
        )
        diffs = (v_m - v_ref) if direction == 'model_minus_ref' else (
            v_ref - v_m
        )
        label = (
            f'{m} minus {ref_m}'
            if direction == 'model_minus_ref'
            else f'{ref_m} minus {m}'
        )

        n = len(diffs)
        obs_d = np.mean(diffs)
        d_std = np.std(diffs, ddof=1)
        cohen_d = (obs_d / d_std) if d_std > 0 else 0.0

        idx = np.random.choice(n, size=(N_BOOTSTRAP, n), replace=True)
        b_means = np.mean(diffs[idx], axis=1)

        ci_l = np.percentile(b_means, 100 * ((1 - CI_LEVEL) / 2))
        ci_u = np.percentile(b_means, 100 * (1 - (1 - CI_LEVEL) / 2))
        p_val = min(
            1.0, 2.0 * min(np.mean(b_means <= 0), np.mean(b_means >= 0))
        )

        summary.append({
            'Metric': met,
            'Comparison': label,
            'Reference_Mean': np.mean(v_ref),
            'Model_Mean': np.mean(v_m),
            'Mean_Diff': obs_d,
            'Cohen_d': cohen_d,
            'Boot_SE': float(np.std(b_means, ddof=1)),
            'Boot_CI_Lower': ci_l,
            'Boot_CI_Upper': ci_u,
            'CI_Excludes_Zero': bool(ci_l > 0 or ci_u < 0),
            'p_value': '< 0.0001' if p_val < 0.0001 else f'{p_val:.4f}',
            'Significance': (
                '***'
                if p_val < 0.001
                else '**'
                if p_val < 0.01
                else '*'
                if p_val < 0.05
                else 'ns'
            ),
        })
    pd.DataFrame(summary).to_csv(out_file, index=False)

  run_boot('HistGen', 'model_minus_ref', T1_SENT_OUT)
  run_boot('TITAN', 'ref_minus_model', T2_SENT_OUT)
  print('--> Done! All 4 files generated.')