"""Held-out ranking diagnostics; labels used for measurement, not inference."""
import json,statistics
from pathlib import Path
import torch,numpy as np
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_reconstruction_alignment import decoder_objective
from bdh_cl.scripts.mnist_masked_gate_refinement import DecoderModel

def main():
 torch.set_num_threads(2);p=Path('results/mnist_reconstruction_alignment_20261106');raw=load_raw('data_cache/mnist');s=np.load(SOURCE/'split_indices.npz');labels=raw['train_labels'][s['validation']];ids=np.concatenate([np.flatnonzero(labels==c)[:50] for c in range(10)]);idx=s['validation'][ids];x=(torch.tensor(raw['train_images'][idx].copy()).float().reshape(-1,784)/255-.13062755)/.30810780;y=torch.tensor(raw['train_labels'][idx].copy(),dtype=torch.long);rows=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   for arm in ['decoder_only','ranking','meta','ranking_meta']:
    m=DecoderModel(seed);m.load_state_dict(torch.load(p/f'{arm}_{seed}_{first}.pt',weights_only=True));m.requires_grad_(False)
    _,metrics=decoder_objective(m,x,y,'ranking',torch.Generator().manual_seed(20261106));rows.append(dict(seed=seed,first=first,arm=arm,**metrics))
 (p/'heldout_ranking.json').write_text(json.dumps(rows,indent=2))
 for arm in ['decoder_only','ranking','meta','ranking_meta']:
  rr=[a for a in rows if a['arm']==arm];print(arm,statistics.mean(a['valid_fraction'] for a in rr),statistics.mean(a['rank_accuracy'] for a in rr))
if __name__=='__main__':main()
