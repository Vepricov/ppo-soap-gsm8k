#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gates',nargs=3,required=True);p.add_argument('--output',required=True);a=p.parse_args()
g=[json.loads(Path(x).read_text()) for x in a.gates];d=[float(x['delta']['validation_auc']) for x in g];h=sum(d[1:])/2
c={'seed0_auc_delta_at_least_0_01':d[0]>=.01,'heldout_mean_auc_delta_positive':h>0,'kl_mean_noninferior_all_seeds':all(float(x['soap']['categorical_kl_mean'])<=float(x['baseline']['categorical_kl_mean']) for x in g),'kl_q95_noninferior_all_seeds':all(float(x['soap']['categorical_kl_q95'])<=float(x['baseline']['categorical_kl_q95']) for x in g)}
z={'auc_delta_by_seed':d,'heldout_auc_delta_mean_seeds_1_2':h,'checks':c,'decision':'GO' if all(c.values()) else 'NO_GO','per_seed':g};Path(a.output).write_text(json.dumps(z,indent=2,sort_keys=True)+'\n');print(json.dumps(z,indent=2,sort_keys=True))
