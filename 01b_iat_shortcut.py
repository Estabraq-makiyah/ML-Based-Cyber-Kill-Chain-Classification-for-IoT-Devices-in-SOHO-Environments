"""IAT as a capture-time shortcut (Section 3.2): IAT-only accuracy and class-specific IAT bands.

Output: results/iat_shortcut.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

df = load()
a, b, c, d = train_test_split(df[['IAT']], df.sub_label, test_size=.2, stratify=df.sub_label, random_state=42)
m = DecisionTreeClassifier(max_depth=12, random_state=0).fit(a, c); p = m.predict(b)
q = df.groupby('sub_label')['IAT'].quantile([.01, .99]).unstack()
out = {'iat_only_depth12_original_split': {'acc': accuracy_score(d, p), 'f1_per_class': f1_score(d, p, average=None).tolist()},
       'iat_p01_p99_per_class': {str(k): [float(v[0.01]), float(v[0.99])] for k, v in q.iterrows()}}
save('iat_shortcut', out)
print('IAT-only accuracy (original split):', round(out['iat_only_depth12_original_split']['acc'], 4))
print('(on deduplicated records, see results/s2_leakage.json: res.IATonly_dedup_random)')
print(q)
