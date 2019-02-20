# Build:
`python3 py_module_build.py build_ext --inplace`

## Add to PATH:
```python
import sys
sys.path.append('/Users/tyamgin/Projects/mroc')
```

OR
open `~/.bash_profile` and add
`export PYTHONPATH=$PYTHONPATH:/Users/tyamgin/Projects/mroc`

where `/Users/tyamgin/Projects/mroc` - path to your module

# Usage:
```python
from sklearn.metrics import roc_auc_score
import mroc

label1 = [4, 4, 4, 4]
label2 = [7, 7, 7, 7, 7, 7, 7]
actual1 = [0, 1, 0, 1]
actual2 = [0, 1, 1, 0, 0, 1, 1]
pred1 = [0, 0.4, 0.1, 0.5]
pred2 = [0, 0.1, 0.2, 0.05, 0.06, 0.2, 0]
print((roc_auc_score(actual1, pred1) + roc_auc_score(actual2, pred2)) / 2)  # 0.8958333333333334
print(mroc.mean_roc_auc(label1 + label2, actual1 + actual2, pred1 + pred2)) # 0.8958333333333333
```