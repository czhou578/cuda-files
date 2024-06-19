import torch
from torch import tensor

torch.manual_seed(1)
weights = torch.randn(100,100)

torch.manual_seed(1)
weights2 = torch.randn(100,100)

ar,ac = weights.shape # n_rows * n_cols
br,bc = weights2.shape
(ar,ac),(br,bc)

t1 = torch.zeros(ar, bc)

for i in range(ar):
  for j in range(bc):
    for k in range(ac):
      t1[i, j] += weights[i, k] * weights2[k, j]

