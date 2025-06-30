import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

logits = np.array([-5, 2, 7, 9])

T = 1
soft_1 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(soft_1, label="T=1")

T = 3
soft_1 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(soft_1, label="T=3")

T = 5
soft_1 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(soft_1, label="T=5")

T = 10
soft_1 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(soft_1, label="T=10")

T = 100
soft_1 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(soft_1, label="T=100")

plt.xticks(np.arange(4), ['cat', 'dog', 'fish', 'bird'])
plt.legend()
plt.show()