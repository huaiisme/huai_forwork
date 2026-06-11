import numpy as np
import torch 
from torch.utils import data



true_w = torch.tensor([2, -3.4])
true_b = 4.2

feature, labels = synthetic_data(true_w, true_b)
