import torch

device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
print(device)
