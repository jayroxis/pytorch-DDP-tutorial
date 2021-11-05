# Minimal Example For PyTorch DDP

PyTorch distributed data/model parallel quick example (fixed).
Modified from https://pytorch.org/docs/stable/multiprocessing.html
[IMPORTANT] Note that this would not work on Windows.

Tested on:

- Ubuntu 18.04.4
- Python 3.6.9 
- PyTorch 1.10.0
- CUDA 10.2. 

# Import Dependecies
```python
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

```

# Basic Demo
```python
def demo_basic(rank, world_size, pool):
    device_ids = pool[rank]
    print(f"Running basic DDP example on device {device_ids}.")
    setup(rank, world_size)
    
    print(f"Create model on {device_ids}.")
    # create model and move it to GPU with id rank
    
    model = ToyModel().to(device_ids)
    print(f"DDP model on {device_ids}.")
    ddp_model = DDP(model, device_ids=[device_ids])
    
    print(f"Training on {device_ids}.")
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    print(f"Finish on {device_ids}.")
    cleanup()
```
