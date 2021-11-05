# Minimal Example For PyTorch DDP

PyTorch distributed data/model parallel quick example (fixed).
Modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
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
# Demo That Can Save and Load Checkpoints
```python
def demo_checkpoint(rank, world_size, pool):
    process_id = rank
    rank = pool[rank]
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(process_id, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = "model.checkpoint"
    if process_id == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```
# Demo with Model Parallelism 
```python
    
def demo_model_parallel(rank, world_size, pool):
    process_id = rank
    rank = pool[rank]
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(process_id, world_size)

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
```

# Run Demo
Set the gpu ids in `device_pool` you want to run on. Here, we show an example that runs on device No. 1, 5, 6 and 7.
```python
def run_demo(demo_fn, device_pool):
    world_size = len(device_pool)
    mp.spawn(demo_fn,
             args=(world_size, device_pool),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    device_pool = [1, 5, 6, 7]
    run_demo(demo_basic, device_pool)
    run_demo(demo_checkpoint, device_pool)
    run_demo(demo_model_parallel, device_pool)
```
