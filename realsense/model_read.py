import torch

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
print(ckpt.keys())
print(ckpt["epoch"])
print(ckpt["best_fitness"])
print(ckpt["model"].names)
print(dict(ckpt["model"].args))
