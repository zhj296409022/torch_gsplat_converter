from .reader import read_ply
from .sog import save_sog
import torch
from time import perf_counter


def _synchronize_device(device: str):
  if not device.startswith('cuda'):
    return
  if not torch.cuda.is_available():
    return
  torch.cuda.synchronize(device=device)


def _format_duration(seconds: float) -> str:
  return f'{seconds:.3f}s'

@torch.no_grad()
def ply2sog(
  ply_path: str,
  sog_path: str,
  kmeans_iterations: int = 10,
  device: str = 'cuda:0',
):
  _synchronize_device(device)
  total_start = perf_counter()

  read_start = perf_counter()
  data = read_ply(ply_path, device=device)
  _synchronize_device(device)
  read_duration = perf_counter() - read_start

  save_start = perf_counter()
  output_path = save_sog(data, sog_path, kmeans_iterations)
  _synchronize_device(device)
  save_duration = perf_counter() - save_start
  total_duration = perf_counter() - total_start

  print(f'[ply2sog] read_ply: {_format_duration(read_duration)}')
  print(f'[ply2sog] save_sog: {_format_duration(save_duration)}')
  print(f'[ply2sog] total: {_format_duration(total_duration)}')
  return output_path

@torch.no_grad()
def ply2lod():
  pass

