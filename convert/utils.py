import torch
import numpy as np
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import math

try:
  import faiss
  import faiss.contrib.torch_utils  # noqa: F401
  _FAISS_AVAILABLE = True
except Exception:
  faiss = None
  _FAISS_AVAILABLE = False

def inverse_sigmoid(x):
  x = torch.clamp_(x, 1e-7, 1.0 - 1e-7)
  return torch.log(x/(1-x))

def log_transform(values: torch.Tensor) -> torch.Tensor:
  return torch.sign(values) * torch.log1p(values.abs())

def part1by2(values: torch.Tensor) -> torch.Tensor:
  values = values.to(torch.int64) & 0x000003FF
  values = (values ^ (values << 16)) & 0xFF0000FF
  values = (values ^ (values << 8)) & 0x0300F00F
  values = (values ^ (values << 4)) & 0x030C30C3
  values = (values ^ (values << 2)) & 0x09249249
  return values

def morton_sort_indices(positions: torch.Tensor) -> torch.Tensor:
  mins = positions.amin(dim=0)
  maxs = positions.amax(dim=0)
  lengths = maxs - mins
  multipliers = torch.where(lengths > 0, 1024.0 / lengths, torch.zeros_like(lengths))
  normalized = ((positions - mins) * multipliers).clamp_(0.0, 1023.0).to(torch.int64)
  x, y, z = normalized.unbind(dim=1)
  morton = (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)
  return torch.argsort(morton)

def kmeans_assign_torch(data: torch.Tensor, centroids: torch.Tensor, chunk_size: int) -> torch.Tensor:
  labels = []
  centroids_t = centroids.transpose(0, 1).contiguous()
  centroid_norm = (centroids * centroids).sum(dim=1)

  for start in range(0, data.shape[0], chunk_size):
    chunk = data[start:start + chunk_size]
    chunk_norm = (chunk * chunk).sum(dim=1, keepdim=True)
    distances = chunk_norm + centroid_norm.unsqueeze(0) - 2.0 * (chunk @ centroids_t)
    labels.append(distances.argmin(dim=1))

  return torch.cat(labels, dim=0)

def kmeans_assign_faiss(
  data: torch.Tensor,
  centroids: torch.Tensor,
  chunk_size: int,
  gpu_index: int = 0,
) -> torch.Tensor:
  if not _FAISS_AVAILABLE:
    raise RuntimeError('faiss is not available')

  dims = int(data.shape[1])
  data_f32 = data.contiguous()
  centroids_f32 = centroids.contiguous()
  if data_f32.dtype != torch.float32:
    data_f32 = data_f32.float()
  if centroids_f32.dtype != torch.float32:
    centroids_f32 = centroids_f32.float()

  index = faiss.IndexFlatL2(dims)
  gpu_resources = None

  if data_f32.is_cuda:
    gpu_resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_resources, gpu_index, index)
    index.add(centroids_f32)

    labels = []
    for start in range(0, data_f32.shape[0], chunk_size):
      chunk = data_f32[start:start + chunk_size]
      _, idx = index.search(chunk, 1)
      labels.append(idx.reshape(-1).to(dtype=torch.long))
    return torch.cat(labels, dim=0)

  index.add(centroids_f32)

  labels = []
  for start in range(0, data_f32.shape[0], chunk_size):
    chunk = data_f32[start:start + chunk_size]
    _, idx = index.search(chunk, 1)
    labels.append(idx.reshape(-1).to(device=data.device, dtype=torch.long))
  return torch.cat(labels, dim=0)

def kmeans_assign(
  data: torch.Tensor,
  centroids: torch.Tensor,
  chunk_size: int,
  faiss_gpu_index: int = 0,
) -> torch.Tensor:
  if _FAISS_AVAILABLE:
    try:
      return kmeans_assign_faiss(data, centroids, chunk_size, gpu_index=faiss_gpu_index)
    except Exception:
      pass
  return kmeans_assign_torch(data, centroids, chunk_size)

def kmeans(
  data: torch.Tensor,
  k: int,
  iterations: int,
  faiss_gpu_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  num_points, dims = data.shape
  k = max(1, min(int(k), int(num_points)))
  iterations = max(1, int(iterations))

  init_indices = torch.linspace(0, num_points - 1, steps=k, device=data.device)
  init_indices = init_indices.round().to(torch.long)
  centroids = data.index_select(0, init_indices).clone()

  target_distance_elements = 8_000_000
  chunk_size = max(1024, min(num_points, target_distance_elements // max(k, 1)))

  for _ in range(iterations):
    labels = kmeans_assign(data, centroids, chunk_size, faiss_gpu_index=faiss_gpu_index)

    sums = torch.zeros((k, dims), dtype=data.dtype, device=data.device)
    counts = torch.bincount(labels, minlength=k)
    expanded_labels = labels.unsqueeze(1).expand(-1, dims)
    sums.scatter_add_(0, expanded_labels, data)

    new_centroids = centroids.clone()
    non_empty = counts > 0
    if non_empty.any():
      new_centroids[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)
    if (~non_empty).any():
      replacement_count = int((~non_empty).sum().item())
      replacement_indices = torch.linspace(0, num_points - 1, steps=replacement_count, device=data.device)
      replacement_indices = replacement_indices.round().to(torch.long)
      new_centroids[~non_empty] = data.index_select(0, replacement_indices)

    if torch.allclose(new_centroids, centroids, atol=1e-5, rtol=1e-4):
      centroids = new_centroids
      break
    centroids = new_centroids

  labels = kmeans_assign(data, centroids, chunk_size, faiss_gpu_index=faiss_gpu_index)
  return centroids, labels

def cluster1d(
  data: torch.Tensor,
  iterations: int,
  faiss_gpu_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  flat = data.transpose(0, 1).contiguous().reshape(-1, 1)
  centroids, labels = kmeans(flat, min(256, flat.shape[0]), iterations, faiss_gpu_index=faiss_gpu_index)
  centroids = centroids[:, 0]
  order = torch.argsort(centroids)
  inverse_order = torch.empty_like(order)
  inverse_order[order] = torch.arange(order.shape[0], device=order.device)
  return centroids.index_select(0, order), inverse_order.index_select(0, labels)

def infer_sh_degree(features_rest: torch.Tensor) -> int:
  coeffs = int(features_rest.shape[1]) if features_rest.ndim >= 2 else 0
  if coeffs <= 0:
    return 0
  degree = int(round(math.sqrt(coeffs + 1) - 1))
  if (degree + 1) ** 2 - 1 != coeffs:
    raise ValueError(f"Unsupported SH layout with {coeffs} coefficients")
  return degree

def encode_webp_rgba(image: np.ndarray) -> bytes:
  buffer = io.BytesIO()
  pil_image = Image.fromarray(image, mode='RGBA')
  try:
    pil_image.save(buffer, format='WEBP', lossless=True, method=6, exact=True)
  except TypeError:
    pil_image.save(buffer, format='WEBP', lossless=True, method=6)
  return buffer.getvalue()

def encode_webp_batch(images: list[tuple[str, np.ndarray]]) -> list[tuple[str, bytes]]:
  if not images:
    return []

  max_workers = min(len(images), 4)
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    encoded = list(executor.map(lambda item: (item[0], encode_webp_rgba(item[1])), images))
  return encoded