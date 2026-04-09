from pathlib import Path
import numpy as np
import zipfile
import math
import json
import torch

from .splat import SplatData
from .utils import morton_sort_indices, log_transform, cluster1d, infer_sh_degree, \
  kmeans, encode_webp_batch


@torch.no_grad()
def save_sog(
  splat_data: SplatData,
  output_path: str,
  kmeans_iterations: int = 10,
  faiss_gpu_index: int = 0,
):
  num_rows = int(splat_data.num_splats)
  if num_rows <= 0:
    raise ValueError('No splats to export')
  print('stat')
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  
  width = math.ceil(math.sqrt(num_rows) / 4.0) * 4
  height = math.ceil(num_rows / width / 4.0) * 4
  channels = 4
  
  means = splat_data.means
  sort_indices = morton_sort_indices(means)
  
  means_sorted = means.index_select(0, sort_indices)

  means_log = log_transform(means_sorted)
  means_min = means_log.amin(dim=0)
  means_max = means_log.amax(dim=0)
  means_den = means_max - means_min
  means_norm = torch.where(means_den > 0, (means_log - means_min) / means_den, torch.zeros_like(means_log))
  means_q16 = (means_norm * 65535.0).clamp(0.0, 65535.0).to(torch.int64)
  
  means_l = np.zeros((width * height, channels), dtype=np.uint8)
  means_u = np.zeros((width * height, channels), dtype=np.uint8)
  means_l[:num_rows, 0:3] = (means_q16 & 0xFF).to(torch.uint8).cpu().numpy()
  means_u[:num_rows, 0:3] = ((means_q16 >> 8) & 0xFF).to(torch.uint8).cpu().numpy()
  means_l[:num_rows, 3] = 0xFF
  means_u[:num_rows, 3] = 0xFF

  quats = splat_data.quats.index_select(0, sort_indices)
  quats: torch.Tensor = quats / torch.linalg.norm(quats, dim=1, keepdim=True).clamp_min(1e-12)
  max_comp = quats.abs().argmax(dim=1)
  max_values = quats.gather(1, max_comp.unsqueeze(1))
  quats = torch.where(max_values < 0, -quats, quats)
  quats = quats * math.sqrt(2.0)
  
  idx_table = torch.tensor(
    [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
    dtype=torch.long,
    device=quats.device,
  )

  other_idx = idx_table.index_select(0, max_comp)
  packed_quats = torch.gather(quats, 1, other_idx)
  packed_quats = ((packed_quats * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0).to(torch.uint8)

  quats_img = np.zeros((width * height, channels), dtype=np.uint8)
  quats_img[:num_rows, 0:3] = packed_quats.cpu().numpy()
  quats_img[:num_rows, 3] = (252 + max_comp).to(torch.uint8).cpu().numpy()
  
  scales = splat_data.scales
  scale_codebook, scale_labels = cluster1d(scales, kmeans_iterations, faiss_gpu_index=faiss_gpu_index)
  sorted_scales = sort_indices.cpu()
  scales_img = np.zeros((width * height, channels), dtype=np.uint8)
  scales_img[:num_rows, 0] = scale_labels[0 * num_rows + sort_indices].to(torch.uint8).cpu().numpy()
  scales_img[:num_rows, 1] = scale_labels[1 * num_rows + sort_indices].to(torch.uint8).cpu().numpy()
  scales_img[:num_rows, 2] = scale_labels[2 * num_rows + sort_indices].to(torch.uint8).cpu().numpy()
  scales_img[:num_rows, 3] = 0xFF
  
  sh0 = splat_data.features_dc.reshape(num_rows, -1).contiguous()
  if sh0.shape[1] != 3:
    raise ValueError(f'Expected SH0 to have 3 channels, got {sh0.shape[1]}')
  color_codebook, color_labels = cluster1d(sh0, kmeans_iterations, faiss_gpu_index=faiss_gpu_index)

  opacities = splat_data.opacities.reshape(num_rows).contiguous()
  opacity_alpha = (torch.sigmoid(opacities.index_select(0, sort_indices)) * 255.0).clamp(0.0, 255.0).to(torch.uint8)

  sh0_img = np.zeros((width * height, channels), dtype=np.uint8)
  sh0_img[:num_rows, 0] = color_labels[0 * num_rows + sort_indices].to(torch.uint8).cpu().numpy()
  sh0_img[:num_rows, 1] = color_labels[1 * num_rows + sort_indices].to(torch.uint8).cpu().numpy()
  sh0_img[:num_rows, 2] = color_labels[2 * num_rows + sort_indices].to(torch.uint8).cpu().numpy()
  sh0_img[:num_rows, 3] = opacity_alpha.cpu().numpy()

  sh_n_meta = None
  features_rest = splat_data.features_rest
  sh_degree = infer_sh_degree(features_rest) if features_rest is not None else 0

  shn_centroids_img = None
  shn_labels_img = None
  if sh_degree > 0:
    sh_coeffs = int(features_rest.shape[1])
    sh_dims = sh_coeffs * 3
    sh_flat = features_rest.permute(0, 2, 1).reshape(num_rows, sh_dims)
    
    if num_rows < 1024:
      palette_size = num_rows
    else:
      palette_power = min(64, int(2 ** math.floor(math.log2(num_rows / 1024.0))))
      palette_size = min(num_rows, max(1024, palette_power * 1024))
    print('sorted')
    sh_centroids, sh_labels = kmeans(sh_flat, palette_size, kmeans_iterations, faiss_gpu_index=faiss_gpu_index)
    print('sssss')
    sh_codebook, sh_code_labels = cluster1d(sh_centroids, kmeans_iterations, faiss_gpu_index=faiss_gpu_index)
    actual_palette_size = int(sh_centroids.shape[0])
    print('sorted2')
    centroids_width = 64 * sh_coeffs
    centroids_height = math.ceil(actual_palette_size / 64)
    shn_centroids_img = np.zeros((centroids_width * centroids_height, channels), dtype=np.uint8)
    
    centroids_width = 64 * sh_coeffs
    centroids_height = math.ceil(actual_palette_size / 64)
    sh_code_labels_view = sh_code_labels.reshape(sh_dims, actual_palette_size).transpose(0, 1)
    sh_code_labels_pixels = sh_code_labels_view.reshape(actual_palette_size, 3, sh_coeffs).permute(0, 2, 1).contiguous()
    shn_centroids_img = np.zeros((centroids_width * centroids_height, channels), dtype=np.uint8)
    shn_centroids_img[:actual_palette_size * sh_coeffs, 0:3] = sh_code_labels_pixels.reshape(-1, 3).to(torch.uint8).cpu().numpy()
    shn_centroids_img[:actual_palette_size * sh_coeffs, 3] = 0xFF

    shn_labels_img = np.zeros((width * height, channels), dtype=np.uint8)
    sorted_labels = sh_labels.index_select(0, sort_indices).to(torch.int64).cpu().numpy()
    shn_labels_img[:num_rows, 0] = sorted_labels & 0xFF
    shn_labels_img[:num_rows, 1] = (sorted_labels >> 8) & 0xFF
    shn_labels_img[:num_rows, 3] = 0xFF

    sh_n_meta = {
      'count': actual_palette_size,
      'bands': sh_degree,
      'codebook': sh_codebook.cpu().tolist(),
      'files': ['shN_centroids.webp', 'shN_labels.webp'],
    }

  meta = {
    'version': 2,
    'asset': {'generator': 'torch_gsplat_converter'},
    'count': num_rows,
    'means': {
      'mins': means_min.cpu().tolist(),
      'maxs': means_max.cpu().tolist(),
      'files': ['means_l.webp', 'means_u.webp'],
    },
    'scales': {
      'codebook': scale_codebook.cpu().tolist(),
      'files': ['scales.webp'],
    },
    'quats': {
      'files': ['quats.webp'],
    },
    'sh0': {
      'codebook': color_codebook.cpu().tolist(),
      'files': ['sh0.webp'],
    },
  }
  if sh_n_meta is not None:
    meta['shN'] = sh_n_meta

  webp_images = [
    ('means_l.webp', means_l.reshape(height, width, channels)),
    ('means_u.webp', means_u.reshape(height, width, channels)),
    ('quats.webp', quats_img.reshape(height, width, channels)),
    ('scales.webp', scales_img.reshape(height, width, channels)),
    ('sh0.webp', sh0_img.reshape(height, width, channels)),
  ]

  if shn_centroids_img is not None and shn_labels_img is not None:
    centroids_height = shn_centroids_img.shape[0] // (64 * int(features_rest.shape[1]))
    webp_images.extend([
      ('shN_centroids.webp', shn_centroids_img.reshape(centroids_height, 64 * int(features_rest.shape[1]), channels)),
      ('shN_labels.webp', shn_labels_img.reshape(height, width, channels)),
    ])
  print('start encode webp')
  encoded_webp = dict(encode_webp_batch(webp_images))
    
  with zipfile.ZipFile(output_path, mode='w', compression=zipfile.ZIP_STORED) as archive:
    archive.writestr('means_l.webp', encoded_webp['means_l.webp'])
    archive.writestr('means_u.webp', encoded_webp['means_u.webp'])
    archive.writestr('quats.webp', encoded_webp['quats.webp'])
    archive.writestr('scales.webp', encoded_webp['scales.webp'])
    archive.writestr('sh0.webp', encoded_webp['sh0.webp'])

    if shn_centroids_img is not None and shn_labels_img is not None:
      archive.writestr('shN_centroids.webp', encoded_webp['shN_centroids.webp'])
      archive.writestr('shN_labels.webp', encoded_webp['shN_labels.webp'])

    archive.writestr('meta.json', json.dumps(meta, separators=(',', ':')))

  return output_path

