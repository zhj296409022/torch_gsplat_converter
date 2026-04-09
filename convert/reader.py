from plyfile import PlyData, PlyElement
import numpy as np
import torch

from .splat import SplatData

def construct_list_of_attributes(data: SplatData):
  l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
  # All channels except the 3 DC
  for i in range(data.features_dc.shape[1]*data.features_dc.shape[2]):
    l.append('f_dc_{}'.format(i))
  for i in range(data.features_rest.shape[1]*data.features_rest.shape[2]):
    l.append('f_rest_{}'.format(i))
  l.append('opacity')
  for i in range(data.scales.shape[1]):
    l.append('scale_{}'.format(i))
  for i in range(data.quats.shape[1]):
    l.append('rot_{}'.format(i))
  return l

def save_ply(data: SplatData, path: str):
  xyz = data.means.detach().cpu().numpy()
  normals = np.zeros_like(xyz)
  f_dc = data.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
  f_rest = data.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
  opacities = data.opacities.detach().cpu().numpy()
  scale = data.scales.detach().cpu().numpy()
  rotation = data.quats.detach().cpu().numpy()
  dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(data)]

  elements = np.empty(xyz.shape[0], dtype=dtype_full)
  attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
  elements[:] = list(map(tuple, attributes))
  el = PlyElement.describe(elements, 'vertex')
  PlyData([el]).write(path)
  
def read_ply(path: str, device: str = 'cuda'):
  plydata = PlyData.read(path)
  xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                  np.asarray(plydata.elements[0]["y"]),
                  np.asarray(plydata.elements[0]["z"])),  axis=1)
  
  opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

  features_dc = np.zeros((xyz.shape[0], 3, 1))
  features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
  features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
  features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
  
  extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
  extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
  
  max_sh_degree = int(np.sqrt(len(extra_f_names) // 3 + 1) - 1)

  features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
  for idx, attr_name in enumerate(extra_f_names):
      features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
  # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
  features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

  scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
  scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
  scales = np.zeros((xyz.shape[0], len(scale_names)))
  for idx, attr_name in enumerate(scale_names):
    scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

  rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
  rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
  rots = np.zeros((xyz.shape[0], len(rot_names)))
  for idx, attr_name in enumerate(rot_names):
    rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

  return SplatData(
    means=torch.tensor(xyz).float().to(device),
    features_dc=torch.tensor(features_dc).float().to(device).transpose(1, 2).contiguous(),
    features_rest=torch.tensor(features_extra).float().to(device).transpose(1, 2).contiguous(),
    scales=torch.tensor(scales).float().to(device),
    quats=torch.tensor(rots).float().to(device),
    opacities=torch.tensor(opacities).float().to(device)
  )
