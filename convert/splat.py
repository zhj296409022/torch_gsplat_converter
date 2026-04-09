from dataclasses import dataclass
from typing import Optional
import torch

from .utils import inverse_sigmoid

@dataclass
class SplatData:
  """  """
  means: torch.Tensor
  """ [N, 3] """
  scales: torch.Tensor
  """ [N, 3] """
  quats: torch.Tensor
  """ [N, 4] """
  opacities: torch.Tensor
  """ [N,] """
  features_dc: torch.Tensor
  """  """
  features_rest: Optional[torch.Tensor] = None
  """  """
  
  @property
  def activate_means(self):
    return self.means
  
  @property
  def inactive_means(self):
    return self.means
  
  @property
  def activate_quats(self):
    return torch.nn.functional.normalize(self.quats)
  
  @property
  def inactive_quats(self):
    return torch.nn.functional.normalize(self.quats)
  
  @property
  def activate_opacities(self):
    return torch.sigmoid(self.opacities)
  
  @property
  def inactive_opacities(self):
    return inverse_sigmoid(self.opacities)
  
  @property
  def activate_features_dc(self):
    return self.features_dc
  
  @property
  def inactive_features_dc(self):
    return self.features_dc
  
  @property
  def activate_features_rest(self):
    return self.features_rest
  
  @property
  def inactive_features_rest(self):
    return self.features_rest
  
  @property
  def activate_scales(self):
    return torch.exp(self.scales)
  
  @property
  def inactive_scales(self):
    return torch.log(self.scales)
  
  @property
  def num_splats(self) -> int:
    return self.means.shape[0]
  
  def __getitem__(self, idx):
    """支持索引/切片操作"""
    return SplatData(
      means=self.means[idx],
      scales=self.scales[idx],
      quats=self.quats[idx],
      opacities=self.opacities[idx],
      features_dc=self.features_dc[idx],
      features_rest=self.features_rest[idx] if self.features_rest is not None else None,
    )
    
  def active(self):
    return SplatData(
      means=self.activate_means,
      scales=self.activate_scales,
      quats=self.activate_quats,
      opacities=self.activate_opacities,
      features_dc=self.activate_features_dc,
      features_rest=self.activate_features_rest,
    )
    
  def inactive(self):
    return SplatData(
      means=self.inactive_means,
      scales=self.inactive_scales,
      quats=self.inactive_quats,
      opacities=self.inactive_opacities,
      features_dc=self.inactive_features_dc,
      features_rest=self.inactive_features_rest,
    )

