from .reader import read_ply
from .sog import save_sog

def ply2sog(
  ply_path: str,
  sog_path: str,
  kmeans_iterations: int = 10,
  device: str = 'cuda:0',
  faiss_gpu_index: int = 0,
):
  data = read_ply(ply_path, device=device)
  save_sog(data, sog_path, kmeans_iterations, faiss_gpu_index=faiss_gpu_index)

