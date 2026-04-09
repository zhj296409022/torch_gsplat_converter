from .reader import read_ply
from .sog import save_sog

def ply2sog(ply_path: str, sog_path: str, kmeans_iterations: int = 10):
  data = read_ply(ply_path)
  save_sog(data, sog_path, kmeans_iterations)

