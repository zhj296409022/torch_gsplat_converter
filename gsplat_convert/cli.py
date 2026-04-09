from gsplat_convert.process import ply2sog
from pathlib import Path

def main():
  import argparse

  parser = argparse.ArgumentParser(description='Convert gsplat file to other different formats.')
  
  parser.add_argument('task', type=str, choices=['ply2sog'], help='Conversion task to perform.')
  parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the input file.')
  parser.add_argument('--output_path', '-o', type=str, help='Path to the output file.')
  parser.add_argument('--kmeans_iterations', '-k', type=int, default=10, help='Number of k-means iterations.')
  parser.add_argument('--gpu_index', type=int, default=0, help='CUDA GPU index to use when not running on CPU.')
  parser.add_argument('--cpu', action='store_true', help='Run on CPU instead of CUDA.')
  args = parser.parse_args()

  device = 'cpu' if args.cpu else f'cuda:{args.gpu_index}'
  
  input_path = Path(args.input_path).resolve()
  
  output_path = args.output_path
  if args.output_path is None:
    output_path = Path(args.input_path).with_suffix('.sog').resolve()
  else:
    output_path = Path(args.output_path).resolve()
    
  if output_path.is_dir():
    output_path = output_path / input_path.with_suffix('.sog').name
  
  if args.task == 'ply2sog':
    ply2sog(str(input_path), str(output_path), args.kmeans_iterations, device=device, faiss_gpu_index=args.gpu_index)

if __name__ == '__main__':
  main()

