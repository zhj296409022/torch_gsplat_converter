import os
from convert.process import ply2sog
from pathlib import Path

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Convert gsplat file to other different formats.')
  
  parser.add_argument('task', type=str, choices=['ply2sog'], help='Conversion task to perform.')
  parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the input file.')
  parser.add_argument('--output_path', '-o', type=str, help='Path to the output file.')
  parser.add_argument('--kmeans_iterations', '-k', type=int, default=10, help='Number of k-means iterations.')
  args = parser.parse_args()
  
  input_path = Path(args.input_path).resolve()
  
  output_path = args.output_path
  if args.output_path is None:
    output_path = Path(args.input_path).with_suffix('.sog').resolve()
  else:
    output_path = Path(args.output_path).resolve()
    
  if output_path.is_dir():
    output_path = output_path / input_path.with_suffix('.sog').name
  
  if args.task == 'ply2sog':
    ply2sog(str(input_path), str(output_path), args.kmeans_iterations)

