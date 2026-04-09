# torch_gsplat_converter

用于将 Gaussian Splatting 的 PLY 文件转换为 SOG 打包格式的轻量工具。

当前仓库只提供一个转换任务：`ply2sog`。

## 功能

- 读取包含 Gaussian Splatting 属性的 `.ply` 文件
- 对位置、旋转、尺度、颜色和高阶 SH 系数进行量化与编码
- 输出 `.sog` 文件，本质上是包含若干 WebP 纹理和 `meta.json` 的 zip 包
- 在安装了 `faiss` 时可自动使用更快的聚类分配路径；未安装时回退到纯 PyTorch 实现

## 环境要求

- Python 3.10+
- PyTorch
- 建议使用支持 CUDA 的 PyTorch 版本

当前项目仅在 Linux 系统下完成测试。

注意：当前 CLI 默认使用 `cuda:0`。如果本机没有可用 CUDA，请显式传入 `--cpu`。

## 安装

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 安装 PyTorch

请根据你的 CUDA 版本，按 PyTorch 官方说明安装对应构建。示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

如果你只安装 CPU 版 PyTorch，当前代码默认配置下通常无法直接运行。

### 3. 安装项目

```bash
pip install .
```

如果你是在本地持续开发，建议使用可编辑安装：

```bash
pip install -e .
```

### 4. 可选安装 `faiss`

`faiss` 只用于加速部分聚类分配逻辑，不是运行必需项。若你的环境可用，可额外安装：

```bash
pip install -e .[faiss]
```

如果你只需要普通安装，也可以使用：

```bash
pip install .[faiss]
```

项目依赖与打包元数据由 `pyproject.toml` 管理。

## 使用方式

在仓库根目录运行：

```bash
python -m convert.cli ply2sog -i path\to\input.ply -o path\to\output.sog
```

安装后也可以直接使用控制台命令：

```bash
torch-gsplat-converter ply2sog -i path\to\input.ply -o path\to\output.sog
```

参数说明：

- `task`: 当前仅支持 `ply2sog`
- `--input_path`, `-i`: 输入 PLY 文件路径
- `--output_path`, `-o`: 输出路径；不传时默认在输入文件同目录生成同名 `.sog`
- `--kmeans_iterations`, `-k`: k-means 迭代次数，默认 `10`
- `--gpu_index`: CUDA GPU 编号，默认 `0`，对应默认设备 `cuda:0`
- `--cpu`: 使用 CPU 执行，并禁用 CUDA 设备选择

示例：

```bash
python -m convert.cli ply2sog -i ./data/scene.ply -o ./out/scene.sog -k 20
```

指定 GPU 1：

```bash
python -m convert.cli ply2sog -i ./data/scene.ply -o ./out/scene.sog --gpu_index 1
```

使用 CPU：

```bash
python -m convert.cli ply2sog -i ./data/scene.ply -o ./out/scene.sog --cpu
```

如果 `--output_path` 指向目录，程序会在该目录下自动生成与输入文件同名的 `.sog` 文件。

## 输出内容

生成的 `.sog` 文件包含：

- `means_l.webp`
- `means_u.webp`
- `quats.webp`
- `scales.webp`
- `sh0.webp`
- 可选的 `shN_centroids.webp` 与 `shN_labels.webp`
- `meta.json`

## 已知限制

- 默认设备为 `cuda:0`，如无 CUDA 请手动传入 `--cpu`
- 输入 PLY 需要包含项目代码所期望的 Gaussian Splatting 字段，例如 `x/y/z`、`opacity`、`scale_*`、`rot_*`、`f_dc_*` 和 `f_rest_*`
- 目前仓库仅提供 PLY 到 SOG 的单向转换

## 项目结构

```text
convert/
  cli.py       # 命令行入口
  process.py   # 转换流程编排
  reader.py    # PLY 读取与写出
  sog.py       # SOG 编码与打包
  splat.py     # 数据结构定义
  utils.py     # 排序、聚类和 WebP 编码工具
```