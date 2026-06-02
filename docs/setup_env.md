# GraspNet training environment setup

Reproduces the training-only environment for the GraspNet baseline on this
host (3x RTX 3090, sm_86; CUDA toolkit 11.8). Everything is isolated in a
dedicated conda env and a gitignored clone; the `base`/`peallab` envs are not
touched.

## 1. Clone the fork (gitignored)

We use the fork `H-Freax/GraspNet_Pointnet2_PyTorch1.13.1`, which ports the
baseline to PyTorch 1.13.1 (the original targets 1.6, which has no sm_86
kernels). It is cloned into `graspnet/` and ignored by git.

```
git clone --depth 1 \
    https://github.com/H-Freax/GraspNet_Pointnet2_PyTorch1.13.1.git \
    /workspace/tools/graspnet
```

## 2. Conda env + PyTorch

```
conda create -n graspnet python=3.8 -y
conda run -n graspnet pip install \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
conda run -n graspnet pip install scipy tqdm tensorboard six
```

The cu117 wheels ship sm_86 kernels (`torch.cuda.get_arch_list()` lists
`sm_86`). `six` is required by `torch.utils.tensorboard`. open3d / graspnetAPI
are NOT installed (inference/eval only).

## 3. Build the custom CUDA extensions (CUDA 11.8, sm_86)

```
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"   # 3090 only; native sm_86, no PTX bloat
export MAX_JOBS=4                    # do not spawn 128 compile jobs
cd /workspace/tools/graspnet/pointnet2 && conda run -n graspnet python setup.py install
cd /workspace/tools/graspnet/knn      && conda run -n graspnet python setup.py install
```

cu117 torch + cu118-built extensions are ABI-compatible within CUDA 11.x.

## 4. Import bootstrap (the fork mixes import styles)

The fork mixes package-relative imports (`models/*`, `dataset/*` use
`..pointnet2`, `..utils`) with flat absolute imports (`pointnet2/*`,
`utils/label_generation.py`, `pytorch_utils.py` use `import pointnet2_utils`
etc.). Our scripts handle this by:

1. adding `graspnet/pointnet2`, `graspnet/utils`, `graspnet/knn`,
   `graspnet/dataset` to `sys.path` (for the flat imports), and
2. registering `graspnet` as a namespace package pointing at the clone, which
   skips the fork's `graspnet/__init__.py` (it eagerly imports
   `graspnet_baseline` -> open3d/graspnetAPI, which we do not need).

This pattern is implemented in `claude_test/graspnet_ddp_smoke.py` and in the
vendored `graspnet/train.py`.

## 5. Vendored train entrypoint

The fork ships the upgraded model + extensions but no `train.py`. We adapt the
original baseline's `train.py` into `graspnet/train.py` (gitignored, vendored).
Changes vs the original:

- the import bootstrap above,
- `--num_workers` is configurable (the original hardcoded 4),
- paper-faithful defaults: `--batch_size 4`, `--lr_decay_steps 60,100`,
  `--lr_decay_rates 0.1,0.1` (lr 1e-3 -> 1e-4 @60 -> 1e-5 @100),
- `--dry_run` builds the model + optimizer without any dataset.

## 6. Verification (no dataset required)

```
# Extensions import + sm_86 kernels + DataParallel/DDP smoke test:
conda run -n graspnet python claude_test/graspnet_ddp_smoke.py
# Train entrypoint builds model+optimizer without data:
CUDA_VISIBLE_DEVICES=0 conda run -n graspnet python graspnet/train.py --dry_run
```

Both pass on 3x RTX 3090. Actual training needs the dataset
(see `docs/dataset_download.md`) and is launched via
`scripts/train_graspnet.sh`.
