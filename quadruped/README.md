# quadruped

RL for the Unitree Go2 (walking → jumping → backflip) using Isaac Lab, with sim2sim in MuJoCo and deployment on the real robot.

## Layout

```
quadruped/
├── third_party/
│   └── IsaacLab/      # Isaac Lab v3.0.0 source checkout (git-ignored, see below)
└── ...                # our own task package and scripts
```

Isaac Lab is **not** committed to this repo. It is cloned into `third_party/IsaacLab` and installed in editable mode, so the code you read there is exactly the code that runs, and "Go to Definition" in the editor lands in that checkout.

## Tested setup

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 (x86_64) |
| GPU / driver | RTX 5090, NVIDIA 595.99.02 (Isaac Lab 3.0 requires ≥ 580.65.06) |
| Python | 3.12 (conda env `env_isaaclab`) |
| Isaac Sim | 6.2.0 (pip) |
| PyTorch | 2.11.0 + cu130 |
| Isaac Lab | v3.0.0 |

## Setup

### 1. System packages

```bash
sudo apt install cmake build-essential
```

### 2. Conda environment with Isaac Sim and PyTorch

```bash
conda create -n env_isaaclab python=3.12
conda activate env_isaaclab
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

Check that PyTorch sees the GPU and supports Blackwell (`sm_120`):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), 'sm_120' in torch.cuda.get_arch_list())"
# expected: 2.11.0+cu130 True True
```

### 3. (Machines with an integrated GPU) Use only the NVIDIA Vulkan driver

If the CPU has an integrated GPU (e.g. AMD Ryzen 9000 series), Isaac Sim may enumerate it as a second render device. To make the env only see the NVIDIA driver, add conda activation hooks:

```bash
E=$CONDA_PREFIX/etc/conda
mkdir -p $E/activate.d $E/deactivate.d
cat > $E/activate.d/zz_vulkan_nvidia_only.sh <<'EOF'
export _OLD_VK_DRIVER_FILES="${VK_DRIVER_FILES-}"
export _OLD_VK_ICD_FILENAMES="${VK_ICD_FILENAMES-}"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
EOF
cat > $E/deactivate.d/zz_vulkan_nvidia_only.sh <<'EOF'
export VK_DRIVER_FILES="${_OLD_VK_DRIVER_FILES}"; [ -z "$VK_DRIVER_FILES" ] && unset VK_DRIVER_FILES
export VK_ICD_FILENAMES="${_OLD_VK_ICD_FILENAMES}"; [ -z "$VK_ICD_FILENAMES" ] && unset VK_ICD_FILENAMES
unset _OLD_VK_DRIVER_FILES _OLD_VK_ICD_FILENAMES
EOF
conda deactivate && conda activate env_isaaclab
```

Check the path with `ls /usr/share/vulkan/icd.d/` first. In the Isaac Sim startup log, the GPU table should then list only the NVIDIA card.

### 4. Clone and install Isaac Lab

From the repo root:

```bash
git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git quadruped/third_party/IsaacLab
cd quadruped/third_party/IsaacLab
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer'
```

This installs the core Isaac Lab packages in editable mode, plus the Newton physics backend, the `rsl_rl` library and the visualizers. The other RL libraries (skrl, sb3, rl_games) are skipped.

### 5. Smoke test

```bash
cd quadruped/third_party/IsaacLab
isaacsim                                                               # Isaac Sim GUI on its own
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py              # headless (default)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit    # with the Kit viewer window
```

The first launch accepts the EULA (type `Yes`, or set `OMNI_KIT_ACCEPT_EULA=YES`) and compiles shaders, which can take several minutes. During that time the window may look frozen. If GNOME shows "not responding", click **Wait**.

## Usage

The Go2 velocity tasks in Isaac Lab 3.0 are named `Isaac-Velocity-Flat-UnitreeGo2` and `Isaac-Velocity-Rough-UnitreeGo2`. The 2.x names with a `-v0` suffix no longer exist. Training goes through the `isaaclab` CLI:

```bash
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 64 --viz kit
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 4096
```

- Runs are headless by default. Add `--viz kit` to open a viewer. The 2.x `--headless` flag is gone.
- The physics backend is chosen with a Hydra-style override: `physics=newton_mjwarp` (Newton, the default) or `physics=isaacsim_physx` (PhysX via Isaac Sim).
- Logs and checkpoints go to `logs/` under the current working directory, which is git-ignored.

### Where to read the code

| What | Path (under `third_party/IsaacLab/source/`) |
|---|---|
| Go2 env + PPO configs | `isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/` |
| Shared velocity-task MDP (obs, rewards, terminations) | `isaaclab_tasks/isaaclab_tasks/core/velocity/` |
| Go2 robot / actuator config | `isaaclab_assets/` |
| RL wrapper for rsl_rl | `isaaclab_rl/` |

## Known issues

- **Isaac Sim 5.1 crashes with NVIDIA driver 595.** It segfaults in `librtx.scenedb.plugin.so` right after `app ready`, in both GUI and headless mode. NVIDIA's fix is to upgrade to Isaac Sim 6.x ([IsaacSim#677](https://github.com/isaac-sim/IsaacSim/issues/677), [#687](https://github.com/isaac-sim/IsaacSim/issues/687)). That is why this setup uses Isaac Sim 6.2 / Isaac Lab 3.0 instead of 5.1 / 2.3.
- `pip check` reports minor pin conflicts between `isaacsim-kernel` / `isaacsim-core` and packages pulled in by Isaac Lab (`click`, `psutil`, `urllib3`, `llvmlite`). Both smoke tests above still pass.
- Over plain SSH there is no display. Run headless, or use `--video` to record clips.
