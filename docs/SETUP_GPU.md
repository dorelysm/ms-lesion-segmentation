# GPU setup

## AMD ROCm on Windows (what this project actually used)

This project was trained on an **AMD Radeon RX 9060 XT** via AMD's official "PyTorch on Windows"
ROCm 7.2.1 build, which preserves the standard `torch.cuda` API — `src/train.py` and
`src/evaluate.py` need no code changes to use the GPU, they already do
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

### Requirements

- **Python 3.12** exactly (the ROCm wheels are built for `cp312`; other Python versions won't work).
- **AMD graphics driver 26.2.2+** (check via Windows Settings → About, or the AMD Software / Adrenalin app).
- **Visual Studio Build Tools**, "Desktop development with C++" workload + a Windows SDK — MIOpen's
  HIPRTC JIT kernel compiler needs the MSVC standard library headers on Windows. Without this, GPU ops
  like `BatchNorm` fail with `miopenStatusUnknownError` / `'type_traits' file not found`. Install via:
  ```
  # download https://aka.ms/vs/17/release/vs_buildtools.exe, then:
  vs_buildtools.exe --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
  ```

### Environment setup

```powershell
py -3.12 -m venv .venv-rocm
.venv-rocm\Scripts\activate        # Windows

# ROCm SDK (~1.4 GB)
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

# PyTorch/torchvision/torchaudio with ROCm support (~823 MB) -- do this BEFORE `pip install -r requirements.txt`,
# since the plain `torch`/`torchvision` entries there would otherwise install CPU-only wheels over these.
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

pip install -r requirements.txt
```

### Verify

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True, AMD Radeon RX 9060 XT
```

Other AMD Windows GPUs (RX 7000/9000 series) supported by the same ROCm 7.2.1 build should work the
same way — check AMD's [Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
for your card.

## NVIDIA (CUDA)

On a machine with an NVIDIA GPU, skip the section above entirely:

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or the current CUDA index
pip install -r requirements.txt
```

The rest of the pipeline (`src/`, `configs/`) is identical either way — the device selection is
automatic.

## No GPU

Everything still runs on CPU; `--smoke-test` (see main README) is fast enough for that. A full
5-fold run on the full dataset is only practical with a GPU.
