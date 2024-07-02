# Installation Guide

Ensure you are in a Python environment with the following version:
```
python==3.11.0
```

Execute the installation script:
```
bash scripts/install.sh
```

## Possible Issues

When running `scripts/install.sh`, you may encounter an error during the installation of Mamba's `selective_scan` package. This can happen due to a version mismatch between `nvcc` and the CUDA version displayed by `nvidia-smi`. Below are steps to ensure that the CUDA versions match:

### Ensuring CUDA Version Consistency

Before installing the `selective_scan` package, it is crucial to confirm that your CUDA environment is correctly configured. Make sure that the version of `nvcc` (CUDA compiler) matches the CUDA driver version shown by `nvidia-smi`. Version inconsistencies can lead to compilation errors or runtime failures. Follow these steps to check and configure your CUDA environment:

1. **Check CUDA Driver Version**:
   Open a terminal and run the following command to view the CUDA driver version:
   ```
   nvidia-smi
   ```
   Note the CUDA version displayed on the screen.

2. **Check NVCC Version**:
   In the terminal, run the following command to check the version of `nvcc`:
   ```
   nvcc --version
   ```
   Ensure that this version matches the version shown by `nvidia-smi`.

3. **Configure Environment Variables**:
   If you find a version mismatch, you may need to adjust `CUDA_HOME` and other related environment variables. This usually means specifying the correct path to the version of CUDA you wish to use. For example, if you want to use CUDA 11.7, set:
   ```
   export CUDA_HOME=/usr/local/cuda-11.7
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```
   Replace `/usr/local/cuda-11.7` with your actual CUDA installation path.

4. **Re-validate Environment Settings**:
   After setting the environment variables, reopen a new terminal window and re-run the above `nvidia-smi` and `nvcc --version` commands to confirm the settings are correct and the versions match.

By ensuring the consistency of CUDA driver and toolkit versions, you can significantly reduce the risk of runtime issues and installation failures, particularly for libraries that rely on specific CUDA features.
