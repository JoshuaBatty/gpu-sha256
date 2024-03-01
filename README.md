# gpu-sha256
sha256 running in a compute shader with wgpu

sha256 code is from https://github.com/MarcoCiaramella/sha256-gpu

running the example should produce something similar to the following output:

```
Total messages to hash: 2000000

Rust sequential sha256 took: 653.687875ms
Rust parallel sha256 took: 212.407625ms

Dispatching 62500 workgroups with 32 threads each
GPU Compute Shader: 15.707541ms
```
