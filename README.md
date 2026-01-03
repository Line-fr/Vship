# Vship : Fast Metric Computation on GPU

An easy to use high-performance Library for GPU-accelerated visual fidelity
metrics with SSIMULACRA2, Butteraugli & CVVDP.

## Overview

`vship` provides hardware-accelerated implementations of:

- **SSIMULACRA2**: A perceptual image quality metric from Cloudinary's
  SSIMULACRA2
- **Butteraugli**: Google's psychovisual image difference metric from libjxl
- **CVVDP**: University of Cambridge's psychovisual video quality metric

The plugin uses HIP/CUDA for GPU acceleration, providing significant performance
improvements over CPU implementations. It can be used with a simple binary (FFVship), as a vapoursynth plugin and has a C API.

There are precompiled binaries ready to be used in the release section.

## Projects Featuring Vship

If you want to use Vship with a pre-defined workflow, here are some projects
featuring Vship:
- [Av1an](https://github.com/rust-av/Av1an): A Cross-platform command-line AV1 / VP9 / HEVC / H264 encoding framework with per scene quality encoding 
- [xav](https://github.com/emrakyz/xav): A simpler alternative to Av1an dedicated to Target Quality Encoding trying to be as fast as possible.
- [SSIMULACRApy](https://codeberg.org/Kosaka/ssimulacrapy): A Python script to
  compare videos and output their SSIMU2 scores using various metrics (by
  [Kosaka](https://codeberg.org/Kosaka))
- [`metrics`](https://github.com/psy-ex/metrics): A perceptual video metrics
  toolkit for video encoder developers (by the
  [Psychovisual Experts Group](https://github.com/psy-ex/metrics))
- [`vs_align`](https://github.com/pifroggi/vs_align): A vapoursynth plugin
  measuring temporal offset between two videos
- [`chunknorris`](https://github.com/Boulder08/chunknorris): A python script
  to adjust the quality encoding parameters at each scene of a video base on objective metrics
- [Media-Metrologist](https://github.com/Av1ation-Association/Media-Metrologist): Media-Metrologist is a library for measuring video quality using a suite of metrics on a per-scene and per-frame basis.
- [`lvsfunc`](https://github.com/Jaded-Encoding-Thaumaturgy/lvsfunc): JET project containing various functions to help with video processing.

## Installation

The steps to build `vship` from source are provided below.
See [Compilation and Usage](./doc/VshipAPI.md#compilation-and-usage) for more details.
For compiling on Windows, see [FFVship Windows Compilation](./doc/FFVship-Windows-Compilation.md).

### Dependencies
For all build options the following are required:

- `make`
- `hipcc` (AMD HIP SDK) or `nvcc` (NVIDIA CUDA SDK)

Additionally, to build the FFvship cli tool:

- [ffms2](https://github.com/FFMS/ffms2) (and libavutil header to compile)
- [pkg-config](https://gitlab.freedesktop.org/pkg-config/pkg-config)

### Build Instructions

1. Use the appropriate target for your gpu or use case.

```bash
#libvship Build
make buildcuda     # Build for the current systems Nvidia gpu
make buildcudaall  # Build for all supported Nvidia gpus
make build         # Build for the current systems AMD gpu
make buildall      # Build for all supported AMD gpus

#FFVship CLI linux tool build (requires libvship built before FFVSHIP)
make buildFFVSHIP
```

2. Install libvship and eventually the FFVship executable.

The `install` target automatically detects and installs only the components that were built.
```bash
make install
#for arch, you need to use another prefix:
make install PREFIX=/usr
```

## Library Usage

### FFVship

To control the performance-to-VRAM trade-off, set the `-g` argument in FFVship to control the
number of GPU threads to use. You can also control the number of decoder threads with `-t`.
I recommend 4 GPU threads as a good compromise between performance and VRAM usage.

This contains only some of the numerous options offered by Vship
I recommend checking the doc or using -h to get the full list.
```
usage: ./FFVship [-h] [--source SOURCE] [--encoded ENCODED]
                    [-m {SSIMULACRA2, Butteraugli, CVVDP}]
                    [--start start] [--end end] [-e --every every]
                    [-t THREADS] [-g gpuThreads] [--gpu-id gpu_id]
                    [--json OUTPUT]
                    [--list-gpu]
```

### Vapoursynth

To control the performance-to-VRAM trade-off, set the `numStream` argument to control
how many GPU threads to use. I recommend 4 as a good compromise between both.

### SSIMULACRA2

See [SSIMULACRA2](./doc/SSIMULACRA2.md) for details like calculating VRAM usage.

```python
import vapoursynth as vs
core = vs.core

# Load reference and distorted clips
ref = core.bs.VideoSource("reference.mp4")
dist = core.bs.VideoSource("distorted.mp4")

# Calculate SSIMULACRA2 scores
#numStream controls the performance-to-VRAM trade-off
result = ref.vship.SSIMULACRA2(dist, numStream = 4)

# Extract scores from frame properties
scores = [frame.props["_SSIMULACRA2"] for frame in result.frames()]

# Print average score
print(f"Average SSIMULACRA2 score: {sum(scores) / len(scores)}")
```

### Butteraugli

See [BUTTERAUGLI](./doc/BUTTERAUGLI.md) for more details like calculating VRAM usage.

```python
import vapoursynth as vs
core = vs.core

# Load reference and distorted clips
ref = core.bs.VideoSource("reference.mp4")
dist = core.bs.VideoSource("distorted.mp4")

# Calculate Butteraugli scores
# distmap controls whether to return a visual distortion map or the reference clip
# intensity_multiplier controls sensitivity
# qnorm controls which Norm to calculate and return in _BUTTERAUGLI_QNorm
result = ref.vship.BUTTERAUGLI(dist, distmap = 0, numStream = 4, qnorm = 5)

# Extract scores from frame properties (three different norms available)
scores_3norm = [frame.props["_BUTTERAUGLI_3Norm"] for frame in result.frames()]
scores_infnorm = [frame.props["_BUTTERAUGLI_INFNorm"] for frame in result.frames()]
scores_5norm = [frame.props["_BUTTERAUGLI_QNorm"] for frame in result.frames()]

# Alternatively get all scores in one pass
all_scores = [frame.props["_BUTTERAUGLI_3Norm"],
               frame.props["_BUTTERAUGLI_INFNorm"],
               frame.props["_BUTTERAUGLI_QNorm"]]
              for frame in result.frames()]

# Print average scores
print(f"Average Butteraugli 3Norm distance: {sum(scores_3norm) / len(scores_3norm)})
print(f"Average Butteraugli MaxNorm distance: {sum(scores_infnorm) / len(scores_infnorm)})
print(f"Average Butteraugli 5Norm distance: {sum(scores_5norm) / len(scores_5norm)})

# Output grayscale visualation of distortion for visual analysis
ref.vship.BUTTERAUGLI(dist, distmap = 1, numStream = 4).set_output()
```

### CVVDP

See [CVVDP](./doc/CVVDP.md) for more details like calculating VRAM usage.

```python
import vapoursynth as vs
core = vs.core

# Load reference and distorted clips
ref = core.bs.VideoSource("reference.mp4")
dist = core.bs.VideoSource("distorted.mp4")

# Calculate CVVDP scores
# distmap controls whether to return a visual distortion map or the reference clip
# model_name controls which Display Model to use
# resizeToDisplay conrols whether or not to resize the reference and distorted inputs to the Display Model resolution
result = ref.vship.CVVDP(dist, distmap = 0, model_name = "standard_fhd", resizeToDisplay = 0)

# Extract scores from frame properties
scores = [frame.props["_CVVDP"] for frame in result.frames()]

# Only use the last score of CVVDP. (it takes into account every frame that it has seen up to now)
#it is different because it is an actually temporal metric unlike others
print(f"CVVDP Video Score: {scores[-1]}")

# Output grayscale visualation of distortion for visual analysis
ref.vship.CVVDP(dist, distmap = 1).set_output()
```

## Performance

Testing Hardware: Ryzen 7940HS + RTX 4050 Mobile (strong CPU - weak GPU configuration)
Testing Clip: 1080p 1339 frames

SSIMU2 Implementation | HW Type | Time
--- | --- | ---
JXL | CPU | 115s
VSZIP | CPU | 68s
Vapoursynth Vship | GPU | 25s
FFVship | GPU | 9s

Butteraugli Implementation | HW Type | Time
--- | --- | ---
JXL | CPU | 239s
Vapoursynth Vship | GPU | 47s
FFVship | GPU | 37s

CVVDP Implementation | HW Type | Score | Time
--- | --- | --- | ---
Original Repo | GPU | 9.4808 | 162s
FFVship | GPU | 9.52268 | 22s


`vship` dramatically outperforms CPU-based and GPU-based implementations of these metrics
while preserving a high degree of accuracy.

## References

- Butteraugli Source Code:
  [libjxl/libjxl](https://github.com/libjxl/libjxl/tree/main/lib/jxl/butteraugli)
- SSIMULACRA2 Source Code:
  [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2)
- CVVDP Source Code: 
  [gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP/tree/main?tab=readme-ov-file)

## Credits

Special thanks to dnjulek for the Zig-based SSIMULACRA2 implementation in
[vszip](https://github.com/dnjulek/vapoursynth-zip).

## License

This project is licensed under the MIT license. License information is provided
by the [LICENSE](LICENSE).
