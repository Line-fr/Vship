# SSIMULACRA2

## Overview of SSIMULACRA2

SSIMULACRA2 is a colorspace aware metric which tries to correlate with MOS on a wide quality range using similar technics as VMAF or MSSIM.

## Arguments

Name | Type | Required | Default
--- | --- | --- | ---
reference | `vapoursynth.VideoNode` | Yes
distorted | `vapoursynth.VideoNode` | Yes
numStream | `int` | No | core.num_threads
gpu_id | `int` | No | `0`

### reference

Reference clip to compare distorted clip to. Must be a [Vapoursynth VideoNode][vs-videonode].

### distorted

Distorted clip to compare to reference clip with. It must be a [Vapoursynth VideoNode][vs-videonode] with the same length, width, and height as the reference clip.

### numStream

It corresponds to the number of concurrent frames running through the GPU at a time. Lowering it allows to control and lower the amount of VRAM consumed. By default, as much stream are created as there are vapoursynth threads. (which is the maximum amount, any higher will get lowered to this value)

### gpu_id

ID of the GPU to run VSHIP on. It will perform the GPU Check functions as described in the [Error Management][wiki-error-management] page.

## SSIMULACRA2 Return Values

The method will always return a [Vapoursynth VideoNode][vs-videonode] with the following property on each individual [VideoFrame][vs-videoframe]: `_SSIMULACRA2`

### VRAM Consumption

VRAM consumption can be calculated using the following: `9 * 4 * width * height * 4/3` where width and height refer to the dimensions of the video. Bytes per Vapoursynth thread: `Plane Buffer * sizeof(float) * width * height * 4/3`.

[wiki-error-management]: Vship-Error-Managment.md