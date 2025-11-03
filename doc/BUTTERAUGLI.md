# Butteraugli

## Overview of Butteraugli

Butteraugli is a metric made to measure the estimated distance you would need to be to a screen to not perceive the difference between reference and distorded.
Its correlation to MOS is mainly interesting at Very high quality since it is extremely relevant in a Supra Threshold setting (lower quality).
This metric tends to be used since it has very few false negative: if Butteraugli says your frame is of high quality, you can rest assured that it is of high quality. However, it tends to have a non negligible amount of false positive, when it sees issues that a typical human would not see.

## Arguments

Name | Type | Required | Default
--- | --- | --- | ---
reference | `vapoursynth.VideoNode` | Yes
distorted | `vapoursynth.VideoNode` | Yes
qnorm | `int` | No | `2`
intensity_multiplier | `float` | No | `203`
distmap | `int` (`0`-`1`) | No | `0`
numStream | `int` | No | core.num_threads
gpu_id | `int` | No | `0`

### reference

Reference clip to compare distorted clip to. Must be a [Vapoursynth VideoNode][vs-videonode].

### distorted

Distorted clip to compare to reference clip with. It must be a [Vapoursynth VideoNode][vs-videonode] with the same length, width, and height as the reference clip.

### qnorm

Instead of retrieving norm2, you can choose to retrieve any norm you wish on the distortion map specified by this integer

### intensity_multiplier

Luminosity of the target screen in nits.

### distmap

When enabled (`1`), method returns the distmap. Otherwise, the method returns the distorted VideoNode. The distmap can be used for additional processing not offered by VSHIP. Below is an example where the method returns the distmap.

```py
# reference, distorted, and distmap are vapoursynth.VideoNode instances
distmap = reference.vship.BUTTERAUGLI(distorted, distmap = 1)
```

`distmap` is a [Vapoursynth VideoNode][vs-videonode] with the following properties:

```
VideoNode
        Format: GrayS
        Width: reference.width
        Height: reference.height
        Num Frames: len(reference)
        FPS: reference_fps
```

`distmap` will also contain the [Butteraugli return values][wiki-butteraugli-return-values].

### numStream

It corresponds to the number of concurrent frames running through the GPU at a time. Lowering it allows to control and lower the amount of VRAM consumed. By default, as much stream are created as there are vapoursynth threads. (which is the maximum amount, any higher will get lowered to this value)

### gpu_id

ID of the GPU to run VSHIP on. It will perform the GPU Check functions as described in the [Error Management][wiki-error-management] page.

## Butteraugli Return Values

The method will always return a [Vapoursynth VideoNode][vs-videonode] with the following properties on each individual [VideoFrame][vs-videoframe]: `_BUTTERAUGLI_QNorm`, `_BUTTERAUGLI_3Norm`, `_BUTTERAUGLI_INFNorm`. These values return regardless of the [`distmap`][wiki-distmap] argument value provided.

Name | Type | Description
--- | --- | ---
`_BUTTERAUGLI_QNorm` | `float` |  Norm configurable with argument of the distmap
`_BUTTERAUGLI_2Norm` | `float` |  equal to the above, deprecated
`_BUTTERAUGLI_3Norm` | `float` | Norm 3 of the distmap
`_BUTTERAUGLI_INFNorm` | `float` | Maximum value of the distmap

## Performance Discussion

### VRAM Consumption

VRAM consumption can be calculated using the following: `31 * 4 * width * height` where width and height refer to the dimensions of the video. Bytes per Vapoursynth thread: `Plane Buffer * sizeof(float) * width * height`.

[wiki-error-management]: Vship-Error-Managment.md

[vs-videonode]: https://www.vapoursynth.com/doc/pythonreference.html#VideoNode
[vs-videoframe]: https://www.vapoursynth.com/doc/pythonreference.html#VideoFrame