# CVVDP

## Overview of CVVDP

CVVDP is a metric that, like Butteraugli works mainly at the sensitivity level, trying to estimate the probability that a human would notice a difference between 2 images.
CVVDP is unlike the other 2 a Video Metric, meaning that it takes into account the temporal behavior of a video. 
It has been shown to be a very good metric on various independant benchmarks.

## Arguments

Name | Type | Required | Default
--- | --- | --- | ---
reference | `vapoursynth.VideoNode` | Yes
distorted | `vapoursynth.VideoNode` | Yes
model_name | `str` | No | `standard_fhd`
model_config_json | `str` | No | ""
resizeToDisplay | `int` | No | `0`
distmap | `int` | No | `0`
gpu_id | `int` | No | `0`

### reference

Reference clip to compare distorted clip to. Must be a [VapourSynth VideoNode][vs-videonode].

### distorted

Distorted clip to compare to reference clip with. It must be a [VapourSynth VideoNode][vs-videonode] with the same length, width, and height as the reference clip.

### distmap

When enabled (`1`), method returns the distmap. Otherwise, the method returns the distorted VideoNode. The distmap can be used for additional processing not offered by VSHIP. Below is an example where the method returns the distmap.

```py
# reference, distorted, and distmap are vapoursynth.VideoNode instances
distmap = reference.vship.CVVDP(distorted, distmap = 1)
```

`distmap` is a [VapourSynth VideoNode][vs-videonode] with the following properties:

```
VideoNode
        Format: GrayS
        Width: resized_width
        Height: resized_height
        Num Frames: len(reference)
        FPS: reference_fps
```

`distmap` will also contain the [CVVDP return values]

### model_name

Argument used for CVVDP, it allows to specify a display defined in the [CVVDP configuration](../src/cvvdp/display_models.hpp).

The list of possible model is here:
- standard_4k
- standard_fhd
- standard_hdr_pq
- standard_hdr_hlg
- standard_hdr_dark
- standard_hdr_linear_zoom

Each Display has its own resolution. An accurate result of the rescaled content to the specified screen can be obtained setting resizeToDisplay to 1 too.

### model_config_json

Allows to specify a path to a config file written the same way as the original CVVDP repo. Here is the format supported by vship:

```
{
        "new_Display_Name" : {
                "name": "string",
                "colorspace": "string that is either HDR, sRGB/SDR or CVVDP stuff: BT.709-linear, BT.2020-PQ, BT.2020-HLG",
                "resolution": [width_int, height_int],
                "max_luminance": float,
                "viewing_distance_meters": float,
                "diagonal_size_inches": float,
                "contrast": float,
                "E_ambient": float,
                "k_refl": float,
                "exposure": float,
                "source": "string",
        },
        "standard_fhd" : {
                "name": "overwrite default colorspace for standard_fhd display",
                "colorspace": "HDR",
        }
}
```

if the display name is new, then all fields except name, source, k_refl and exposure will have to be specified (k_refl and exposure have defaults)
if the display name corresponds to another display defined earlier in the config file or in the basic displays, the new values will overwrite older ones and may not require to be all set as such.

### gpu_id

ID of the GPU to run VSHIP on. It will perform the GPU Check functions as described in the [Error Management][wiki-error-management] page.

## CVVDP Return Values

The method will always return a [VapourSynth VideoNode][vs-videonode] with the following property on each individual [VideoFrame][vs-videoframe]: `_CVVDP`

As CVVDP is a video metric, it does its own score accumulation to return a single score.
As such, ONLY THE LAST SCORE SHOULD BE USED but every score should be computed! In fact, the score of frame i corresponds to the score of the sequence from frame 0 to frame i.

### VRAM Consumption

VRAM consumption can be calculated using the following: `4 * width * height * (10*4/3 + fps*3*0.5)` where width and height refer to the dimensions of the video and fps to its frames per second. This formula is an approximation.

[wiki-error-management]: Vship-Error-Managment.md
[vs-videonode]: https://www.vapoursynth.com/doc/pythonreference.html#VideoNode
[vs-videoframe]: https://www.vapoursynth.com/doc/pythonreference.html#VideoFrame