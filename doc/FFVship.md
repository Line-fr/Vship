# FFVship

## Usage

```
usage: ./FFVship [-h] [--source SOURCE] [--encoded ENCODED]
                    [-m {SSIMULACRA2, Butteraugli, CVVDP}]

                    [--start start] [--end end] [-e --every every]
                    [--encoded-offset offset]
                    [--source-indices indexList] [--encoded-indices indexList]

                    [-t THREADS] [-g gpuThreads] [--gpu-id gpu_id]
                    [--json OUTPUT]
                    [--list-gpu]

                    [--cropTopSource cropSize] [--cropTopEncoded cropSize]
                    [--cropBottomSource cropSize] [--cropBottomEncoded cropSize]
                    [--cropLeftSource cropSize] [--cropLeftEncoded cropSize]
                    [--cropRightSource cropSize] [--cropRightEncoded cropSize]

                    Specific to Butteraugli: 
                    [--intensity-target Intensity(nits)]
                    [--qnorm MinwoskiInteger]

                    Specific to CVVDP:
                    [--displayModel modelKey] [--resizeToDisplay]
```

## Arguments

Name | Type | Required | Default
--- | --- | --- | ---
--reference | `Video Path` | Yes
--encoded | `Video Path` | Yes
-m | `Metric` | No | SSIMULACRA2
--start | `Integer` | No | 0
--end | `Integer` | No | None
--every | `Integer` | No | 1
-t | `Integer` | No | 1
-g | `int` | No | 8
--gpu-id | `int` | No | 0
--json | `output path` | No | None
--list-gpu | None | No | None
--crop* | `int` | No | 0
--intensity-target | `float` | No | `203`
--displayModel | `str` | No | `standard_fhd`
--resizeToDisplay | None | No | None

## Details

### Reference / Encoded

You specify the video path of files you want to compare using FFVship. FFVship is currently unable to process images and will only work on YUV Video input (which accounts for almost any video in existence).

### start / end / every

When specified, FFVship will compute metrics starting from frame --start up to --end and computing metrics for one frame every --every frames

### encoded-offset

When source and encoded are not aligned, we may need to offset the computing window. How this work is that --start and --end are used to express which frames are going to be computed in the **Source**. encoded file is going to see --start+offset --end+offset encoded. Note that end and start are automatically truncated to match the length. Example:

```
--start 0 --end 5 --encoded-offset 1
Source_frames : 0, 1, 2, 3
Encoded_frames : 1, 2, 3, 4
```

Note that --start 0 --end 4 would have been equivalent there. It also accepts negative offsets

```
--start 0 --end 5 --encoded-offset -1
Source_frames : 1, 2, 3, 4
Encoded_frames : 0, 1, 2, 3
```

Note that --start 1 --end 5 would have given the same result here.

### --source-indices / --encoded-indices

Allows to specify specific frames to compute metrics on. It is affected by start/every/encoded_offset but end is not taken into account. --source-indices can be specified alone and will apply to both source and encoded but --encoded-indices cannot be specified alone.

example: --start 10 --every 2 --encoded-offset 1 --source-indices 13
compare: frame 10+2*13 = 36 for source and frame 10+2*13+1 = 37 for encoded

### --crop...

There are multiple crop functions that can be seen in -h to define the crop rectangle of both source and encoded

beware that any size mismatch will imply a resize! example:

my source is 1910x1080 and my encoded is 1920x1080

I crop my source at left by 10 to 1900x1080 and my encoded at left by 10 to 1910x1080
Then FFVship will resize the decode the encoded to 1920x1080 then resize to 1910x1080 then crop to 1900x1080.
(the step is: decode to normal size, resize to the right size such that once cropped, it matches source cropped size)

### -t / -g

-t corresponds to the number of decoding process used. It is recommended to increase this value to -t 2 or -t 3 if your CPU has too much cores for the decoding process to fully use it (if it not already saturating your GPU). -g correponds to the number of stream in the GPU, -g 3 is usually considered enough, it can be lowered to reduce VRAM or increased a bit to help the GPU parallelize frames better if it fails to being fully used.

### --gpu-id / --list-gpu

--gpu-id allows to choose a GPU on the system if it has multiple GPUs. A list of GPUs detected on the system and their corresponding IDs can be obtained by running `FFVship --list-gpu`

### --source-index / --encoded-index / --cache-index

--source-index and --encoded-index specify path to read or write index.
if only --cache-index is specified and not the path, it defaults to the path of the file + ".ffindex"

it will first try to read it and compute again if it fails.
it will then write the index to disk if it had to compute it again and --cache-index is specified.

### --live-score-output

This option allows to get the score in real time without any other parasitic output like the progress bar.
This is only meant for usage in external scripts wanting to implement their own progress bar.
The output in stdout possesses as a first line the number of frames to be computed on (which is also the number of lines that are going to come after this one).
Each next line is in format : {subjective_frame_index} {score0} {score1} ...
Frames can be given in the wrong order due to parallelism.

### --displayModel

Argument used for CVVDP, it allows to specify a display defined in the [CVVDP configuration](../src/cvvdp/display_models.hpp).

The list of possible model is here:
- standard_4k
- standard_fhd
- standard_hdr_pq
- standard_hdr_hlg
- standard_hdr_dark
- standard_hdr_linear_zoom

### --version

Returns the version of ffvship to check for functionalities and bugs as a dependency.

### --json

This method is made to be able to retrieve per-frame scores from the computation instead of simple statistics. The format of the output json file is as follow:

```
#SSIMULACRA2
[[score_frame_0], [score_frame_1], ...]

#Butteraugli
[[NormQ_score_frame_0, Norm3_score_frame_0, NormINF_score_frame_0], [NormQ_score_frame_1, Norm3_score_frame_1, NormINF_score_frame_1], ... ]

#CVVDP
[[score_frame_cumulated_0_to_0], [score_frame_cumulated_0_to_1], [score_frame_cumulated_0_to_2], ...]

(the last cvvdp score is the cvvdp score of the whole clip)
```