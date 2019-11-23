### Swap space and time

Video is 3D volume of X-dimension, Y-dimension and Time-dimension (X, Y, T).
So the idea is like photo-finish: swap X and T dimensions.
For example frame #25 includes all states of #25 column of the original video over the time.
The most left column is now the state of #25 column at time = 0, and the most right is at the most latest time.
  
Requirements:
1. opencv-python
2. tqdm

`pip install -r requirements.txt`

How to run:

`python3 swap_space_time.py --input example/input.mp4 --output example/output.mp4`

## Visual example:

Let's consider we have a video like this:

![example/input.gif](example/input.gif)

We swap X and Time:

![example/output.gif](example/output.gif)

## Command line options

`--input <filepaph>` - path to the input video

`--output <filepath>` – path to the output video

`--codec <codec>` – output codec, default one is "MP4V"

`--jpeg-quality <quality>` – JPEG quality (20 to 100) when decomposing video to frames

`--algo <algorithm>` – pick one of 3 algorithms: "naive", "mmap" or "batched" (default and the most efficient).

`--batch <n>` – batch size for the batched algorithms. Good values are in 20-500 I guess. Default value is 60.

`--limit <n>` – limit input frame count from the beginning of the video. Default is -1 than means there is no limit.

`--write-frames` – this is a flag that tells the program to write output frames separately as PNG images instead of writing a video. 
It will create a folder named similar to the output video name but adding "-frames" suffix to it. By default this mode is off.