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

