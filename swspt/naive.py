from swspt.helpers import remapper_gen, progress_bar, load_source_frame
import numpy as np


def naive_transformer(out_movie, width, height, count, temp_dir, batch_size):
    new_times = list(remapper_gen(count, width))

    for _, new_time in progress_bar(new_times, unit='frame'):
        # each is new frame:
        new_frame = np.ndarray((height, width, 3), dtype=np.uint8)
        for x, frame in remapper_gen(width, count):
            old_frame = load_source_frame(frame, temp_dir)

            new_frame[:, x, :] = old_frame[:, new_time, :]
        out_movie.write(new_frame)
