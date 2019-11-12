from swspt.helpers import remapper_gen, progress_bar, load_source_frame, out_frame_name
import numpy as np
from functools import lru_cache
import util
import os
import cv2 as cv2


def block_name(ids, temp_dir):
    ident = '_'.join(map(str, ids))
    return util.get_file_name(ident, temp_dir, 'block', 'png')


def batched_transformer(out_movie, width, height, count, temp_dir, batch_size):
    image_loader = lru_cache(maxsize=batch_size)(load_source_frame)

    new_times = list(remapper_gen(count, width))
    new_xs = list(remapper_gen(width, count))

    new_times_chunked = list(util.chunks(new_times, batch_size))
    new_xs_chunked = list(util.chunks(new_xs, batch_size))

    # util.sep()
    # print(new_times_chunked)
    # util.sep('.')
    # print(new_xs_chunked)
    # util.sep()

    blockn = 0
    # новый кадр какой из старых координат х соотвествует (new_t <= old_x)
    for i, new_times_batch in enumerate(progress_bar(new_times_chunked, unit='batch')):
        # новая координата х из какого старого кадра берется (new_x <= old_t)
        for j, new_xs_batch in enumerate(new_xs_chunked):
            old_frames = np.stack([image_loader(old_frame, temp_dir) for new_x, old_frame in new_xs_batch])
            # old_frames has shape (batch_size, height, width, 3)  (full width!)

            block_width = old_frames.shape[0]
            new_blocks = np.zeros((block_width, height, block_width, 3))

            old_t_batch_start = i * batch_size
            new_blocks[:] = old_frames[:, :, slice(old_t_batch_start, old_t_batch_start + block_width), :]

            print(new_blocks.shape)

            for block_i, block in enumerate(new_blocks):
                name = block_name((j, block_i), temp_dir)
                cv2.imwrite(name, block)
                blockn += 1

    print('max i =', len(new_times_chunked))
    print('max j =', len(new_xs_chunked))
    print('max block_i = ', batch_size)

    exit()

    blocks = []
    for j, new_xs_batch in enumerate(new_xs_chunked):
        block_width = len(new_xs_batch)
        for block_i in range(block_width):
            block_img = cv2.imread(block_name((j, block_i), temp_dir))
            blocks.append(block_img)

        full_frame = np.hstack(tuple(blocks))
        out_movie.write(full_frame)
