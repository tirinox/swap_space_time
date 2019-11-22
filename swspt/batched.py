from swspt.helpers import remapper_gen, progress_bar, load_source_frame, out_frame_name
import numpy as np
from functools import lru_cache
import util
import os
from collections import Counter
import cv2 as cv2
import shutil


def block_name(ids, temp_dir):
    # fixme: debug stuff!!
    temp_dir = 'example/t'
    # if os.path.isdir(temp_dir):
    #     shutil.rmtree(temp_dir)
    # os.makedirs(temp_dir, exist_ok=True)

    ident = '_'.join(map(str, ids))
    return util.get_file_name(ident, temp_dir, 'block', 'png')


def batched_transformer(out_movie, width, height, count, temp_dir, batch_size, loader=load_source_frame):
    image_loader = lru_cache(maxsize=batch_size)(loader)

    new_times = list(remapper_gen(count, width))
    new_xs = list(remapper_gen(width, count))

    new_times_chunked = list(util.chunks(new_times, batch_size))
    new_xs_chunked = list(util.chunks(new_xs, batch_size))

    # new_times_batch - новый кадр какой из старых координат х соотвествует (new_t <= old_x)
    # new_xs_batch - новая координата х из какого старого кадра берется (new_x <= old_t)

    for j, new_xs_batch in enumerate(progress_bar(new_xs_chunked)):
        old_frames = np.stack([image_loader(old_frame, temp_dir) for new_x, old_frame in new_xs_batch])

        for i, new_times_batch in enumerate(new_times_chunked):
            # old_frames has shape (batch_size, height, width, 3)  (full width!)

            block_width = len(new_xs_batch)
            block_duration = len(new_times_batch)
            new_blocks = np.zeros((block_width, height, block_duration, 3))

            # все нужные старые координаты (они могут быть прорежены!)
            old_xs = [old_x for new_t, old_x in new_times_batch]

            new_blocks[:] = old_frames[:, :, old_xs, :]

            new_blocks = np.swapaxes(new_blocks, 0, 2)

            for k in range(block_duration):
                name = block_name((i, j, k), temp_dir)
                cv2.imwrite(name, new_blocks[k, :, :, :])

    total_frames = 0
    for i, new_times_batch in enumerate(new_times_chunked):
        for k, _ in enumerate(new_times_batch):
            blocks = []
            for j, new_xs_batch in enumerate(new_xs_chunked):
                name = block_name((i, j, k), temp_dir)
                block = cv2.imread(name)
                blocks.append(block)

            full_frame = np.hstack(tuple(blocks))
            out_movie(full_frame)

            total_frames += 1

    print(f'{total_frames} frame written')
