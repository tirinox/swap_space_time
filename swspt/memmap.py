from functools import lru_cache
from swspt.helpers import load_source_frame, out_frame_name, progress_bar, remapper_gen
import numpy as np


def mmap_transformer(writer, width, height, count, temp_dir, batch_size):
    image_loader = lru_cache(maxsize=batch_size)(load_source_frame)

    def make_new_np_frame_mmapped(index):
        return np.memmap(out_frame_name(index, temp_dir), dtype=np.uint8, mode='w+', shape=(height, width, 3))

    print('Mapping files...')
    mb = height * width * 3 * count / (2**30)
    print(f'It will take {mb:.2f} GB')

    descriptors = {}
    for _, dest_frame_id in progress_bar(list(remapper_gen(count, width))):
        if dest_frame_id not in descriptors:
            descriptors[dest_frame_id] = make_new_np_frame_mmapped(dest_frame_id)

    print('Doing transformation...')

    for x, source_frame_id in progress_bar(list(remapper_gen(width, count))):
        source_frame = image_loader(source_frame_id, temp_dir)
        for _, dest_frame_id in remapper_gen(count, width):
            new_frame = descriptors[dest_frame_id]
            new_frame[:, x, :] = source_frame[:, dest_frame_id, :]

    print('Coding output movie...')

    for _, dest_frame_id in progress_bar(list(remapper_gen(count, width))):
        writer.write(descriptors[dest_frame_id])

    print('Cleaning up trash...')
