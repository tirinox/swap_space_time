import cv2
import os
import sys
import tempfile
from arg_parser import arg_parser, ArgFlag, ArgParameter
import util
import numpy as np
import tqdm
from functools import lru_cache
 

def get_input_frame_name(index, tmp_dir):
    return util.get_file_name(index, tmp_dir, prefix='in', ext='jpg')


def progress_bar(it, unit='frame'):
    return tqdm.tqdm(it, file=sys.stdout, unit=unit)


def split_video_to_frames(movie_input, tmp_path, jpeg_quality):
    cap = cv2.VideoCapture(movie_input)
    if not cap.isOpened():
        util.error('could not open input file')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Input frame count: {count} ({width} x {height}) @ {fps:.2f} fps')

    frame_counter = 0
    for _ in progress_bar(range(count)):
        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if ret:
            frame_name = get_input_frame_name(frame_counter, tmp_path)
            cv2.imwrite(frame_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

            frame_counter += 1
        else:
            break

    cap.release()

    print(f'Real frame count: {frame_counter}', flush=True)

    return width, height, fps, frame_counter


def load_source_frame(frame_index, tmp_path):
    return cv2.imread(get_input_frame_name(frame_index, tmp_path), cv2.IMREAD_COLOR)


def out_frame_name(index, tmp_path):
    return util.get_file_name(index, tmp_path, prefix='pf', ext='npy')


def remapper_gen(from_dim, to_dim):
    for i in range(from_dim):
        yield i, int(i * to_dim / from_dim)


def smart_transformer(writer, width, height, count, temp_dir, batch_size):
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

    print('Reading frames...')

    for x, source_frame_id in progress_bar(list(remapper_gen(width, count))):
        #  x, номер исходного кадра
        source_frame = image_loader(source_frame_id, temp_dir)
        for _, dest_frame_id in remapper_gen(count, width):
            new_frame = descriptors[dest_frame_id]
            new_frame[:, x, :] = source_frame[:, dest_frame_id, :]

    print('Coding output movie...')

    for _, dest_frame_id in progress_bar(list(remapper_gen(count, width))):
        writer.write(descriptors[dest_frame_id])

    print('Cleaning up trash...')


def naive_transformer(out_movie, width, height, count, temp_dir, batch_size):
    new_times = list(remapper_gen(count, width))

    for _, new_time in progress_bar(new_times, unit='frame'):
        # each is new frame:
        new_frame = np.ndarray((height, width, 3), dtype=np.uint8)
        for x, frame in remapper_gen(width, count):
            old_frame = load_source_frame(frame, temp_dir)

            new_frame[:, x, :] = old_frame[:, new_time, :]
        out_movie.write(new_frame)


def do_work(config: dict):
    jpeg_quality = int(config['jpeg-quality'])
    assert 20 <= jpeg_quality <= 100

    input_file = config['input']
    output_file = config['output']

    batch_size = int(config['batch'])
    assert 1 <= batch_size <= 1000

    transformer = naive_transformer if config['slow'] else smart_transformer

    with tempfile.TemporaryDirectory() as temp_dir:
        print('Temp directory is', temp_dir)
        width, height, fps, count = split_video_to_frames(input_file, temp_dir, jpeg_quality)

        if width > count:
            print(f'Warning! Frame count ({count}) is less then the width ({width}).')
            print(f'Ideally it should have length of {(width / fps):.1f} sec')

        fourcc = config['codec']
        out_movie = cv2.VideoWriter(output_file,
                                    cv2.VideoWriter_fourcc(*fourcc),
                                    fps, (width, height))

        # out video
        # width => time, time => width
        # (1280 x 720 x (100 frames) => 1280 (stretch out 100) x 720 (keep) x (1280 frames)

        transformer(out_movie, width, height, count, temp_dir, batch_size)

        out_movie.release()

    print('Done!')


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('output', True),
        ArgParameter('codec', False, 'x264'),
        ArgParameter('jpeg-quality', False, 80),
        ArgParameter('batch', False, 100),
        ArgFlag('slow')
    ])

    do_work(config)
