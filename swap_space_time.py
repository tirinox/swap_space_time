import cv2
import os
import sys
import tempfile
from arg_parser import arg_parser, ArgFlag, ArgParameter
import util
import numpy as np
import tqdm


def get_input_frame_name(index, tmp_dir):
    return util.get_file_name(index, tmp_dir, prefix='in', ext='jpg')


def split_video_to_frames(movie_input, tmp_path, jpeg_quality):
    cap = cv2.VideoCapture(movie_input)
    if not cap.isOpened():
        util.error('could not open input file')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f'Input frame count: {count} ({width} x {height}) @ {fps:.2f} fps')

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_name = get_input_frame_name(frame_counter, tmp_path)
            cv2.imwrite(frame_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

            if verbose:
                print(f'Frame: {frame_counter}, size: {os.path.getsize(frame_name) // 1024} KB')
            frame_counter += 1
        else:
            break
    cap.release()

    if verbose:
        print(f'Real frame count: {frame_counter}')

    return width, height, fps, frame_counter


def image_feed(frame_index, tmp_path):
    print('read img', frame_index)
    return cv2.imread(get_input_frame_name(frame_index, tmp_path), cv2.IMREAD_COLOR)


def partial_frame_name(index, tmp_path):
    return util.get_file_name(index, tmp_path, prefix='pf', ext='npy')


def save_part_frame(data: np.array, frame_index, tmp_path):
    data.save(partial_frame_name(frame_index, tmp_path))


def load_part_frame(frame_index, tmp_path):
    try:
        return np.load(partial_frame_name(frame_index, tmp_path))
    except FileNotFoundError:
        return None


def remapper_gen(from_dim, to_dim):
    for i in range(from_dim):
        yield i, int(i * to_dim / from_dim)


def naive_transformer(output_file, fps, width, height, count, temp_dir):
    fourcc = config['codec']
    out_movie = cv2.VideoWriter(output_file,
                                cv2.VideoWriter_fourcc(*fourcc),
                                fps, (width, height))

    new_times = list(remapper_gen(count, width))

    for _, new_time in tqdm.tqdm(new_times, unit='frame'):
        # each is new frame:
        new_frame = np.ndarray((height, width, 3), dtype=np.uint8)
        for x, frame in remapper_gen(width, count):
            old_frame = image_feed(frame, temp_dir)

            new_frame[:, x, :] = old_frame[:, new_time, :]
        out_movie.write(new_frame)
    out_movie.release()


def do_work(config: dict):
    jpeg_quality = int(config['jpeg-quality'])
    assert 20 <= jpeg_quality <= 100

    input_file = config['input']
    output_file = config['output']

    batch_size = int(config['batch'])
    assert 1 <= batch_size <= 1000

    with tempfile.TemporaryDirectory() as temp_dir:
        width, height, fps, count = split_video_to_frames(input_file, temp_dir, jpeg_quality)

        if width > count:
            print(f'Warning! Frame count ({count}) is less then the width ({width}).')
            print(f'Ideally it should have length of {(width / fps):.1f} sec')

        # out video
        # width => time, time => width
        # (1280 x 720 x (100 frames) => 1280 (stretch out 100) x 720 (keep) x (1280 frames)

        naive_transformer(output_file, fps, width, height, count, temp_dir)


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('output', True),
        ArgParameter('codec', False, 'x264'),
        ArgParameter('jpeg-quality', False, 80),
        ArgParameter('batch', False, 10),
        ArgFlag('verbose'),
    ])
    verbose = config['verbose']  # global

    do_work(config)
