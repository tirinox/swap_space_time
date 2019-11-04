import cv2
import os
import sys
import tempfile
from arg_parser import arg_parser, ArgFlag, ArgParameter
import util
from functools import lru_cache


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


@lru_cache(maxsize=100)
def image_feed(frame_index, tmp_path):
    return cv2.imread(get_input_frame_name(frame_index, tmp_path), cv2.IMREAD_COLOR)


def remapper_gen(from_dim, to_dim):
    for i in range(from_dim):
        yield int(i * to_dim / from_dim)


def do_work(config: dict):
    jpeg_quality = int(config['jpeg-quality'])
    assert 20 <= jpeg_quality <= 100

    input_file = config['input']
    output_file = config['output']

    with tempfile.TemporaryDirectory() as temp_dir:
        width, height, fps, count = split_video_to_frames(input_file, temp_dir, jpeg_quality)

        # out video
        # width => time, time => width
        # (1280 x 720 x (100 frames) => 1280 (stetch out 100) x 720 (keep) x (1280 frames)


    #
    # cap = cv2.VideoCapture(config['input'])
    # if not cap.isOpened():
    #     return
    #
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # if verbose:
    #     print(f'Input frame count: {count} ({width} x {height}) @ {fps} fps')
    #
    # fourcc = config['codec']
    # out = cv2.VideoWriter(config['output'], cv2.VideoWriter_fourcc(*fourcc),
    #                       fps, (width, height))
    #
    # os.makedirs('outtemp', exist_ok=True)
    # i = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         print(f'Frame: {i}')
    #         cv2.imshow('Frame', frame)
    #         out.write(frame)
    #         cv2.imwrite(f'outtemp/{i}.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    #         i += 1
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    #
    # cap.release()
    # out.release()


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('output', True),
        ArgParameter('codec', False, 'x264'),
        ArgParameter('jpeg-quality', False, 80),
        ArgFlag('verbose'),
    ])
    verbose = config['verbose'] # global

    print(list(remapper_gen(220, 45)))

    # do_work(config)
