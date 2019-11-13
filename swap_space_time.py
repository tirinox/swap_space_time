import cv2
import sys
import tempfile
from arg_parser import arg_parser, ArgFlag, ArgParameter
from swspt.naive import naive_transformer
from swspt.memmap import mmap_transformer
from swspt.batched import batched_transformer
from swspt.helpers import split_video_to_frames


def do_work(config: dict):
    jpeg_quality = int(config['jpeg-quality'])
    assert 20 <= jpeg_quality <= 100

    input_file = config['input']
    output_file = config['output']

    batch_size = int(config['batch'])
    assert 1 <= batch_size <= 1000

    limit = int(config['limit'])

    transformer = {
        'naive': naive_transformer,
        'batched': batched_transformer,
        'mmap': mmap_transformer
    }[config['algo']]

    with tempfile.TemporaryDirectory() as temp_dir:
        print('Temp directory is', temp_dir)
        width, height, fps, count = split_video_to_frames(input_file, temp_dir, jpeg_quality, limit)

        if width > count:
            print(f'Warning! Frame count ({count}) is less then the width ({width}). '
                  f'Ideally it should have length of {(width / fps):.1f} sec')

        fourcc = config['codec']
        out_movie = cv2.VideoWriter(output_file,
                                    cv2.VideoWriter_fourcc(*fourcc),
                                    fps, (width, height))

        # out video
        # width => time, time => width
        # (1280 x 720 x (100 frames) => 1280 (stretch out 100) x 720 (keep) x (1280 frames shrink to 100)
        writer = lambda frame: out_movie.write(frame)
        transformer(writer, width, height, count, temp_dir, batch_size)

        out_movie.release()

    print('Done!')


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('output', True),
        ArgParameter('limit', -1),
        ArgParameter('codec', False, 'x264'),
        ArgParameter('jpeg-quality', False, 80),
        ArgParameter('batch', False, 10),
        ArgParameter('algo', 'mmap')
    ])

    do_work(config)
