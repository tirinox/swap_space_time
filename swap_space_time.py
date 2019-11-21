import cv2
import sys
import os
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

        write_frames = config['write-frames']

        if write_frames:
            output_dir = output_file + '-frames'
            os.makedirs(output_dir, exist_ok=True)
            frame_i = 0

            def _writer(frame):
                nonlocal frame_i
                cv2.imwrite(os.path.join(output_dir, f'frame_{frame_i:05}.png'), frame)
                frame_i += 1

            writer = _writer
        else:
            if os.path.exists(output_file):
                print(f'Removed existing output file: "{output_file}"')
                os.remove(output_file)

            fourcc = config['codec']
            out_movie = cv2.VideoWriter(output_file,
                                        cv2.VideoWriter_fourcc(*fourcc),
                                        fps, (width, height))

            writer = lambda frame: out_movie.write(frame)


        # out video
        # width => time, time => width
        # (1280 x 720 x (100 frames) => 1280 (stretch out 100) x 720 (keep) x (1280 frames shrink to 100)

        transformer(writer, width, height, count, temp_dir, batch_size)

        if not write_frames:
            out_movie.release()

    print('Done!')


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('output', True),
        ArgParameter('limit', -1),
        ArgParameter('codec', False, 'mp4v'),
        ArgParameter('jpeg-quality', False, 80),
        ArgParameter('batch', False, 10),
        ArgParameter('algo', 'mmap'),
        ArgFlag('write-frames', False)
    ])

    do_work(config)
