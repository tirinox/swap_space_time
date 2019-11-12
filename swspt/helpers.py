import util
import sys
import tqdm
import cv2


def get_input_frame_name(index, tmp_dir):
    return util.get_file_name(index, tmp_dir, prefix='in', ext='jpg')


def progress_bar(it, unit='frame'):
    return tqdm.tqdm(it, file=sys.stdout, unit=unit)


def split_video_to_frames(movie_input, tmp_path, jpeg_quality, limit):
    cap = cv2.VideoCapture(movie_input)
    if not cap.isOpened():
        util.error('could not open input file')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Input frame count: {count} ({width} x {height}) @ {fps:.2f} fps')

    print('Decomposing the source movie...')

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

        if 0 < limit == frame_counter:
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
