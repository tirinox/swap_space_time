from swspt.batched import batched_transformer
from tempfile import TemporaryDirectory
import os
import numpy as np
import cv2 as cv2
import math
from tqdm import tqdm


def test_bt1():
    def frame_from_text(lines: str):
        img = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = []
            for symbol in line:
                if symbol == ' ':
                    continue
                if symbol == 'A':
                    color = (255, 255, 255)
                elif symbol == 'a':
                    color = (128, 128, 128)
                elif symbol == '-':
                    color = (0, 0, 0)
                else:
                    continue
                row.append(color)
            img.append(row)
        return np.array(img, dtype=np.uint8)

    tmp_dir = 'example'
    images = list(map(frame_from_text, [
        [
            "---A",
            "----",
            "A---",
            "----",
            "----",
        ],
        [
            "--A-",
            "----",
            "AA--",
            "----",
            "----",
        ],
        [
            "-A--",
            "----",
            "aAA-",
            "----",
            "----",
        ],
        [
            "A---",
            "----",
            "-aAA",
            "----",
            "----",
        ]
        ,
        [
            "----",
            "----",
            "--aA",
            "----",
            "----",
        ]
    ]))

    for i, img in enumerate(images):
        cv2.imwrite(f'example/__temp_{i}.png', img)

    images_t = np.swapaxes(images, 0, 2)

    for i, img in enumerate(images_t):
        cv2.imwrite(f'example/__swapped_{i}.png', img)

    def loader(frame_index, temp_dir):
        return images[frame_index]

    i = 0

    def dummy_writer(frame):
        nonlocal i
        cv2.imwrite(f'example/__test_out_{i}.png', frame)
        i += 1

    batched_transformer(dummy_writer, 4, 5, 4, tmp_dir, 2, loader)


def make_sample_movie():
    def random_color():
        return np.random.uniform((0, 0, 0), (255, 255, 255)).astype(dtype=np.uint8)

    def pattern(w, h, color1, color2, hs=1, dv=20):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        dv2 = dv // 20
        start = 0
        for y in range(h):
            i = start
            for x in range(w):
                img[y, x] = color1 if i % dv > dv2 else color2
                i += 1
            start += hs
        return img


    def background(w, h, cw, ch):
        ncx = math.ceil(w / cw)
        ncy = math.ceil(h / ch)
        over_w = int(ncx * cw)
        over_h = int(ncy * ch)
        img = np.zeros((over_h, over_w, 3), dtype=np.uint8)

        for x in range(0, over_w, cw):
            for y in range(0, over_h, ch):
                color1 = random_color()
                color2 = random_color()
                img[slice(y, y + ch), slice(x, x + cw), :] = pattern(cw, ch, color1, color2)

        return img[:h, :w, :]


    def background_grad(w, h, *args):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            for y in range(h):
                img[y, x] = (int(255 * x / w), int(255 * y / h), 0)
        return img


    FPS = 30
    W, H = 120, 80
    FRAMES = 60
    NAME = './example/gen_sample_video.mp4'

    if os.path.exists(NAME):
        os.remove(NAME)

    out_movie = cv2.VideoWriter(NAME,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                FPS, (W, H))

    bg = background_grad(W, H, 20, 20)
    for fr in tqdm(range(FRAMES)):
        x = int(fr * 2.5)
        y = H // 2
        r = 10 + fr // 8
        frame = bg.copy()
        cv2.circle(frame, (x, y), r, (255, 0, 0), -1)
        cv2.circle(frame, (x, y), r + 10, (255, 255, 255), -1)
        cv2.putText(frame, f'{fr}', (10, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
        cv2.putText(frame, f'{fr}', (W - 40, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
        # cv2.imshow('1', frame)
        # cv2.waitKey()
        out_movie.write(frame)

    out_movie.release()
    os.system(f'open "{NAME}"')


make_sample_movie()
