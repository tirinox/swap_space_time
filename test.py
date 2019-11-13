from swspt.batched import batched_transformer
from tempfile import TemporaryDirectory
import os
import numpy as np
import cv2


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


test_bt1()