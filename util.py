import os

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:(i + n)]


def error(text):
    print('Error:', text)
    exit(-1)


def get_file_name(index, tmp_dir, prefix, ext):
    return os.path.join(tmp_dir, f'{prefix}_{index}.{ext}')


def sep(s='-', n=120):
    print(s * n)