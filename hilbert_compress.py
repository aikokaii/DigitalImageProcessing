import sys
import struct
import zlib
from PIL import Image


def _rot(n: int, x: int, y: int, rx: int, ry: int):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return (y, x)
    return (x, y)


def hilbert_index_to_xy(idx: int, order: int):
    n = 1 << order
    x = y = 0
    s = 1
    t = idx
    while s < n:
        rx = (t >> 1) & 1
        ry = (t & 1) ^ rx
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def compress_image(in_path: str, out_path: str):
    img = Image.open(in_path).convert('RGB')
    w, h = img.size

    M = next_power_of_two(max(w, h))
    padded = Image.new('RGB', (M, M), (0, 0, 0))
    padded.paste(img, (0, 0))

    pixels = list(padded.getdata())
    order = M.bit_length() - 1

    reordered = bytearray()
    for i in range(M * M):
        x, y = hilbert_index_to_xy(i, order)
        r, g, b = pixels[y * M + x]
        reordered.extend([r, g, b])

    compressed = zlib.compress(bytes(reordered), level=9)

    with open(out_path, 'wb') as f:
        f.write(b'HILC')
        f.write(struct.pack('>II', w, h))
        f.write(struct.pack('>I', M))
        f.write(struct.pack('>I', len(compressed)))
        f.write(compressed)


def decompress_image(in_path: str, out_path: str):
    with open(in_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'HILC':
            raise ValueError("Not a valid Hilbert‑compressed file")

        w, h = struct.unpack('>II', f.read(8))
        M = struct.unpack('>I', f.read(4))[0]
        comp_len = struct.unpack('>I', f.read(4))[0]
        compressed = f.read(comp_len)

    data = zlib.decompress(compressed)
    expected_len = M * M * 3
    if len(data) != expected_len:
        raise ValueError("Corrupted compressed data – size mismatch")

    order = M.bit_length() - 1
    pixels = [None] * (M * M)

    for i in range(M * M):
        x, y = hilbert_index_to_xy(i, order)
        r, g, b = data[3*i : 3*i+3]
        pixels[y * M + x] = (r, g, b)

    padded_img = Image.new('RGB', (M, M))
    padded_img.putdata(pixels)

    cropped = padded_img.crop((0, 0, w, h))
    cropped.save(out_path)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()
    in_file = sys.argv[2]
    out_file = sys.argv[3]

    if mode == 'compress':
        compress_image(in_file, out_file)
        print(f"Compressed {in_file} -> {out_file}")
    elif mode == 'decompress':
        decompress_image(in_file, out_file)
        print(f"Decompressed {in_file} -> {out_file}")
    else:
        print(f"Unknown mode: {mode}. Use 'compress' or 'decompress'.")
        sys.exit(1)


if __name__ == '__main__':
    main()

#for compression
#python hilbert_compress.py compress image.jpg/png output.hil

#for decompression
#python hilbert_compress.py decompress output.hil restored.jpg/png
