import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_ycbcr(img: np.ndarray):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    output = np.stack((y, cb, cr), axis=-1)
    return np.clip(output, 0, 255).astype(np.uint8)


def create_mask(img: np.ndarray, cb_low: int, cb_high: int, cr_low: int, cr_high: int):
    mask_cb = (cb_low < img[:, :, 1]) & (img[:, :, 1] < cb_high)
    mask_cr = (cr_low < img[:, :, 2]) & (img[:, :, 2] < cr_high)
    return np.uint8(mask_cb & mask_cr)


def filter_mask(mask: np.ndarray, kernel_size: int = 5):
    return cv2.medianBlur(mask, kernel_size)


def find_center(mask: np.ndarray):
    m00 = np.sum(mask)
    m10 = np.sum(mask * np.arange(mask.shape[1]))
    m01 = np.sum(mask * np.arange(mask.shape[0]).reshape(-1, 1))

    if m00 == 0:
        return False

    return (int(m10 / m00), int(m01 / m00))


def main():
    parser = argparse.ArgumentParser(
        description="Find the center of an area with skintone like color."
    )
    parser.add_argument("path", type=str, help="Path to the source image")
    args = parser.parse_args()

    img = cv2.imread(args.path, cv2.IMREAD_COLOR_RGB)
    img_ycbcr = to_ycbcr(img)
    mask = create_mask(img_ycbcr, 85, 115, 135, 165)
    filtered_mask = filter_mask(mask)
    center = find_center(filtered_mask)

    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Oryginal")

    plt.subplot(1, 5, 2)
    plt.imshow(img_ycbcr)
    plt.axis("off")
    plt.title("YCbCr")

    plt.subplot(1, 5, 3)
    plt.imshow(mask, "grey")
    plt.axis("off")
    plt.title("Mask")

    plt.subplot(1, 5, 4)
    plt.imshow(filtered_mask, "grey")
    plt.axis("off")
    plt.title("Filtered mask")

    plt.subplot(1, 5, 5)
    plt.imshow(filtered_mask, "grey")
    plt.axis("off")
    if center:
        plt.axhline(center[1], color="red")
        plt.axvline(center[0], color="red")
    plt.title("Center")

    plt.tight_layout()
    plt.gcf().set_size_inches((15, 5))
    plt.show()


if __name__ == "__main__":
    main()
