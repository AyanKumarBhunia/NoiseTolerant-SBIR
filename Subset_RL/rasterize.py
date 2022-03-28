import numpy as np
from bresenham import bresenham
import scipy.ndimage
from PIL import Image


def get_stroke_num(vector_image):
    return len(np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1])

def select_strokes(vector_image, strokes):
    """
    :param vector_image: x,y,p coordinate array
    :param strokes: stroke indexes to keep
    :return: vector_image: after keeping only selected strokes
    """
    c = vector_image
    c_split = np.split(c[:, :2], np.where(c[:, 2])[0] + 1, axis=0)[:-1]

    c_selected = []
    for i in strokes:
        c_selected.append(c_split[i])

    xyp = []
    for i in c_selected:
        p = np.zeros((len(i), 1))
        p[-1] = 1
        xyp.append(np.hstack((i, p)))
    xyp = np.concatenate(xyp)
    return xyp


def mydrawPNG_fromlist(vector_image, stroke_idx, Side=256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    if stroke_idx.size < 1:
        raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
        return Image.fromarray(raster_image).convert('RGB')

    stroke_idx = stroke_idx.squeeze()
    # print(stroke_idx)

    if stroke_idx.size == 1:
        stroke_idx = [stroke_idx.item()]

    vector_image = np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1]
    vector_image = [vector_image[x] for x in stroke_idx]

    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

        for i_pos in range(1, len(stroke)):
            cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= Side and cord[1] <= Side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error')
            initX, initY = int(stroke[i_pos, 0]), int(stroke[i_pos, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return Image.fromarray(raster_image).convert('RGB')


def mydraw_redPNG_fromlist(vector_image, stroke_idx, Side=256):
    vector_image = np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1]
    vector_image = [vector_image[x] for x in stroke_idx]

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)

    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

        for i_pos in range(1, len(stroke)):
            cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= Side and cord[1] <= Side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error')
            initX, initY = int(stroke[i_pos, 0]), int(stroke[i_pos, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    channels = np.stack((raster_image,) * 3, axis=-1)
    return np.moveaxis(channels, -1, 0)


def mydrawPNG(vector_image, Side=256):
    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return raster_image


def preprocess(sketch_points, side=256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points


def rasterize_Sketch(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images = mydrawPNG(sketch_points)
    return raster_images


def convert_to_red(image):
    l = image.shape[1]
    image[1] = np.zeros((l,l))
    image[2] = np.zeros((l,l))
    return image


def convert_to_blue(image):
    l = image.shape[1]
    image[0] = np.zeros((l,l))
    image[1] = np.zeros((l,l))
    return image


def convert_to_green(image):
    l = image.shape[1]
    image[0] = np.zeros((l,l))
    image[2] = np.zeros((l,l))
    return image

def convert_to_black(image):
    l = image.shape[1]
    image[0] = np.zeros((l,l))
    image[1] = np.zeros((l,l))
    image[2] = np.zeros((l,l))
    return image