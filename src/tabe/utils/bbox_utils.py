import numpy as np


def convert_bbox_to_relative(bbox_xyxy, img_shape):
    return [bbox_xyxy[0] / img_shape[0], bbox_xyxy[1] / img_shape[1], bbox_xyxy[2] / img_shape[0],
            bbox_xyxy[3] / img_shape[1]]


def convert_bbox_to_absolute(bbox_xyxy, img_shape):
    return [int(bbox_xyxy[0] * img_shape[0]), int(bbox_xyxy[1] * img_shape[1]), int(bbox_xyxy[2] * img_shape[0]),
            int(bbox_xyxy[3] * img_shape[1])]


def turn_bbox_into_mask(bbox_xyxy, mask_shape):
    bounding_box_mask = np.zeros(mask_shape)
    bbox_xyxy = np.array(bbox_xyxy)
    bbox_xyxy[::2] = bbox_xyxy[::2].clip(0, mask_shape[0] - 1)
    bbox_xyxy[1::2] = bbox_xyxy[1::2].clip(0, mask_shape[1] - 1)
    bounding_box_mask[bbox_xyxy[0]:bbox_xyxy[2] + 1, bbox_xyxy[1]:bbox_xyxy[3] + 1] = 1

    return bounding_box_mask


def add_h_w_perc_to_bbox(bbox_xyxy, h_perc=10, w_perc=10):
    extended_bbox_xyxy = np.array(bbox_xyxy).copy()
    control_bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
    control_bbox_h = bbox_xyxy[3] - bbox_xyxy[1]

    extended_bbox_xyxy[0] -= control_bbox_w * (w_perc / 100)
    extended_bbox_xyxy[1] -= control_bbox_h * (h_perc / 100)
    extended_bbox_xyxy[2] += control_bbox_w * (w_perc / 100)
    extended_bbox_xyxy[3] += control_bbox_h * (h_perc / 100)

    return extended_bbox_xyxy.tolist()


def draw_psuedo_bbox_on_input(im_size, psuedo_bbox, in_front_obj_arr, added_bbox_perc=10):
    if np.all([p < 1 for p in psuedo_bbox]):
        psuedo_bbox = convert_bbox_to_absolute(psuedo_bbox, im_size)
    if added_bbox_perc > 0:
        psuedo_bbox = add_h_w_perc_to_bbox(psuedo_bbox, h_perc=added_bbox_perc, w_perc=added_bbox_perc)
    bounding_box_mask = turn_bbox_into_mask(psuedo_bbox, im_size)
    in_front_obj_and_psuedo_bbox = in_front_obj_arr.copy()
    in_front_obj_and_psuedo_bbox[bounding_box_mask == 0] = 0
    in_front_obj_and_psuedo_bbox[in_front_obj_and_psuedo_bbox == 127] = 0

    return in_front_obj_and_psuedo_bbox, bounding_box_mask


def calc_bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    width = x2 - x1
    height = y2 - y1
    return width * height
