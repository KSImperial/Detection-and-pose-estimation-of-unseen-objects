#base file created by Seung Back

import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import json
from pycocotools import mask as m
import datetime
import ast


coco_json = {
    "info": {
        "description": "Unknown object Amodal Instance Segmentation (UOAIS)",
        "url": "https://github.com/gist-ailab/uoais",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "Seunghyeok Back",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ],
    "categories": [
        {
            'id': 1,
            'name': 'object',
            'supercategory': 'shape',
        }
    ],
    "images": [],
    "annotations": []
}


def get_file_paths(data_root, mode, env, file_paths):
    
    file_path_name =  "{}_{}_file_paths.json".format(mode, env)
    file_path_json = os.path.join(data_root, file_path_name)

    data_root = os.path.join(data_root, "{}/{}".format(mode, env))
    file_names = sorted(os.listdir(os.path.join(data_root, "color")))
    print(len(file_names))
    visible_masks = glob.glob(os.path.join(data_root, 'visible_mask/*'))
    amodal_masks = glob.glob(os.path.join(data_root, 'amodal_mask/*'))
    print("Writing UOAIS file paths", mode)
    
    visible_dict = {}
    for visible_mask in visible_masks:
        id = visible_mask.split("/")[-1].split("_")[0]
        if id in visible_dict.keys():
            visible_dict[id].append(visible_mask)
        else:
            visible_dict[id] = [visible_mask]
    amodal_dict = {}
    for amodal_mask in amodal_masks:
        id = amodal_mask.split("/")[-1].split("_")[0]
        if id in amodal_dict.keys():
            amodal_dict[id].append(amodal_mask)
        else:
            amodal_dict[id] = [amodal_mask]

    # for file_name in tqdm(file_names):
    n_file = 2500 if mode == "val" else 999
    for i in range(1, n_file + 1):
        image_id = str(i)
        # image_id = file_name.split('.')[0]
        file_paths["color"].append(os.path.join(data_root, "color", image_id + '.png'))
        file_paths["depth"].append(os.path.join(data_root, 'depth', image_id + '.png'))
        file_paths["visible_mask"].append(sorted(visible_dict[list(visible_dict)[0]])[(int(image_id))])
        # file_paths["visible_mask"].append(sorted(visible_dict["bin\\visible"][int(image_id)])) ##leading zeros by KS - original
        #file_paths["visible_mask"].append(sorted(visible_dict[str(image_id.rjust(5,'0'))])) ##leading zeros by KS
#        file_paths["amodal_mask"].append(sorted(amodal_dict["bin\\amodal"][int(image_id)]))
        file_paths["amodal_mask"].append(sorted(amodal_dict[list(amodal_dict)[0]])[int(image_id)])
        #file_paths["amodal_mask"].append(sorted(amodal_dict[image_id.rjust(5,'0')]))

    with open(file_path_json, "w") as f:
        json.dump(file_paths, f)
    return file_paths


def mask_to_rle(mask):
    rle = m.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) ==0:
        return None, None, None, None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)


def create_image_info(image_id, file_name, depth_file_name):
    return {
        "id": image_id,
        "file_name": file_name,
        "depth_file_name": depth_file_name,
        "width": 640,
        "height": 480,
        "date_captured": datetime.datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }        


def create_uoais_annotation(data_root, mode):
    
    file_paths = {"color": [], "depth": [], "visible_mask": [], "amodal_mask": []}
    #file_paths = get_file_paths(data_root, mode, "bin", file_paths)
    file_paths = get_file_paths(data_root, mode, "tabletop", file_paths)

    coco_ann_name = ''.join(("coco_anns_uoais_amodal_", mode)) + ".json"
    coco_json_path = os.path.join(data_root, "annotations", coco_ann_name)

    print("Loading UOAIS dataset [{}]: Num_IMG: {}".format(data_root, len(file_paths["color"])))
    annotations = []
    image_infos = []
    annotation_id = 1
    for img_id in tqdm(range(len(file_paths["color"]))):

        color_path = file_paths["color"][img_id] 
        depth_path = file_paths["depth"][img_id]
        visible_mask_paths = file_paths["visible_mask"][img_id]
        amodal_mask_paths = file_paths["amodal_mask"][img_id]
        visible_masks = {}
        amodal_masks = {}
        occluded_masks = {}
        occluding_masks = {}
        occluded_rates = {}

        inst_id = 0
        listOne = []
        listTwo = []
        listOne.append(visible_mask_paths)
        listTwo.append(amodal_mask_paths)
        for visible_mask_path, amodal_mask_path in zip(listOne,listTwo):
            visible_mask = cv2.imread(visible_mask_path)
            visible_mask = cv2.resize(visible_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            visible_mask = np.array(visible_mask[:, :, 0], dtype=bool, order='F')

            amodal_mask = cv2.imread(amodal_mask_path)
            amodal_mask = cv2.resize(amodal_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            amodal_mask = np.array(amodal_mask[:, :, 0], dtype=bool, order='F')
            
            # get only occluded mask with overlapping ratio > 0.05
            occluded_mask_all = np.uint8(np.logical_and(amodal_mask, np.logical_not(visible_mask)))
            valid_contours = []
            occluded_rate = 0
            contours, _ = cv2.findContours(np.uint8(occluded_mask_all*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            valid_contours = []
            for contour in contours:
                overlapping_ratio = cv2.contourArea(contour) / np.sum(amodal_mask)
                if overlapping_ratio >= 0.05:
                    valid_contours.append(contour)
            
            occluded_mask = np.uint8(np.zeros((480, 640)))
            if len(valid_contours) != 0: 
                # if occluded -> amodal = visible + occluded
                occluded_mask = cv2.drawContours(occluded_mask, valid_contours, -1, 255, -1)
                occluded_mask = np.array(occluded_mask, dtype=bool, order='F')
                amodal_mask = np.bitwise_or(occluded_mask, visible_mask)
            else: # if no occluded -> amodal = visible
                amodal_mask = visible_mask
                occluded_mask = np.zeros_like(visible_mask, dtype=bool, order='F')
            
            amodal_masks[inst_id] = amodal_mask
            visible_masks[inst_id] = visible_mask
            occluded_masks[inst_id] = occluded_mask
            occluded_rate = np.sum(occluded_mask) / np.sum(amodal_mask)
            occluded_rates[inst_id] = occluded_rate
            occluding_masks[inst_id] = np.zeros_like(occluded_mask)
            inst_id += 1

        # assign occluded mask of object A as occluding mask of object B if the occluded mask of B is fully covered by visible mask of object B
        for a in occluded_masks.keys():
            occluded_mask_a = occluded_masks[a]
            if np.sum(occluded_mask_a) == 0:
                continue
            for b in visible_masks.keys():
                visible_mask_b = visible_masks[b]
                overlap = np.sum(np.bitwise_and(occluded_mask_a, visible_mask_b))
                if overlap == 0:
                    continue 
                overlap_ratio = overlap / np.sum(occluded_mask_a)
                if overlap_ratio > 0.05:
                    occluding_masks[b] = np.bitwise_and(np.bitwise_or(occluded_mask_a, occluding_masks[b]), visible_masks[b])

                    # cv2.imwrite("vis/img{}_inst{}_amodal.png".format(img_id, b), 255*np.uint8(amodal_masks[b]))
                    # cv2.imwrite("vis/img{}_inst{}_visible.png".format(img_id, b), 255*np.uint8(visible_masks[b]))
                    # cv2.imwrite("vis/img{}_inst{}_occluded.png".format(img_id, b), 255*np.uint8(occluded_masks[b]))
                    # cv2.imwrite("vis/img{}_inst{}_occluding.png".format(img_id, b), 255*np.uint8(occluding_masks[b]))

        for idx in visible_masks.keys():

            visible_mask = visible_masks[idx]
            amodal_mask = amodal_masks[idx]
            occluded_mask = occluded_masks[idx]
            occluding_mask = occluding_masks[idx]
            visible_bbox = get_bbox(visible_mask)
            amodal_bbox = get_bbox(amodal_mask)

            if visible_bbox[0] is None: 
                print("Filtering none bbox")
                continue
            if visible_bbox[2] <= 1 or visible_bbox[3] <= 1:
                print("Filtering too small mask", color_path)
                continue

            annotation = {}
            annotation["id"] = annotation_id
            annotation_id += 1
            annotation["image_id"] = img_id
            annotation["category_id"] = 1
            annotation["bbox"] = amodal_bbox
            annotation["height"] = 480
            annotation["width"] = 640
            annotation["iscrowd"] = 0
            annotation["segmentation"] = mask_to_rle(amodal_mask)
            annotation["area"] = int(np.sum(amodal_mask))
            annotation["visible_mask"] = mask_to_rle(visible_mask)
            annotation["visible_bbox"] = visible_bbox
            annotation["occluded_mask"] = mask_to_rle(occluded_mask)
            annotation["occluding_mask"] = mask_to_rle(occluding_mask)
            annotation["occluded_rate"] = occluded_rates[idx]
            annotation["occluding_rate"] = np.sum(occluded_masks[idx]) / np.sum(amodal_masks[idx])

            annotations.append(annotation)
            image_infos.append(create_image_info(img_id, "/".join(color_path.split("/")[-3:]), "/".join(depth_path.split("/")[-3:])))

    coco_json["annotations"] = annotations
    coco_json["images"] = image_infos
    with open(coco_json_path, "w") as f:
        print("Saving annotation as COCO format to", coco_json_path)
        json.dump(coco_json, f)
    return coco_json_path

if __name__ == "__main__":
    
    data_root = "./datasets/UOAIS-SIM"
    create_uoais_annotation(data_root, mode="train")
    create_uoais_annotation(data_root, mode="val")