#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image
import labelme_utils as utils


def main(input_dir, output_dir, label_list):

    os.makedirs(osp.join(output_dir, 'JPEGImages'))
    os.makedirs(osp.join(output_dir, 'SegmentationClass'))
    os.makedirs(osp.join(output_dir, 'SegmentationClassPNG'))
    os.makedirs(osp.join(output_dir, 'SegmentationClassVisualization'))
    os.makedirs(osp.join(output_dir, 'SegmentationObject'))
    os.makedirs(osp.join(output_dir, 'SegmentationObjectPNG'))
    os.makedirs(osp.join(output_dir, 'SegmentationObjectVisualization'))
    print('Creating dataset:', output_dir)

    class_names = []
    class_name_to_id = {}
    # for i, line in enumerate(open(labels).readlines()):
    #     class_id = i - 1  # starts with -1
    #     class_name = line.strip()
    #     class_name_to_id[class_name] = class_id
    #     if class_id == -1:
    #         assert class_name == '__ignore__'
    #         continue
    #     elif class_id == 0:
    #         assert class_name == '_background_'
    #     class_names.append(class_name)
    class_name_to_id['_background_'] = 0
    class_names.append('_background_')
    for i in range(len(label_list)):
        class_id = i + 1
        class_name = label_list[i]
        class_name_to_id[class_name] = class_id
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = utils.label_colormap(255)

    for label_file in glob.glob(osp.join(input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                output_dir, 'JPEGImages', base + '.jpg')
            out_cls_file = osp.join(
                output_dir, 'SegmentationClass', base + '.npy')
            out_clsp_file = osp.join(
                output_dir, 'SegmentationClassPNG', base + '.png')
            out_clsv_file = osp.join(
                output_dir,
                'SegmentationClassVisualization',
                base + '.jpg',
            )
            out_ins_file = osp.join(
                output_dir, 'SegmentationObject', base + '.npy')
            out_insp_file = osp.join(
                output_dir, 'SegmentationObjectPNG', base + '.png')
            out_insv_file = osp.join(
                output_dir,
                'SegmentationObjectVisualization',
                base + '.jpg',
            )

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            PIL.Image.fromarray(img).save(out_img_file)

            cls, ins = utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
                type='instance',
            )
            ins[cls == -1] = 0  # ignore it.

            # class label
            utils.lblsave(out_clsp_file, cls)
            np.save(out_cls_file, cls)
            clsv = utils.draw_label(
                cls, img, class_names, colormap=colormap)
            PIL.Image.fromarray(clsv).save(out_clsv_file)

            # instance label
            utils.lblsave(out_insp_file, ins)
            np.save(out_ins_file, ins)
            instance_ids = np.unique(ins)
            instance_names = [str(i) for i in range(max(instance_ids) + 1)]
            insv = utils.draw_label(ins, img, instance_names)
            PIL.Image.fromarray(insv).save(out_insv_file)


if __name__ == '__main__':
    """
    # It generates:
    #   - data_dataset_voc/JPEGImages
    #   - data_dataset_voc/SegmentationClass
    #   - data_dataset_voc/SegmentationClassVisualization
    #   - data_dataset_voc/SegmentationObject
    #   - data_dataset_voc/SegmentationObjectVisualization
    ./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
    """
    main(r"E:\file", r"E:\file_output", ["container_top"])
