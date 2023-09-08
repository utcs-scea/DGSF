"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import re
import time

import cv2
import numpy as np

import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")

import boto3, io

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)


class Imagenet(dataset.Dataset):

    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None):
        super(Imagenet, self).__init__()
        if image_size is None:
            self.image_size = [224, 224, 3]
        else:
            self.image_size = image_size
        if not cache_dir:
            cache_dir = os.getcwd()
        self.image_list = []
        self.label_list = []
        self.count = count
        self.use_cache = use_cache
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        self.data_path = data_path
        self.pre_process = pre_process
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False

        not_found = 0
        #if image_list is None:
        #    # by default look for val_map.txt
        #    image_list = os.path.join(data_path, "val_map.txt")

        image_list_io = io.BytesIO()
        s3_client.download_fileobj("hf-dgsf", "resnet/val_map_2048.txt", image_list_io)
        image_list_io.seek(0)
        img_list = image_list_io.read().decode('UTF-8')

        #os.makedirs(self.cache_dir, exist_ok=True)

        start = time.time()
        #with open(image_list, 'r') as f:
        cc = 0

        from threading import Lock
        l = Lock()

        def download_obj(line):
            image_name, label = line.split(" ")
            preproc_name = image_name+".npy"
            img = io.BytesIO()
            s3_client.download_fileobj("hf-dgsf", f"resnet/inputs/{preproc_name}", img)
            img.seek(0)
            loaded = np.load(img, allow_pickle=False)
            l.acquire()
            self.image_list.append(loaded)
            self.label_list.append(int(label))
            l.release()
            img.close()
        
        #TODO: parallelize
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as executor:
            for line in img_list.splitlines():
                executor.submit(download_obj, line)

            #src = os.path.join(data_path, image_name)
            #dst = os.path.join(self.cache_dir, image_name)
            #if not os.path.exists(src) and not os.path.exists(dst + ".npy"):
            #    # if the image does not exists ignore it
            #    not_found += 1
            #    continue
            #os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
            #if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                # TODO: make this multi threaded ?
            #    img_org = cv2.imread(src)
            #    processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
            #    np.save(dst, processed)
            
            #self.image_list.append(image_name)
            
            # limit the dataset if requested
            #if self.count and len(self.image_list) >= self.count:
            #    break

        self.total_count = len(self.image_list)

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

        self.label_list = np.array(self.label_list)

    def get_item(self, nr):
        """Get image by number in the list."""
        #dst = os.path.join(self.cache_dir, self.image_list[nr])
        #img = np.load(dst + ".npy")
        #return img, self.label_list[nr]

        return self.image_list[nr], self.label_list[nr]

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src