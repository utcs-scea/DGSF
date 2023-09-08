import os
import cv2
import numpy as np


class LFWDataset:

    __slots__ = ["dataset_path", "pairs_list_path", "nameLs", "nameRs", 
                 "folds", "flags"]
    def __init__(self, dataset_path, pairs_list_path):
        self.dataset_path = dataset_path
        self.pairs_list_path = pairs_list_path
        self.nameLs = []
        self.nameRs = []
        self.flags = []

        with open(pairs_list_path, 'r') as fin:
            all_lines = fin.read().splitlines()
            pairs = all_lines[1:]
        for i, p in enumerate(pairs):
            p = p.split('\t')
            if len(p) == 3:
                nameL = os.path.join(self.dataset_path, p[0], 
                #nameL = os.path.join(p[0], 
                                     "{}_{:04}.jpg.npy".format(p[0], int(p[1])))
                nameR = os.path.join(self.dataset_path, p[0], 
                #nameR = os.path.join(p[0], 
                                     "{}_{:04}.jpg.npy".format(p[0], int(p[2])))
                flag = True
            elif len(p) == 4:
                nameL = os.path.join(self.dataset_path, p[0], 
                #nameL = os.path.join(p[0], 
                                     "{}_{:04}.jpg.npy".format(p[0], int(p[1])))
                nameR = os.path.join(self.dataset_path, p[2], 
                #nameR = os.path.join(p[2], 
                                     "{}_{:04}.jpg.npy".format(p[2], int(p[3])))
                flag = False
            else:
                raise Exception("Unrecognized line: {}".format(p))
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.flags.append(flag)
        #for i in range(256):
        #    print(f"{self.nameLs[i]}, {self.nameRs[i]}, {self.flags[i]}")
        print("Loaded lfw list")

    def get_item(self, nr):
        nameL_img = np.load(self.nameLs[nr])
        nameR_img = np.load(self.nameRs[nr])
        return nameL_img, nameR_img, self.flags[nr]

    def get_samples(self, id_list):
        nameL_imgs = []
        nameR_imgs = []
        flags = []
        for id_ in id_list:
            nameL_img, nameR_img, flag = self.get_item(id_)
            nameL_imgs.append(nameL_img)
            nameR_imgs.append(nameR_img)
            flags.append(flag)
        nameL_imgs = np.array(nameL_imgs)
        nameR_imgs = np.array(nameR_imgs)
        flags = np.array(flags)
        return nameL_imgs, nameR_imgs, flags