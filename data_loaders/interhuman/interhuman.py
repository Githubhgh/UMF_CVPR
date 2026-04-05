import numpy as np
import torch
import random
import os
from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from data_loaders.interhuman.utils.utils import *
#from data_loaders.interhuman.utils.plot_script import *
from data_loaders.interhuman.utils.preprocess import *


#from body_parts import *

from yacs.config import CfgNode as CN
_C = CN(new_allowed=True)

def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

#from utils.rotation_conversions import matrix_to_rotation_6d


def read_data_list(file_path):
    try:
        with open(file_path, "r") as file:
            return file.readlines()
    except Exception as e:
        print(e, file_path)
        return []


def get_config(config_file: str, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
      cfg = default_config()
    else:
      cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


class InterHumanDataset(data.Dataset):
    def __init__(self, mode, role_graph = False, dataset_cfg_path="configs/datasets.yaml", data_root=None):

        if mode=='train':
            opt = get_config(dataset_cfg_path).interhuman.clone()
        elif mode == 'test':
            opt = get_config(dataset_cfg_path).interhuman_test.clone()
        elif mode == 'val':
            opt = get_config(dataset_cfg_path).interhuman_test.clone()

        if data_root:
            opt.defrost()
            opt.DATA_ROOT = data_root
            opt.freeze()


        self.role_graph = role_graph
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE

        #ignore_list = []
        # try:
        #     ignore_list = open(os.path.join(opt.DATA_ROOT, "ignore_list.txt"), "r").readlines()
        # except Exception as e:
        #     print(e)
        data_list = []

        if self.opt.MODE == "train":
            data_list = read_data_list(os.path.join(opt.DATA_ROOT, "train.txt"))
            if 'debug' in os.environ.get('UMF'):
                data_list = open(os.path.join(opt.DATA_ROOT, "train_debug.txt"), "r").readlines()
        elif self.opt.MODE == "val":
            data_list = read_data_list(os.path.join(opt.DATA_ROOT, "test.txt"))
        elif self.opt.MODE == "test":
            data_list = read_data_list(os.path.join(opt.DATA_ROOT, "test.txt"))
            # if 'debug' in os.environ.get('UMF'):
            #     data_list = read_data_list(os.path.join(opt.DATA_ROOT, "test_debug.txt"))



        random.shuffle(data_list)
        # data_list = data_list[:70]

        index = 0
        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT)):
            for file in tqdm(files):
                if file.endswith(".npy") and "person1" in root:
                    motion_name = file.split(".")[0]
                    # if file.split(".")[0]+"\n" in ignore_list: # or int(motion_name)>1000
                    #     print("ignore: ", file)
                    #     continue
                    if file.split(".")[0]+"\n" not in data_list:
                        continue
                    file_path_person1 = pjoin(root, file)
                    file_path_person2 = pjoin(root.replace("person1", "person2"), file)
                    text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")

                    #try:
                    texts = [item.replace("\n", "") for item in open(text_path, "r", encoding='utf-8').readlines()]
                    #except:
                        #texts = []
                        #print('Erros in ', text_path)
                    texts_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]


                    motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)
                    motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)
                    if motion1 is None:
                        continue

                    if self.cache:
                        self.motion_dict[index] = [motion1, motion2]
                        self.motion_dict[index+1] = [motion1_swap, motion2_swap]
                    else:
                        self.motion_dict[index] = [file_path_person1, file_path_person2]
                        self.motion_dict[index + 1] = [file_path_person1, file_path_person2]




                    if 'indi_text' in os.environ.get('UMF'):
                        self.data_list.append({
                            "name": motion_name,
                            "motion_id": index,
                            "swap":False,
                            "texts":texts,
                            "texts_individual1":texts_individual1_swap,
                            "texts_individual2":texts_individual2_swap,
                        })

                        if opt.MODE == "train":
                            self.data_list.append({
                                "name": motion_name+"_swap",
                                "motion_id": index+1,
                                "swap": True,
                                "texts": texts_swap,
                                "texts_individual1":texts_individual1,
                                "texts_individual2":texts_individual2,
                            })
                    else:
                        self.data_list.append({
                            "name": motion_name,
                            "motion_id": index,
                            "swap":False,
                            "texts":texts
                        })

                        if opt.MODE == "train":
                            self.data_list.append({
                                "name": motion_name+"_swap",
                                "motion_id": index+1,
                                "swap": True,
                                "texts": texts_swap,
                            })
                    index += 2


        
        # if self.role_graph:
        #     self.data_list = return_role_graph(self.opt.MODE, self.data_list)
        
        self.normalizer_ih = MotionNormalizer()
        self.mean_ih = self.normalizer_ih.motion_mean
        self.std_ih = self.normalizer_ih.motion_std
        print("total in Interhuman dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]


        if 'indi_text' in os.environ.get('UMF'):
            text_individual1 = random.choice(data["texts_individual1"]).strip()
            text_individual2 = random.choice(data["texts_individual2"]).strip()



        ### Deal with the motion
        if self.cache:
            full_motion1, full_motion2 = self.motion_dict[motion_id]
        else:
            file_path1, file_path2 = self.motion_dict[motion_id]
            motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)
            motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)
            if swap:
                full_motion1 = motion1_swap
                full_motion2 = motion2_swap
            else:
                full_motion1 = motion1
                full_motion2 = motion2

        length = full_motion1.shape[0]
        if length > self.max_length:
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        if np.random.rand() > 0.5:
            motion1, motion2 = motion2, motion1
        motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)

        gt_motion1 = motion1
        gt_motion2 = motion2

        #ego_motion = motion1

        from data_loaders.interhuman.interhuman import MotionNormalizerTorch, MotionNormalizer
        # mean.shape [262]

        #gt_motion1 = (gt_motion1 - self.mean_ih) / self.std_ih 
        #gt_motion2 = (gt_motion2 - self.mean_ih) / self.std_ih 
        gt_length = len(gt_motion1)
        if 'relative_cond' in os.environ.get('UMF'):
            pos1 = gt_motion1[:, :21*3]
            pos2 = gt_motion2[:, :21*3]
            if np.random.rand() > 0.5:
                pos1, pos2 = pos2, pos1
            relative_position_all = pos1 - pos2 # [73, 63]
            relative_orientation_all = qbetween_np(pos1.reshape(-1, 21, 3),  pos2.reshape(-1, 21, 3)).reshape(-1, 21*4) # [73, 21, 4]
            relative = np.concatenate((relative_orientation_all, relative_position_all), -1) # [-1, 147] -> 84 + 63
            if gt_length < self.max_gt_length:
                padding_len = self.max_gt_length - gt_length
                D = relative.shape[1]
                padding_zeros = np.zeros((padding_len, D))
                relative = np.concatenate((relative, padding_zeros), axis=0) # [299, 147]



        
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)


        assert len(gt_motion1) == self.max_gt_length
        assert len(gt_motion2) == self.max_gt_length

        # if np.random.rand() > 0.5:
        #     gt_motion1, gt_motion2 = gt_motion2, gt_motion1

        gt_motion1 = (gt_motion1 - self.mean_ih) / self.std_ih 
        gt_motion2 = (gt_motion2 - self.mean_ih) / self.std_ih
        
        motion = np.concatenate([gt_motion1, gt_motion2], 1)
        #motion = gt_motion1
        m_length = length
        

        # if m_length>self.max_length:
        #     print('None')

        ### Deal with the text
        # if self.role_graph:
        #     text_data = random.choice(data["texts"])
        #     #print(data["texts"])
        #     if isinstance(text_data, str):
        #         print('text')
        #     text = text_data['caption']
            
        #     V, entities, relations = [], [], text_data["relations"]

        #     for i in text_data["V"]:
        #         V.append(text_data["V"][i]['spans'])

        #     for i in text_data["entities"]:
        #         entities.append(text_data["entities"][i]['spans'])

        #     return (
        #             None,
        #             None,
        #             text, #caption
        #             None, #sent_len,
        #             motion, # gt_motion1: (300, 262) # 262 = 22*3 + 21*6
        #             gt_length+1,
        #             None, #"_".join(tokens),
        #             V,
        #             entities,
        #             relations
        #             )

        text = random.choice(data["texts"]).strip()
        if 'indi_text' in os.environ.get('UMF'): 
            return None, None, text, None, motion, len(motion1), None, text_individual1, text_individual2 
        if 'relative_cond' in os.environ.get('UMF'): 
            return None, None, text, None, motion, len(motion1), None, name, relative
        else:
            return None, None, text, None, motion, len(motion1), None, name
    
        
        #return name, text, gt_motion1, gt_motion2, gt_length
        # t2m only need index 2,4,5,6 -> caption, motion, m_length, 
        
        #caption -> [0:3]
        #text: 'two people are standing shoulder to shoulder. one person points to the distance with their left hand, and then leads the other person forward by pulling them gently.'





