from torchvision import transforms, models
import torch
from PIL import Image
from pathlib import Path
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import pdb
import warnings
import sys
import os

cwd = os.getcwd()
module2add = '/'.join(cwd.split("/")[:-1])
sys.path.append(module2add)

from models.feature_extractor import FeatureExtractor
from configs.global_configs import configs as global_configs

from time import perf_counter
warnings.simplefilter("ignore")

# cmd-line args for ease of access
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tvsumm', type=str)
args = parser.parse_args()

# load in helpers
video_dir = os.path.join(global_configs.dataset_dir, args.dataset, "videos")
vsumm_data_path = f'../../helpers/h5_helpers/eccv16_dataset_{args.dataset}_google_pool5.h5'
"""
preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.
"""
vsumm_data = h5py.File(vsumm_data_path)


class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fea_net = FeatureExtractor()

def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8)
        recall = overlap / (true_sum + 1e-8)
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)


def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    # tqdm.write('Overlap: '+str(overlap_arr))
    # tqdm.write('True summary n_key: '+str(true_sum_arr))
    # tqdm.write('Oracle smmary n_key: '+str(oracle_sum))
    # tqdm.write('Final F-score: '+str(best_fscore))
    return oracle_summary


def video2fea(video_path, model, helper):
    video = cv2.VideoCapture(video_path.as_uri())
    idx = video_path.as_uri().split('.')[0]
    idx = idx.split('/')[-1]

    # tqdm.write('Processing video '+idx)
    
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length//320

    fea = []
    text_fea = []
    label = []

    name = helper[idx]["id"]
    usr_sum_arr = vsumm_data[name]['user_summary'][()]
    usr_sum = get_oracle_summary(usr_sum_arr) 
    cps = vsumm_data[name]['change_points'][()]
    n_frame_per_seg = vsumm_data[name]['n_frame_per_seg'][()]
    text = helper[idx]["text"]
    text_fea.append(model.encode_text(text).cpu())

    i = 0
    success, frame = video.read()
    while success:
        if (i+1) % ratio == 0:
            fea.append(model.encode_image(Image.fromarray(frame)).cpu())
            try:
                label.append(usr_sum[i])
            except:
                pdb.set_trace()
        i += 1
        success, frame = video.read()
    fea = torch.stack(fea)
    text_fea = torch.stack(text_fea)
    fea = fea[:320]
    label = label[:320]
    
    v_data = {}
    v_data['id'] = name
    v_data['image_features'] = fea.squeeze(1)
    v_data["text_features"] = text_fea.view((1, 512))
    v_data['label'] = label
    v_data['length'] = len(usr_sum)
    v_data['change_points'] = cps
    v_data['n_frame_per_seg'] = n_frame_per_seg
    v_data['picks'] = [ratio*i for i in range(320)]
    v_data['user_summary'] = usr_sum_arr
    helper[idx].update(v_data)

    if fea.shape[0] != 320 or len(label) != 320:
        print('error in video ', idx, fea.shape[0], len(label))


def make_dataset(video_dir):
    helper = torch.load(f"../../datasets/{args.dataset}/helpers/{args.dataset}_database.pt")
    video_dir = Path(video_dir).resolve()
    video_list = list(video_dir.glob('*.mp4'))
    video_list.sort()

    for video_path in tqdm(video_list, desc='Video', ncols=80, leave=False):
        start_time = perf_counter()
        video2fea(video_path, fea_net, helper)
    
    print(" ")
    for k, v in helper.items():
        v.pop("features")
        helper[k] = v

    print("  ")
    torch.save(helper, f"../../datasets/{args.dataset}/helpers/{args.dataset}_database.pt")
    end_time = perf_counter()
    print("Took time:", end_time - start_time)
    
if __name__ == '__main__':
    make_dataset(video_dir)

vsumm_data.close()