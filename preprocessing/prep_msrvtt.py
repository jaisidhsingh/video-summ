import json
import pandas as pd
from tqdm import tqdm
import os
import sys

# add src dir path
cwd = os.getcwd()
module2add = '/'.join(cwd.split("/")[:-1])
sys.path.append(module2add)

# import local modules
from configs.global_configs import configs as global_configs
from vsumm.src.models.language_models import *


# get captions for each video in a dictionary for easy access
def prepare_video_text(anno_path, split="train_val"):
	kw_model = KeyWordGenerator()
	ts_model = SummaryGenerator()

	with open(anno_path, "r") as f:
		annos = json.load(f)

	df = pd.DataFrame(annos["sentences"])
	video_ids = df["video_id"].unique().tolist()
	video_ids.sort()

	video2texts_map = {}
	for vid in tqdm(video_ids):
		sents_for_video = df.loc[df["video_id"] == vid, "caption"]
		sents_for_video = sents_for_video.tolist()
		all_sents = ". ".join(sents_for_video)

		key_words = kw_model.predict(all_sents)
		summary = ts_model.predict(all_sents)
		video2texts_map[vid] = {
			"key_words": key_words,
			"summary": summary,
			"sentences": sents_for_video
		}

	print(f"Parsed captions for {len(video_ids)} videos. Now saving...")	

	save_dir = os.path.join(global_configs.dataset_dir, dataset2prep, "helpers")
	save_name = f"{split}_video_text.json"

	with open(os.path.join(save_dir, save_name), "w") as f:
		json.dump(video2texts_map, f)
	
	print(f"Done! Saved at {os.path.join(save_dir, save_name)}")


# driver code
if __name__ == "__main__":
	# this is the dataset for the visual-language pretraining
	dataset2prep = "msrvtt"
	trainval_anno_path = os.path.join(
		global_configs.dataset_dir, 
		dataset2prep, "train_val_videodatainfo.json"
	)
	test_anno_path = os.path.join(
		global_configs.dataset_dir, 
		dataset2prep, "test_videodatainfo.json"
	)

	prepare_video_text(trainval_anno_path, split="train_val")
	prepare_video_text(test_anno_path, split="test")
