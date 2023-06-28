import os
import sys
import torch
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

cwd = os.getcwd()
module2add = '/'.join(cwd.split("/")[:-1])
sys.path.append(module2add)

from configs.training_configs import cfg as training_config
from configs.global_configs import cfg as global_config
from configs.data_configs import cfg as data_config
from configs.model_configs import cfg as model_config
from utils.eval import *


def clip_loss(image_features, text_features):
	ce_loss = torch.nn.CrossEntropyLoss()
	loss = 0

	image_features = image_features / image_features.norm(dim=1, keepdim=True)
	text_features = text_features / text_features.norm(dim=1, keepdim=True)

	logits_per_image = 100 * image_features @ text_features.t()
	logits_per_text = logits_per_image.t()

	ground_truth = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=global_config.device)
	loss += ce_loss(logits_per_image, ground_truth)
	loss += ce_loss(logits_per_text, ground_truth)
	loss = loss / 2
	return loss


def train_scoring_model(
		args, epoch,
		training_config, global_config,
		scoring_model, 
		train_loader, 
		optimizer, 
		criterion_cls,
		criterion_recon, 
		scheduler,
		use_clip_loss=True
	):
	scoring_model.train()
	running_loss = 0
	batch_counter = 0
	# start with batches
	for data in train_loader:
		features = data['features'].float().squeeze(2).to(args.device)
		labels = data['labels'].float().to(args.device)
		# zero out the gradients accumulated
		optimizer.zero_grad()
		# forward pass
		scores, output_features = scoring_model(features)

		if use_clip_loss:
			_, indices = scores.topk(k=training_config.clip_loss_topk, dim=-1)
			key_features = features[indices].view((training_config.clip_loss_topk, 512)) # sanity check
			text_features_for_loss = torch.cat([text_features for _ in range(training_config.clip_loss_topk)])
			loss_clip = clip_loss(key_features, text_features_for_loss)			

		else:
			loss_clip = 0

		loss_cls = training_config.cls_weight*criterion_cls(scores, labels) 
		loss_recon = training_config.recon_weight*criterion_recon(output_features, features)

		loss = loss_cls + loss_recon + loss_clip
		running_loss += loss.item()
		# backward pass
		loss.backward()
		optimizer.step()
		scheduler.step()
		batch_counter += 1

	if (epoch + 1) % training_config.save_point == 0:
		ckpt_save_path = os.path.join(
			global_config.ckpt_dir, f"{args.dataset_name}_scoring_model_epoch_{epoch+1}.pt"
		)
		data2save = {
			'model': scoring_model.state_dict(),
			'optimizer': optimizer.state_dict()
		}
		torch.save(data2save, ckpt_save_path)
		print("Checkpoint saved")

	return running_loss

def evaluate(args, scoring_model, test_loader, criterion_cls, criterion_recon):
	scoring_model.eval()
	eval_arr = []
	data2return = {}

	running_loss = 0
	batch_counter = 0
	for data in test_loader:
		features = data['features'].float().squeeze(2).to(args.device)
		labels = data['labels'].float().to(args.device)

		with torch.no_grad():
			scores, output_features = scoring_model(features)
			loss_cls = training_config.cls_weight*criterion_cls(scores, labels)
			loss_recon = training_config.recon_weight*criterion_recon(output_features, features)
			loss = loss_cls + loss_recon
			running_loss += loss.item()
		
		pred_score = torch.softmax(scores, dim=1) # softmax across frames
		pred_score, pred_selected, pred_summary = select_keyshots(data, pred_score[0])
		true_summary_arr = data['user_summary'][0]
		pred_summary = torch.tensor(pred_summary)

		eval_res = [eval_metrics(pred_summary.numpy(), true_summary.numpy()) for true_summary in true_summary_arr]
		eval_res = np.mean(eval_res, axis=0).tolist()

		eval_arr.append(eval_res)

		video_name = data['video_name'][0]
		data2return[video_name] = {
			'evaluation': eval_res,
			'pred_score': pred_score,
			'pred_selected': pred_selected,
			'pred_summary': pred_summary,
		}
		batch_counter += 1

	data2return["evaluation_results"] = np.stack(eval_arr).mean(axis=0)
	return data2return, running_loss

def print2spaces():
	print(" ")
	print(" ")

def print2lines():
	print("-------------------------------------------------------------------------------------")
	print("-------------------------------------------------------------------------------------")