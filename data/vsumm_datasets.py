import torch
from torch.utils.data import Dataset, DataLoader
from collator_functions import vsumm_collator


class TvSummDataset(Dataset):
	def __init__(self, dataset_file, split):
		self.data = torch.load(dataset_file)
		self.video_features = []
		self.labels = []
		self.lengths = []
		self.change_points = []
		self.n_frame_per_seg = []
		self.picks = []
		self.user_summary = []
		self.video_names = []
		self.texts = []
		self.ids = []
		
		for k, v in self.data.items():
			self.video_names.append(str(k))
			self.video_features.append(v['features'])
			self.labels.append(v['label'])
			self.lengths.append(v['length'])
			self.change_points.append(v['change_points'])
			self.n_frame_per_seg.append(v['n_frame_per_seg'])
			self.picks.append(v['picks'])
			self.user_summary.append(v['user_summary'])
			self.texts.append(v['text'])
			self.ids.append(v['id'])

		self.split = split
		if split == 'train':
			self.split_slice = int(0.8*len(self.video_names))
		else:
			self.split_slice = int(-1*(0.2 * len(self.video_names)))
			
	def __len__(self):
		if self.split == 'train':
			return len(self.video_names[:self.split_slice])
		if self.split == 'test':
			return len(self.video_names[self.split_slice:])

	def __getitem__(self, idx):
		video_name = self.video_names[idx]
		features = self.video_features[idx]
		labels = torch.tensor(self.labels[idx])
		length = self.lengths[idx]
		change_points = self.change_points[idx]
		nfps = self.n_frame_per_seg[idx]
		picks = self.picks[idx]
		user_summary = self.user_summary[idx]
		text = self.texts[idx]

		data2return = {
			"video_name": video_name,
			"features": features,
			"labels": labels,
			"length": length,
			"change_points": change_points,
			"nfps": nfps,
			"picks": picks,
			"user_summary": user_summary,
			"text": text,
			"id": self.ids[idx]
		}
		return features, labels, text, data2return


class SumMeDataset(Dataset):
	def __init__(self, dataset_file, split):
		self.data = torch.load(dataset_file)
		self.video_features = []
		self.labels = []
		self.lengths = []
		self.change_points = []
		self.n_frame_per_seg = []
		self.picks = []
		self.user_summary = []
		self.video_names = []
		self.texts = []
		self.ids = []
		
		for k, v in self.data.items():
			self.video_names.append(str(k))
			self.video_features.append(v['features'])
			self.labels.append(v['label'])
			self.lengths.append(v['length'])
			self.change_points.append(v['change_points'])
			self.n_frame_per_seg.append(v['n_frame_per_seg'])
			self.picks.append(v['picks'])
			self.user_summary.append(v['user_summary'])
			self.texts.append(v['text'])
			self.ids.append(v['id'])

		self.split = split
		if split == 'train':
			self.split_slice = int(0.8*len(self.video_names))
		else:
			self.split_slice = int(-1*(0.2 * len(self.video_names)))
			
	def __len__(self):
		if self.split == 'train':
			return len(self.video_names[:self.split_slice])
		if self.split == 'test':
			return len(self.video_names[self.split_slice:])

	def __getitem__(self, idx):
		video_name = self.video_names[idx]
		features = self.video_features[idx]
		labels = torch.tensor(self.labels[idx])
		length = self.lengths[idx]
		change_points = self.change_points[idx]
		nfps = self.n_frame_per_seg[idx]
		picks = self.picks[idx]
		user_summary = self.user_summary[idx]
		text = self.texts[idx]

		data2return = {
			"video_name": video_name,
			"features": features,
			"labels": labels,
			"length": length,
			"change_points": change_points,
			"nfps": nfps,
			"picks": picks,
			"user_summary": user_summary,
			"text": text,
			"id": self.ids[idx]
		}
		return features, labels, text, data2return


def test(dataset):
	dataset_file = f"../../datasets/{dataset}/helpers/{dataset}_database.pt"
	train_dataset = TvSummDataset(dataset_file, split="train")
	train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=vsumm_collator)
	for f, l, t, d in train_loader:
		print(f.shape, l.shape, type(t), type(d))
		break

test("summe")