import torch

def vsumm_collator(x):
	data = [batch_item for batch_item in x]
	features = [item[0] for item in data]
	labels = [item[1] for item in data]
	text = [item[2] for item in data]
	data = [item[3] for item in data]

	features = torch.stack(features)
	labels = torch.stack(labels)

	return features, labels, text, data
