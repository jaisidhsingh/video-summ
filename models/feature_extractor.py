import torch
import torch.nn as nn
import clip
from copy import deepcopy


class FeatureExtractor():
	def __init__(self):
		model, preprocess = clip.load("ViT-B/32", device="cuda")
		self.model = deepcopy(model)
		self.model.cuda()
		self.transforms = preprocess

		for param in self.model.parameters():
			param.requires_grad = False
		
		self.model.eval()

	def encode_image(self, image):
		with torch.no_grad():
			img_tensor = self.transforms(image).unsqueeze(0).cuda()
			features = self.model.encode_image(img_tensor)
		return features
	
	def encode_text(self, text):
		with torch.no_grad():
			text = clip.tokenize([text]).cuda()
			features = self.model.encode_text(text)
		return features
