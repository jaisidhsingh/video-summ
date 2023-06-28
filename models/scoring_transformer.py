import torch
import torch.nn as nn
from .positional_encoding import *


class ScoringTransformer(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.embedding_dim = cfg.embedding_dim
		self.num_heads = cfg.num_heads
		self.num_blocks = cfg.num_blocks

		if self.cfg.encoding_type == 'cosine':
			self.positional_encoding = CosinePositionalEncoding(cfg.embedding_dim, cfg.seq_len)
		else:
			self.positional_encoding = LearnablePositionalEncoding(cfg.embedding_dim, cfg.seq_len)

		self.encoder_blocks = []
		self.decoder_blocks = []
		for _ in range(cfg.num_blocks):
			encoder_block = nn.TransformerEncoderLayer(
				d_model=self.embedding_dim,
				nhead=self.num_heads,
				batch_first=True
			)
			decoder_block = nn.TransformerDecoderLayer(
				d_model=self.embedding_dim,
				nhead=self.num_heads,
				batch_first=True
			)
			self.encoder_blocks.append(encoder_block)
			self.decoder_blocks.append(decoder_block)
		
		self.encoder_blocks = nn.Sequential(*self.encoder_blocks)
		self.decoder_blocks = nn.Sequential(*self.decoder_blocks)
		
		self.activation = nn.Sigmoid()
		self.fc_out = nn.Linear(cfg.embedding_dim, 1)

	def forward(self, x):
		x = self.positional_encoding(x)
		hidden = torch.zeros_like(x).to(x.device)

		# encoder_outputs = []
		for i in range(self.num_blocks):
			hidden = self.encoder_blocks[i](x + hidden)
			# encoder_outputs.append(hidden)

		# encoder_outputs.reverse()
		# decoder_hidden = torch.zeros_like(hidden).to(x.device)

		for i in range(self.num_blocks):
			# decoderhidden = hidden + encoder_outputs[i]
			hidden = self.decoder_blocks[i](x, hidden)

		features = hidden	
		scores = self.fc_out(hidden)
		return self.activation(scores).squeeze(2), features
