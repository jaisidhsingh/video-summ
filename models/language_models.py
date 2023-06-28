from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch


class SummaryGenerator():
	def __init__(self):
		self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
		self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

	def predict(self, input_text):
		inputs = self.tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)

		output_ids = self.model.generate(inputs["input_ids"], num_beams=2, min_length=5, max_length=20)
		summary = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		return summary


class KeyWordGenerator():
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
		self.model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")

	def predict(self, input_text):
		inputs = self.tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)

		output_ids = self.model.generate(inputs["input_ids"], num_beams=2, min_length=5, max_length=20)
		keys = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		return keys.split(";")

def testing():
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

	embedding_dim = gpt2.transformer.wte.weight.shape[1]
	inputs = torch.randn((2, 10, embedding_dim))

	outputs = gpt2(inputs_embeds=inputs)
	logits = outputs.logits
	predicted_ids = logits.argmax(dim=-1)
	generated_text = tokenizer.batch_decode(predicted_ids)
	print(generated_text)

