import torch
import random
import torch.nn as nn
from transformers import ElectraModel, ElectraForPreTraining, ElectraForMaskedLM

from transformers import ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.generator = ElectraForMaskedLM.from_pretrained(pretrained_model_name_or_path='google/electra-base-generator').to(self.device)
        self.discriminator = ElectraForPreTraining.from_pretrained(pretrained_model_name_or_path='google/electra-base-discriminator').to(self.device)
        # share embeddings weights between generator and discriminator
        self.generator.base_model.embeddings = self.discriminator.base_model.embeddings

    def forward(self, tokens, tokens_with_mask, generator_mask, discriminator_mask):
        tokens = tokens.to(self.device)
        tokens_with_mask = tokens_with_mask.to(self.device)
        generator_mask = generator_mask.to(self.device)
        discriminator_mask = discriminator_mask.to(self.device)

        generator_out = self.generator(input_ids=tokens_with_mask, attention_mask=generator_mask)
        generator_logits = generator_out.logits
        generator_yhat = generator_logits.argmax(-1)

        discriminator_y = torch.where(generator_yhat == tokens, torch.zeros(tokens.shape).to(self.device), torch.ones(tokens.shape).to(self.device))

        discriminator_out = self.discriminator(input_ids=generator_yhat, attention_mask=discriminator_mask)
        discriminator_logits = discriminator_out.logits

        return generator_logits, tokens, discriminator_logits, discriminator_y
