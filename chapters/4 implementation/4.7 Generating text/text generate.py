import torch
import torch.nn as nn

def Yapp(model,idx, max_new_tokens, context_size):
	for _ in range(max_new_tokens):
		idx_cond = idx[:, -context_size:]  # Crop context if needed
		with torch.no_grad():
			logits = model(idx)
		

		logits = logits[:, -1, :]  # Focus on last token
		probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
		idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Greedily pick the next token
		idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence

	return idx
