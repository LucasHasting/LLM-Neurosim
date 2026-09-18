from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle

#downlaod model if needed
#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "ModelCloud/tinyllama-15M-stories"

#retrieve model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

#download model
with open("model.pkl", "wb") as file:
    pickle.dump(model.state_dict(), file)

#show weight names and shape
for i in model.state_dict().keys():
    print(f"{i}: {model.state_dict()[i].shape}")

#show token info
#print(tokenizer)
