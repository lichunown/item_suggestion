import torch
from transformers import AutoModelForCausalLM
model_base=AutoModelForCausalLM.from_pretrained("./gemma-2b")
from peft import (
    PeftModel
)
model=PeftModel.from_pretrained(model_base,'./saved_models')
model.eval()
model=model.merge_and_unload()

# Save the merged model to a directory.
model.save_pretrained("./merged_gemma-2b",save_function=torch.save)
load_model=AutoModelForCausalLM.from_pretrained("./merged_gemma-2b")