from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./model/")

with init_empty_weights():
  model = AutoModelForCausalLM.from_config(config)

codeverb_map = infer_auto_device_map(model)

print(codeverb_map)