import os

# set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# load the model and processor
ckpt = "model_data/siglip"  # SigLIP2 base model + image processor (see README "Downloads")
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

image_path = "/path/to/image.jpg"  # <-- any test image
# load the image
image = load_image(image_path)
inputs = processor(images=[image], return_tensors="pt").to(model.device)

# run infernece
with torch.no_grad():
    vision_outputs = model.get_image_features(**inputs)

# in transformers v5, get_image_features returns a BaseModelOutputWithPooling;
# the pooled vector is the SigLIP image embedding
last_hidden_state = vision_outputs.last_hidden_state
image_embeddings = vision_outputs.pooler_output

print("Last hidden state shape:", last_hidden_state.shape)
print("Image embeddings shape:", image_embeddings.shape)
