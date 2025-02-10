from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import GenerationProfile, QueryMode as QM, OutputMode as OM
from functions import cosine_similarity, norm, \
                    degree_angle, rad_angle

models = [
    "meta-llama/Llama-3.2-1B",
    "facebook/layerskip-llama3.2-1B"
]

MODEL = models[1]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

prompt = "Once upon a time in a futuristic city,"

inputs = tokenizer(prompt, return_tensors="pt")

# output['hidden_states'] -> this is a tuple of tensors, for each new token generated.
# output['hidden_states'][i] -> tuple of tensors of the hidden states -> [1,1,2048]
output = model.generate(
    **inputs,
    output_hidden_states=True,
    return_dict_in_generate=True,
    min_new_tokens = 25,
    max_length = 40
)

generation_profile = GenerationProfile(inputs, output, tokenizer)
generation_profile.print_summary()
generation_profile.query(cosine_similarity, QM.PAIRWISE, OM.PLOT)
