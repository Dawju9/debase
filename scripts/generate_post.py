import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
from datetime import datetime

# Wczytanie modelu i tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# Prompt do generowania treści
prompt = "Napisz bloga na temat wpływu sztucznej inteligencji na nowoczesne technologie."

# Generowanie treści
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs.input_ids, max_length=300, temperature=0.7, top_p=0.95)

# Przetwarzanie wyniku
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Zapisywanie posta do pliku
posts_dir = "posts"
os.makedirs(posts_dir, exist_ok=True)
file_name = os.path.join(posts_dir, f"post_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

with open(file_name, "w") as file:
    file.write(generated_text)

print(f"Post zapisany w: {file_name}")
