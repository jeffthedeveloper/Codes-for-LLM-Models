import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carregar o modelo pré-treinado e o tokenizador
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Função para gerar respostas
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Exemplo de uso
user_input = "Qual é o clima hoje?"
response = generate_response(user_input)
print(response)
