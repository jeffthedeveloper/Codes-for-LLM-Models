from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Carregar o modelo pré-treinado e o tokenizador
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Função para tokenizar o texto de entrada
def tokenize_function(examples):
    return tokenizer(examples[""text""], return_tensors=""pt"", padding=True, truncation=True)

# Exemplo de treinamento
train_dataset = ... # Carregue seu conjunto de dados
train_dataset = train_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',          
    evaluation_strategy=""epoch"",     
    learning_rate=2e-5,              
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=8,   
    num_train_epochs=3,             
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,        
)

trainer.train()
