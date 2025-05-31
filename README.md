# IA Generativa com Flask e GPT-2

Este projeto demonstra a integração de um modelo de linguagem grande (LLM) pré-treinado, especificamente GPT-2 (com a flexibilidade para usar outros como GPT-J), com um servidor web Flask para fornecer uma API de geração de texto. Ele serve como um exemplo prático de como implantar modelos da biblioteca Hugging Face Transformers em uma aplicação web.

## 📝 Descrição

O objetivo principal é criar uma interface simples e acessível para interagir com um modelo de IA generativa.  Usuários podem enviar um *prompt* (texto inicial) para um endpoint da API e receber uma continuação de texto gerada pelo modelo. Este projeto utiliza o `GPT2LMHeadModel` e `GPT2Tokenizer` da Hugging Face como base para a funcionalidade de geração de texto.

## 🔄 Contexto da Importação e Adaptação

* **Origem do Código**: O núcleo da geração de texto é baseado no modelo `GPT2LMHeadModel` e no `GPT2Tokenizer` da biblioteca Hugging Face Transformers. 
* **Principais Adaptações Realizadas**:
    * O modelo GPT-2 pré-treinado é carregado e encapsulado em uma função de geração de resposta.
    * Uma aplicação Flask (`app.py`) foi desenvolvida para expor essa funcionalidade através de um endpoint HTTP (`/generate`). 
    * O script `Training-and-Fine-TuningLLM.py` é fornecido como uma ferramenta separada para usuários que desejam treinar ou ajustar modelos GPT-2 em seus próprios conjuntos de dados.

## ✨ Funcionalidades Principais

* **Geração de Texto Dinâmica**: Utiliza o modelo GPT-2 para gerar continuações de texto coesas e contextualmente relevantes com base em um *prompt* fornecido pelo usuário.
* **API Web com Flask**: Oferece um endpoint `/generate` (via método POST) que aceita um JSON com um campo "prompt" e retorna o texto gerado. 
* **Flexibilidade do Modelo**: Embora configurado com GPT-2, o código pode ser adaptado para usar outros modelos da Hugging Face (como GPT-J, mencionado no seu portfólio).

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**
* **Flask**: Microframework web para criar a API.
* **Hugging Face Transformers**: Biblioteca para carregar e usar modelos de linguagem pré-treinados.
* **PyTorch**: Backend para o modelo da Hugging Face (Transformers pode também usar TensorFlow).

## 📂 Estrutura do Projeto

```plaintext
codesForLLM/
├── app.py                            # Aplicação principal Flask com a API de geração
├── Training-and-Fine-TuningLLM.py    # Script para treinamento e fine-tuning do modelo (opcional)
├── requirements.txt                  # Dependências do projeto Python
├── LICENSE.md                        # Arquivo de licença do projeto (MIT)
└── README.md                         # Este arquivo
```

## ⚙️ Pré-requisitos

Antes de começar, certifique-se de que você tem o seguinte instalado:
* Python 3.8 ou superior
* `pip` (gerenciador de pacotes Python)
* Git (para clonar o repositório)

## 🚀 Instalação

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/jeffthedeveloper/codesForLLM.git](https://github.com/jeffthedeveloper/codesForLLM.git)
    cd codesForLLM
    ```
    (Nota: O URL do repositório é baseado no seu portfólio.)

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:** 
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Uso

1.  **Execute a aplicação Flask:**
    Certifique-se de que o arquivo principal da aplicação Flask (sugerido como `app.py` acima) está na raiz do projeto.
    ```bash
    python app.py
    ```
    A API estará acessível em `http://localhost:5000` por padrão.

2.  **Use a API para gerar texto:**
    Envie uma requisição POST para `http://localhost:5000/generate` com um corpo JSON contendo o prompt. 

    **Exemplo usando `curl`:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{"prompt": "Em um futuro distante, a humanidade descobriu"}' \
         http://localhost:5000/generate
    ```

    **Resposta esperada (exemplo):**
    ```json
    {
      "response": "Em um futuro distante, a humanidade descobriu uma nova forma de energia que poderia revolucionar a maneira como vivemos."
    }
    ```
    (O texto exato irá variar devido à natureza do modelo generativo.)

3.  **(Opcional) Usando o script de Treinamento/Fine-Tuning:**
    O arquivo `Training-and-Fine-TuningLLM.py` pode ser usado para customizar modelos GPT-2.
    ```python
    # Training-and-Fine-TuningLLM.py
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

    # Carregar o modelo pré-treinado e o tokenizador
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: # Adicionar token de padding se não existir
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


    # Função para preparar o dataset (exemplo com um arquivo de texto)
    def load_dataset(file_path, tokenizer, block_size=128):
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size
        )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, # MLM é False para GPT-2 (geração causal)
    )

    # Carregue seu conjunto de dados de treinamento e validação
    # Substitua 'train.txt' e 'eval.txt' pelos seus arquivos de dados
    # train_dataset = load_dataset('train.txt', tokenizer)
    # eval_dataset = load_dataset('eval.txt', tokenizer) # Opcional, mas recomendado

    # Exemplo de como seria o objeto train_dataset (você precisa fornecer os dados)
    # train_dataset = ... # Carregue seu conjunto de dados formatado
    # print("Certifique-se de carregar seu train_dataset aqui.")


    training_args = TrainingArguments(
        output_dir='./results_finetuned_gpt2', # Diretório de saída para o modelo treinado e checkpoints
        overwrite_output_dir=True,
        num_train_epochs=1, # Número de épocas de treinamento (ajuste conforme necessário)
        per_device_train_batch_size=2, # Tamanho do batch por dispositivo durante o treinamento
        save_steps=10_000, # Salvar checkpoints a cada X passos
        save_total_limit=2, # Manter apenas os últimos X checkpoints
        prediction_loss_only=True, # Se você só quer a perda para avaliação
        # evaluation_strategy="epoch", # Avaliar a cada época (requer eval_dataset)
        learning_rate=5e-5, # Taxa de aprendizado
        # per_device_eval_batch_size=2, # Tamanho do batch para avaliação
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        # train_dataset=train_dataset, # Forneça seu dataset de treino
        # eval_dataset=eval_dataset,   # Forneça seu dataset de avaliação (opcional)
    )

    # Para iniciar o treinamento (descomente e forneça os datasets):
    # if train_dataset:
    #     print("Iniciando o treinamento...")
    #     trainer.train()
    #     trainer.save_model("./results_finetuned_gpt2/final_model") # Salvar o modelo final
    #     tokenizer.save_pretrained("./results_finetuned_gpt2/final_model") # Salvar o tokenizador
    # else:
    #     print("Dataset de treinamento não fornecido. O treinamento não será iniciado.")

    print("Script de treinamento/fine-tuning configurado. Forneça os dados e descomente as seções de treinamento para executar.")
    ```
    **Nota**: Para usar este script, você precisará preparar seus dados de treinamento (por exemplo, em arquivos `.txt`) e ajustar os caminhos e parâmetros conforme necessário.

## 📜 Licença do Projeto Atual

Este projeto é distribuído sob a Licença MIT. Veja o arquivo `LICENSE.md` para mais informações.
A biblioteca Hugging Face Transformers é licenciada sob Apache License 2.0. 

## 🤝 Como Contribuir

Contribuições são bem-vindas! Se este fosse um projeto aberto à comunidade:

1.  Faça um Fork do projeto.
2.  Crie uma Branch para sua Feature (`git checkout -b feature/MinhaNovaFeature`). 
3.  Faça o Commit de suas mudanças (`git commit -m 'Adiciona MinhaNovaFeature'`). 
4.  Faça o Push para a Branch (`git push origin feature/MinhaNovaFeature`). 
5.  Abra um Pull Request.

## 🧑‍💻 Autores e Agradecimentos

* **Autor do Projeto**: Jefferson Firmino Mendes / On My Own 
* **Agradecimentos**: À equipe da Hugging Face pelo desenvolvimento da biblioteca `transformers` e pela disponibilização de modelos como o GPT-2.

## 🔗 Link do Projeto (Portfólio)

[https://github.com/jeffthedeveloper/Codes-for-LLM-Models](https://github.com/jeffthedeveloper/Codes-for-LLM-Models)
```
