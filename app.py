# app.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and tokenizer
# You can change 'gpt2' to other models like 'gpt2-medium', 'gptj-6B' (if resources allow and setup is adjusted)
# For GPT-J, you'd use: from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = 'EleutherAI/gpt-j-6B'
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token_id for open-ended generation if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Function to generate responses
def generate_response(prompt_text):
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device) # Ensure mask is on the same device

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=200,  # Max length of the generated text including the prompt
        num_return_sequences=1,
        no_repeat_ngram_size=2, # Helps prevent repetitive phrases
        temperature=0.7,        # Controls randomness: lower is more deterministic
        top_k=50,               # Considers the top k tokens by probability
        top_p=0.9,              # Nucleus sampling: considers tokens with cumulative probability >= p
        pad_token_id=tokenizer.eos_token_id # Important for some models
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/generate', methods=['POST'])
def handle_generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided. Please send a JSON object with a 'prompt' key."}), 400
    
    user_prompt = data['prompt']
    try:
        generated_text = generate_response(user_prompt)
        # The generated text usually includes the prompt.
        # If you want only the newly generated part, you might need to process 'generated_text'.
        # For example, if generated_text.startswith(user_prompt):
        #   actual_response = generated_text[len(user_prompt):].strip()
        # else:
        #   actual_response = generated_text
        return jsonify({"response": generated_text})
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during generation: {str(e)}")
        return jsonify({"error": "Failed to generate text.", "details": str(e)}), 500

if __name__ == '__main__':
    # Consider environment variables for host and port in production
    app.run(debug=True, host='0.0.0.0', port=5000)