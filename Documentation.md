# Documentation

## Project Overview

This project fine-tunes Large Language Models (LLMs) to predict Amazon product prices based on product descriptions. It demonstrates several key machine learning concepts:

- **Base model evaluation** - Testing untrained models
- **Fine-tuning** - Training models on specific tasks
- **Quantization** - Reducing memory usage
- **LoRA/QLoRA** - Efficient fine-tuning techniques
- **Model evaluation** - Measuring performance

## Part 1: Understanding the Libraries and Imports

### Core ML Libraries
```python
import torch  # PyTorch - main deep learning framework
import transformers  # Hugging Face library for pre-trained models
from transformers import AutoModelForCausalLM, AutoTokenizer
```

**What they do:**
- `torch`: PyTorch is the main framework for building neural networks
- `transformers`: Hugging Face's library containing thousands of pre-trained models
- `AutoModelForCausalLM`: Automatically loads language models for text generation
- `AutoTokenizer`: Converts text to numbers (tokens) that models can understand

### Specialized Fine-tuning Libraries
```python
from peft import LoraConfig, PeftModel  # Parameter Efficient Fine-Tuning
from trl import SFTTrainer, SFTConfig   # Supervised Fine-Tuning
import bitsandbytes  # Quantization (reducing model size)
```

**What they do:**
- `peft`: Makes fine-tuning more memory-efficient by only training small parts of the model
- `trl`: Provides easy-to-use trainers for fine-tuning
- `bitsandbytes`: Reduces model memory usage by using lower precision numbers

### Data and Utilities
```python
from datasets import load_dataset  # Loading datasets from Hugging Face
import wandb  # Weights & Biases for experiment tracking
import matplotlib.pyplot as plt  # Plotting results
```

## Part 2: Understanding Tokenization

### What is a Tokenizer?

A tokenizer converts human text into numbers that AI models can process:

```python
# Example of tokenization
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
text = "The price is $100"
tokens = tokenizer.encode(text)
print(tokens)  # [1, 791, 3430, 374, 400, 1041]
```

**Key concepts:**
- Each word or part of a word becomes a number (token)
- Models work with these numbers, not raw text
- Different models have different tokenizers

### Why Investigate Tokenizers?

The code investigates how different models tokenize numbers:

```python
def investigate_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for number in [0, 1, 10, 100, 999, 1000]:
        tokens = tokenizer.encode(str(number))
        print(f"Tokens for {number}: {tokens}")
```

This is crucial because:
- Some models might split "100" into multiple tokens
- Others might treat it as one token
- This affects how well the model can predict prices

## Part 3: Model Quantization

### What is Quantization?

Quantization reduces the precision of model weights to save memory:

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit precision instead of 32-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation precision
)
```

**Memory savings:**
- Normal model: 32 bits per parameter = ~32GB for 8B parameter model
- 4-bit quantized: 4 bits per parameter = ~4GB for same model
- Slight accuracy loss but massive memory savings

## Part 4: Understanding LoRA (Low-Rank Adaptation)

### What is LoRA?

LoRA is a technique that fine-tunes only small parts of a model instead of the entire model:

```python
lora_config = LoraConfig(
    r=32,  # Rank - how complex the adaptation can be
    lora_alpha=64,  # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which parts to fine-tune
    lora_dropout=0.1,  # Regularization
)
```

**Why LoRA works:**
- Instead of changing all 8 billion parameters
- It adds small "adapter" layers with only ~millions of parameters
- Much faster and requires less memory
- Can achieve similar results to full fine-tuning

## Part 5: Data Preparation and Training

### Dataset Format

The dataset contains product descriptions and prices:
```python
{
    "text": "Product: Laptop\nDescription: High-performance gaming laptop...\n\nPrice is $1299.99",
    "price": 1299.99
}
```

### Data Collator for Completion

This is crucial - we only want to train the model to predict the price, not the description:

```python
from trl import DataCollatorForCompletionOnlyLM
response_template = "Price is $"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
```

**What this does:**
- Masks the product description during training
- Only computes loss on tokens after "Price is $"
- Model learns to predict price given context, not memorize descriptions

### Training Configuration

```python
train_parameters = SFTConfig(
    num_train_epochs=1,  # How many times to see the data
    per_device_train_batch_size=4,  # How many examples per batch
    learning_rate=1e-4,  # How fast to learn
    warmup_ratio=0.03,  # Gradual learning rate increase
    max_seq_length=182,  # Maximum input length
)
```

## Part 6: Evaluation Metrics

### Key Metrics Used

1. **Average Error**: Simple dollar difference
2. **RMSLE (Root Mean Squared Log Error)**: Penalizes large errors more
3. **Hit Rate**: Percentage of predictions within 20% or $40

```python
def color_for(self, error, truth):
    if error < 40 or error/truth < 0.2:  # Good prediction
        return "green"
    elif error < 80 or error/truth < 0.4:  # Okay prediction
        return "orange"
    else:  # Poor prediction
        return "red"
```

### Why RMSLE?

RMSLE treats relative errors equally:
- Being off by $10 on a $50 item is worse than being off by $10 on a $500 item
- Formula: `sqrt(mean((log(truth+1) - log(prediction+1))^2))`

## Part 7: Advanced Prediction Techniques

### Simple Prediction
```python
def model_predict(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=3)
    response = tokenizer.decode(outputs[0])
    return extract_price(response)
```

### Improved Prediction with Top-K Sampling
```python
def improved_model_predict(prompt):
    # Get probability distribution over next tokens
    outputs = model(inputs)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = F.softmax(next_token_logits, dim=-1)
    
    # Get top 3 most likely tokens
    top_prob, top_token_id = next_token_probs.topk(3)
    
    # Weight predictions by probability
    weighted_average = sum(price * prob for price, prob in zip(prices, probs))
```

This is more robust because it considers multiple possible predictions weighted by confidence.

## Part 8: OpenAI Fine-tuning

### Data Format for OpenAI

OpenAI requires a specific JSON format:
```python
def messages_for(item):
    return [
        {"role": "system", "content": "You estimate prices of items."},
        {"role": "user", "content": item.description},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
    ]
```

### JSONL Format
```python
def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        result += '{"messages": ' + json.dumps(messages) + '}\n'
    return result
```

JSONL = JSON Lines, where each line is a separate JSON object.

## Part 9: Key Programming Patterns

### 1. Configuration Management
```python
# Constants at the top make code maintainable
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
```

### 2. Error Handling in Data Processing
```python
def extract_price(s):
    if "Price is $" in s:
        contents = s.split("Price is $")[1].replace(',', '')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0
    return 0
```

### 3. Object-Oriented Testing
```python
class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        # ... initialize other attributes
    
    def run_datapoint(self, i):
        # Test single example
    
    def report(self):
        # Generate final report
```

### 4. Device Management
```python
# Automatically use GPU if available
device_map="auto"
inputs.to("cuda")  # Move to GPU
```

## Part 10: Common Issues and Solutions

### 1. Memory Management
- Use quantization to reduce memory usage
- Use gradient accumulation to simulate larger batch sizes
- Clear cache: `torch.cuda.empty_cache()`

### 2. Training Stability
- Use warmup to gradually increase learning rate
- Use gradient clipping to prevent exploding gradients
- Monitor loss curves in Weights & Biases

### 3. Data Quality
- Ensure consistent formatting
- Handle edge cases in price extraction
- Validate data before training

## Part 11: How to Write This Code Yourself

### Step 1: Start with Simple Evaluation
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # Start with smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Simple prediction function
def predict(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0])
```

### Step 2: Add Data Loading
```python
from datasets import Dataset

# Create simple dataset
data = [
    {"text": "Laptop description...", "price": 999.99},
    {"text": "Phone description...", "price": 599.99},
]
dataset = Dataset.from_list(data)
```

### Step 3: Add Quantization
```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quant_config
)
```

### Step 4: Add LoRA Fine-tuning
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
```

### Step 5: Add Training Loop
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
)
trainer.train()
```

## Part 12: Best Practices

### 1. Experiment Tracking
- Always use Weights & Biases or similar
- Track hyperparameters, metrics, and model outputs
- Save model checkpoints regularly

### 2. Code Organization
- Separate configuration, data loading, training, and evaluation
- Use classes for complex functionality
- Write reusable functions

### 3. Testing
- Test on small datasets first
- Validate data preprocessing steps
- Check model outputs manually

### 4. Documentation
- Comment complex sections
- Use descriptive variable names
- Document hyperparameter choices

## Next Steps to Master This

1. **Start Small**: Try fine-tuning a small model (like DistilBERT) on a simple task
2. **Understand Each Component**: Implement tokenization, quantization, and LoRA from scratch
3. **Read Papers**: Study the original LoRA and QLoRA papers
4. **Practice**: Try different models, datasets, and hyperparameters
5. **Join Communities**: Hugging Face forums, Reddit r/MachineLearning

The key is understanding that modern LLM fine-tuning is about:
- Efficient memory usage (quantization)
- Parameter-efficient training (LoRA)
- Proper data preparation (masking, formatting)
- Systematic evaluation (metrics, visualization)

Each piece builds on the others to create a complete machine learning pipeline!