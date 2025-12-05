# Unsloth-Specific Dataset Requirements

## Overview

Unsloth provides optimized fine-tuning with 2x faster training and 60% less memory usage. However, it requires specific dataset formatting and considerations. This chapter covers Unsloth-specific requirements based on official documentation.

## Why Unsloth-Specific Formatting Matters

From Unsloth documentation:
> "Datasets for LLM fine-tuning must be tokenizable and follow specific structural formats"

Unsloth optimizations depend on:
- Proper chat template application
- Efficient dataset packing
- Correct field mapping
- Token-aware structuring

## Minimum Dataset Size Requirements

From Unsloth docs:
> "Effective fine-tuning requires at least 100 rows; 1,000+ rows produces optimal results. Dataset quality matters more than quantity."

**Recommendations**:
- **Minimum viable**: 100 examples
- **Production quality**: 1,000+ examples
- **Optimal performance**: 10,000+ examples

**Quality over quantity**: 100 high-quality examples > 1,000 low-quality examples

## Required Field Placeholders

### Mandatory Fields

Unsloth requires specific placeholder syntax:

**{INPUT}**: Required - The instruction or prompt field
**{OUTPUT}**: Required - The model's expected response
**{SYSTEM}**: Optional - System prompt customization

### Template Structure Example

```python
alpaca_template = """Below is an instruction. Write a response.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""
```

## Supported Dataset Formats

### 1. Raw Corpus Format

**Purpose**: Continued pretraining

**Structure**:
```json
{"text": "Plain text content from websites or books..."}
```

**Use case**: Domain adaptation, vocabulary expansion

### 2. Instruction Format (Alpaca-style)

**Structure**:
```json
{
  "instruction": "Task description",
  "input": "Optional user query",
  "output": "Expected result"
}
```

**Unsloth template mapping**:
```python
template = """### Instruction:
{instruction}

[[### Input:
{input}]]

### Response:
{output}"""
```

**Note**: Double brackets `[[...]]` indicate optional sections

### 3. ShareGPT Format

**Structure**:
```json
{
  "conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

**Key characteristics**:
- Multi-turn conversations
- "from" field: "human", "gpt", or "system"
- "value" field: message content
- Ideal for chatbot training

### 4. ChatML Format

**Structure**:
```json
{
  "messages": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

**Key characteristics**:
- OpenAI standard format
- "role" field: "user", "assistant", or "system"
- "content" field: message content
- Most widely supported

## Chat Template Application

### Checking Available Templates

```python
from unsloth.chat_templates import CHAT_TEMPLATES

# View all available templates
print(CHAT_TEMPLATES.keys())

# Common templates:
# - "llama-3"
# - "chatml"
# - "alpaca"
# - "mistral"
# - "phi"
# - "gemma"
# - "qwen"
# - "vicuna"
```

### Applying Templates

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Apply chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # Choose appropriate template
)
```

### Template Selection Guide

**Llama 3**: Use "llama-3"
```python
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
```

**ChatGPT-style**: Use "chatml"
```python
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
```

**Simple instruction**: Use "alpaca"
```python
tokenizer = get_chat_template(tokenizer, chat_template="alpaca")
```

**Mistral models**: Use "mistral"
```python
tokenizer = get_chat_template(tokenizer, chat_template="mistral")
```

## Converting ShareGPT to ChatML

From Unsloth documentation:
> "If your dataset uses the ShareGPT format with 'from'/'value' keys instead of the ChatML 'role'/'content' format, you can use the standardize_sharegpt function to convert it first"

### Conversion Function

```python
from unsloth.chat_templates import standardize_sharegpt

# Convert ShareGPT to ChatML
dataset = standardize_sharegpt(dataset)

# Conversions performed:
# "from" → "role"
# "value" → "content"
# "human" → "user"
# "gpt" → "assistant"
```

### Complete Example

```python
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

# Load ShareGPT format dataset
dataset = load_dataset("username/sharegpt-dataset")

# Convert to ChatML
dataset = dataset.map(standardize_sharegpt)

# Now dataset uses "messages" with "role" and "content"
```

## Multi-Turn Conversation Extension

From Unsloth docs:
> "Unsloth introduced the conversation_extension parameter, which selects random rows in single-turn datasets and merges them into one conversation"

### Why Multi-Turn Training?

**Benefits**:
- Improves context handling
- Better conversational flow
- More realistic interactions
- Reduced training time (fewer but longer examples)

### Enabling Conversation Extension

```python
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    conversation_extension=3,  # Merge 3 random rows into one conversation
    dataset_num_proc=4,
)
```

**How it works**:
- Randomly selects N rows from dataset
- Merges into single conversation
- Maintains alternating user/assistant pattern
- Increases effective context length

### When to Use

**Use conversation_extension when**:
- Training chatbots
- Dataset has mostly single-turn examples
- Want to improve multi-turn performance
- Have computational resources for longer sequences

**Avoid when**:
- Already have multi-turn conversations
- Task is single-shot (classification, single completion)
- Limited GPU memory (longer sequences = more memory)

## Dataset Packing

From Unsloth documentation:
> "Batches have a pre-defined sequence length - instead of assigning one batch per sample, multiple small samples can be combined in one batch, increasing efficiency"

### How Packing Works

**Without packing**:
```
Batch 1: [Example 1 (500 tokens) + Padding (1500 tokens)]
Batch 2: [Example 2 (300 tokens) + Padding (1700 tokens)]
Batch 3: [Example 3 (400 tokens) + Padding (1600 tokens)]
```

**With packing**:
```
Batch 1: [Example 1 (500) + Example 2 (300) + Example 3 (400) + Padding (800)]
```

**Efficiency gain**: ~3x in this example

### Enabling Packing

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    packing=True,  # Enable packing
    max_seq_length=2048,
)
```

### Packing Considerations

**Benefits**:
- Reduced padding waste
- Faster training (more tokens per batch)
- Better GPU utilization
- Lower training cost

**Cautions**:
- Slightly more complex loss calculation
- May need attention mask adjustments
- Debugging harder (multiple examples per batch)

**Best for**:
- Variable-length datasets
- Short to medium examples
- Resource-constrained training
- Cost optimization

## The to_sharegpt() Function

Unsloth provides a powerful function to merge dataset columns into conversation format.

### Basic Usage

```python
from unsloth import to_sharegpt

dataset = to_sharegpt(
    dataset,
    merged_prompt="""### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""",
    output_column_name="conversations"
)
```

### Syntax Rules

**Column references**: Use `{column_name}` to insert column values
```python
"{instruction}"  # Inserts the 'instruction' column
"{input}"        # Inserts the 'input' column
"{output}"       # Inserts the 'output' column
```

**Optional sections**: Use `[[optional text]]` for conditional inclusion
```python
"""[[### Input:
{input}]]"""  # Only included if 'input' column has value
```

### Complete Example

**Original dataset**:
```python
{
    "instruction": "Translate to French",
    "input": "Hello world",
    "output": "Bonjour le monde"
}
```

**Conversion**:
```python
dataset = to_sharegpt(
    dataset,
    merged_prompt="""### Instruction:
{instruction}

[[### Input:
{input}]]

### Response:
{output}""",
    output_column_name="text"
)
```

**Result**:
```python
{
    "text": """### Instruction:
Translate to French

### Input:
Hello world

### Response:
Bonjour le monde"""
}
```

## Complete Unsloth Training Pipeline

### Step-by-Step Implementation

```python
# 1. Import libraries
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 2. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 3. Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 4. Load and prepare dataset
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# Convert if needed
dataset = dataset.map(standardize_sharegpt)

# Apply chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

def format_prompts(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# 5. Configure training
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
)

# 6. Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
    packing=True,  # Enable packing
)

# 7. Train
trainer.train()

# 8. Save model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
```

## Model-Specific Templates

### Llama 3 Template Details

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_message}<|eot_id|>
```

### ChatML Template Details

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

### Alpaca Template Details

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
```

## Common Unsloth Issues and Solutions

### Issue: "Template not found"

**Problem**: Using incorrect template name

**Solution**:
```python
# Check available templates
from unsloth.chat_templates import CHAT_TEMPLATES
print(list(CHAT_TEMPLATES.keys()))

# Use exact name from list
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
```

### Issue: ShareGPT format not recognized

**Problem**: Dataset uses "from"/"value" but training expects "role"/"content"

**Solution**:
```python
from unsloth.chat_templates import standardize_sharegpt
dataset = dataset.map(standardize_sharegpt)
```

### Issue: Out of memory during training

**Solutions**:
1. Enable gradient checkpointing (done by default in Unsloth)
2. Reduce batch size
3. Disable packing if using long sequences
4. Reduce max_seq_length
5. Use gradient accumulation

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=8,   # Increase accumulation
    max_seq_length=1024,             # Reduce sequence length
)
```

### Issue: Poor multi-turn performance

**Problem**: Training only on single-turn examples

**Solution**:
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    conversation_extension=3,  # Enable multi-turn merging
    max_seq_length=2048,
)
```

## VRAM-Aware Dataset Design

### Memory Considerations

**Factors affecting VRAM**:
- Sequence length (most important)
- Batch size
- Model size (8B vs 70B)
- LoRA rank
- Gradient checkpointing

### Recommendations by GPU

**24GB VRAM (RTX 3090/4090)**:
- Model: 7B-8B with 4-bit quantization
- Max sequence length: 2048
- Batch size: 2-4
- LoRA rank: 8-16

**16GB VRAM (RTX 4060 Ti)**:
- Model: 7B with 4-bit quantization
- Max sequence length: 1024-1536
- Batch size: 1-2
- LoRA rank: 8

**12GB VRAM (RTX 3060)**:
- Model: 7B with 4-bit quantization
- Max sequence length: 512-1024
- Batch size: 1
- LoRA rank: 4-8

### Optimizing Dataset for VRAM

```python
# Filter long examples
dataset = dataset.filter(
    lambda x: len(tokenizer.encode(x['text'])) <= 1024
)

# Use packing for efficiency
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    packing=True,
    max_seq_length=1024,  # Set based on VRAM
)
```

## References

- [Unsloth Datasets Guide](https://docs.unsloth.ai/basics/datasets-guide)
- [Unsloth Chat Templates](https://docs.unsloth.ai/basics/chat-templates)
- [Unsloth Fine-Tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [Fine-Tune Llama 3.1 with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3)
