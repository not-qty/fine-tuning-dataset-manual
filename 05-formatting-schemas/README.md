# Formatting Schemas and Standards

## Overview

This chapter details the specific data format schemas used across different fine-tuning frameworks, including JSONL structure, ChatML, ShareGPT, and model-specific templates.

## JSONL (JSON Lines) Format

### What is JSONL?

From HuggingFace blog:
> "JSONL (JSON Lines) is widely adopted in modern LLM pipelines because each line is a separate JSON object—ideal for nested and multi-field data"

**Key characteristics**:
- One complete JSON object per line
- No JSON array wrapper
- Streamable and memory-efficient
- Easy to append and process incrementally

### JSONL Structure Example

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well!"}]}
{"messages": [{"role": "user", "content": "Goodbye"}, {"role": "assistant", "content": "See you later!"}]}
```

**Important**: Each line is a complete, valid JSON object

### When to Use JSONL

From HuggingFace documentation:
> "If you do not want to format the data yourself and want --chat-template parameter to format the data for you, you must use JSONL format"

**Best for**:
- Fine-tuning datasets
- Streaming large datasets
- Multi-field structured data
- Chat and instruction formats

### Creating JSONL Files

**Python example**:
```python
import json

data = [
    {
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is Java?"},
            {"role": "assistant", "content": "Java is a programming language"}
        ]
    }
]

with open('dataset.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
```

## ChatML Format (OpenAI Standard)

### Structure

ChatML uses a list of message objects with "role" and "content" keys:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

### Role Specifications

**Valid roles**: "system", "user", "assistant", "function"

**Role purposes**:
- **system**: Sets context and behavior (appears first)
- **user**: Human input
- **assistant**: AI response
- **function**: Function call results (advanced)

### ChatML Special Tokens

Models using ChatML typically employ control tokens:

```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi! How can I help?<|im_end|>
```

**Token meanings**:
- `<|im_start|>`: Marks beginning of message
- `<|im_end|>`: Marks end of message
- Role name follows `im_start`

### Function Calling in ChatML

Extended format for function calls:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Boston\"}"
      }
    },
    {
      "role": "function",
      "name": "get_weather",
      "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
    },
    {"role": "assistant", "content": "It's 72°F and sunny in Boston"}
  ]
}
```

## ShareGPT Format

### Structure

ShareGPT format uses "conversations" array with "from" and "value" keys:

```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is..."},
    {"from": "human", "value": "Can you give an example?"},
    {"from": "gpt", "value": "Sure! Here's an example..."}
  ]
}
```

### Role Naming

**Standard roles**:
- `human`: User input
- `gpt`: Assistant response
- `system`: System message (less common in ShareGPT)

### Converting ShareGPT to ChatML

From Unsloth documentation:
> "If your dataset uses the ShareGPT format with 'from'/'value' keys instead of the ChatML 'role'/'content' format, you can use the standardize_sharegpt function to convert it first"

**Unsloth conversion**:
```python
from unsloth.chat_templates import standardize_sharegpt

dataset = standardize_sharegpt(dataset)
# Converts: from/value → role/content
# Converts: human → user
# Converts: gpt → assistant
```

### When to Use ShareGPT

**Best for**:
- Multi-turn conversations
- Chatbot training data
- Datasets exported from conversation interfaces
- Training data with natural dialogue flow

## Model-Specific Chat Templates

### Llama 3 Format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

**Special tokens**:
- `<|begin_of_text|>`: Start of sequence
- `<|start_header_id|>`: Begin role section
- `<|end_header_id|>`: End role section
- `<|eot_id|>`: End of turn

### Mistral Instruct Format

```
<s>[INST] Hello [/INST] Hi there!</s>[INST] How are you? [/INST] I'm doing well!</s>
```

**Special tokens**:
- `<s>`: Beginning of sequence
- `[INST]` / `[/INST]`: Instruction delimiters
- `</s>`: End of sequence

### Alpaca Format (Legacy but Common)

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a Python function to calculate factorial

### Response:
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```

**Variations with input**:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

### Vicuna Format

```
USER: What is the capital of France?
ASSISTANT: The capital of France is Paris.
USER: What about Germany?
ASSISTANT: The capital of Germany is Berlin.
```

## Applying Chat Templates with HuggingFace

### Using apply_chat_template()

From HuggingFace documentation:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model_name")

messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
]

# Generate formatted string
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Or tokenize directly
tokens = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)
```

### Key Parameters

**tokenize**:
- `True`: Returns token IDs
- `False`: Returns formatted string

**add_generation_prompt**:
- `True`: Adds tokens for assistant response start
- `False`: For training (don't add generation tokens)

From documentation:
> "During model training, apply chat templates as preprocessing with add_generation_prompt=False, since generation prompt tokens aren't useful for training phases"

**continue_final_message**:
- Enables response prefilling
- Removes end-of-sequence tokens
- Used for constrained generation

### Important Tokenization Warning

From HuggingFace:
> "When using tokenize=False initially, set add_special_tokens=False during later tokenization to prevent duplicating these tokens and degrading performance"

**Correct workflow**:
```python
# Step 1: Get formatted text
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)

# Step 2: Tokenize with add_special_tokens=False
tokens = tokenizer(
    formatted_text,
    add_special_tokens=False  # Critical!
)
```

## Unsloth-Specific Requirements

### Mandatory Field Placeholders

From Unsloth documentation:

**Required**:
- `{INPUT}`: Instruction/prompt field
- `{OUTPUT}`: Model's response field

**Optional**:
- `{SYSTEM}`: System prompt field

### Template Example

```python
alpaca_template = """Below is an instruction. Write a response.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""
```

### Using Unsloth Templates

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # or "chatml", "alpaca", etc.
    mapping={"role": "from", "content": "value"}  # For ShareGPT
)
```

### Unsloth's to_sharegpt Function

Merge dataset columns into prompts:

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

**Syntax**:
- `{column_name}`: Include column value
- `[[optional text]]`: Only include if column has value
- `output_column_name`: Target column for merged result

## Dataset Packing

### What is Dataset Packing?

From Unsloth documentation:
> "Batches have a pre-defined sequence length - instead of assigning one batch per sample, multiple small samples can be combined in one batch, increasing efficiency"

### When to Use Packing

**Benefits**:
- Reduces padding waste
- Improves training efficiency
- Better GPU utilization
- Faster training

**Best for**:
- Variable-length sequences
- Short conversations
- Resource-constrained training

**Avoid when**:
- Sequences already near max length
- Cross-sequence contamination concerns
- Debugging (harder to trace issues)

### Implementing Packing

**HuggingFace TRL**:
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    packing=True,  # Enable packing
    max_seq_length=2048
)
```

**Unsloth**:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Packing handled automatically with proper dataset format
```

## Format Validation Checklist

### ChatML Validation

- [ ] All messages have "role" and "content" keys
- [ ] Roles are valid: system/user/assistant/function
- [ ] Content is non-empty string
- [ ] At least one assistant message
- [ ] System message (if present) appears first

### ShareGPT Validation

- [ ] Conversations use "from" and "value" keys
- [ ] From values are "human", "gpt", or "system"
- [ ] Alternating human/gpt pattern (mostly)
- [ ] Value is non-empty string

### JSONL Validation

- [ ] Each line is valid JSON
- [ ] No JSON array wrapper
- [ ] Consistent schema across lines
- [ ] Proper line termination

### Template Application

- [ ] Template matches target model
- [ ] Special tokens correct
- [ ] No token duplication
- [ ] Proper role ordering

## Format Conversion Tools

### Python Utilities

```python
def chatml_to_sharegpt(chatml_data):
    """Convert ChatML to ShareGPT format"""
    role_mapping = {
        "user": "human",
        "assistant": "gpt",
        "system": "system"
    }

    return {
        "conversations": [
            {
                "from": role_mapping.get(msg["role"], msg["role"]),
                "value": msg["content"]
            }
            for msg in chatml_data["messages"]
        ]
    }

def sharegpt_to_chatml(sharegpt_data):
    """Convert ShareGPT to ChatML format"""
    role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system"
    }

    return {
        "messages": [
            {
                "role": role_mapping.get(conv["from"], conv["from"]),
                "content": conv["value"]
            }
            for conv in sharegpt_data["conversations"]
        ]
    }
```

### HuggingFace Datasets Integration

```python
from datasets import load_dataset

# Load ChatML format
dataset = load_dataset("json", data_files="data.jsonl")

# Apply conversion
dataset = dataset.map(lambda x: sharegpt_to_chatml(x))

# Save in new format
dataset.to_json("converted.jsonl")
```

## Common Format Errors

### Incorrect JSONL

**Wrong** (JSON array):
```json
[
  {"messages": [...]},
  {"messages": [...]}
]
```

**Right** (JSONL):
```jsonl
{"messages": [...]}
{"messages": [...]}
```

### Mixed Role Names

**Wrong**:
```json
{"messages": [
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "Hello"},
  {"role": "human", "content": "How are you?"}  // Mixed!
]}
```

**Right**:
```json
{"messages": [
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "Hello"},
  {"role": "user", "content": "How are you?"}
]}
```

### Missing Required Fields

**Wrong**:
```json
{"messages": [
  {"role": "user"},  // Missing content!
  {"content": "Hello"}  // Missing role!
]}
```

**Right**:
```json
{"messages": [
  {"role": "user", "content": "Question"},
  {"role": "assistant", "content": "Hello"}
]}
```

## References

- [HuggingFace Chat Templating](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [HuggingFace TRL Dataset Formats](https://huggingface.co/docs/trl/main/dataset_formats)
- [LLM Dataset Formats 101](https://huggingface.co/blog/tegridydev/llm-dataset-formats-101-hugging-face)
- [Unsloth Chat Templates](https://docs.unsloth.ai/basics/chat-templates)
- [Unsloth Datasets Guide](https://docs.unsloth.ai/basics/datasets-guide)
- [OpenAI Fine-Tuning Guide](https://cookbook.openai.com/examples/how_to_finetune_chat_models)
