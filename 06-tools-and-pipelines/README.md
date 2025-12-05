# Tools and Pipelines

## Overview

This chapter covers the practical tools, libraries, and pipeline implementations for creating, processing, and validating fine-tuning datasets across HuggingFace, OpenAI, and other ecosystems.

## HuggingFace Datasets Library

### Core Functionality

The `datasets` library provides the foundation for dataset manipulation in the HuggingFace ecosystem.

**Installation**:
```bash
pip install datasets
```

**Key features**:
- Automatic format conversion
- Streaming for large datasets
- Built-in caching
- Memory-mapped operations
- Integration with transformers

### Loading Datasets

**From local files**:
```python
from datasets import load_dataset

# JSONL
dataset = load_dataset("json", data_files="train.jsonl")

# CSV
dataset = load_dataset("csv", data_files="train.csv")

# Parquet
dataset = load_dataset("parquet", data_files="train.parquet")

# Multiple files
dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "validation": "val.jsonl"
    }
)
```

**From HuggingFace Hub**:
```python
dataset = load_dataset("username/dataset-name")
dataset = load_dataset("username/dataset-name", split="train")
```

### Dataset Operations

**Filtering**:
```python
# Remove short examples
dataset = dataset.filter(lambda x: len(x['text']) > 100)

# Language filtering
dataset = dataset.filter(lambda x: x['language'] == 'en')

# Custom conditions
def is_high_quality(example):
    return (
        len(example['response']) > 50 and
        len(example['response']) < 1000 and
        'http' not in example['response']
    )

dataset = dataset.filter(is_high_quality)
```

**Mapping (transformation)**:
```python
# Apply chat template
def apply_template(example):
    messages = example['messages']
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    return {'formatted': formatted}

dataset = dataset.map(apply_template)

# Batched processing (faster)
dataset = dataset.map(apply_template, batched=True, batch_size=1000)

# With multiple processes
dataset = dataset.map(apply_template, num_proc=4)
```

**Selecting and shuffling**:
```python
# Take first N examples
dataset = dataset.select(range(1000))

# Random shuffle
dataset = dataset.shuffle(seed=42)

# Train/val split
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train = split_dataset['train']
val = split_dataset['test']
```

**Deduplication**:
```python
# Remove exact duplicates
dataset = dataset.unique(column='text')
```

### Streaming for Large Datasets

**Why streaming**:
- Process datasets larger than RAM
- Faster iteration start
- Lower memory footprint

**Enable streaming**:
```python
dataset = load_dataset(
    "large/dataset",
    streaming=True
)

# Iterate without loading everything
for example in dataset['train'].take(100):
    process(example)
```

### Saving Datasets

```python
# Save to disk
dataset.save_to_disk("path/to/dataset")

# Save as JSONL
dataset.to_json("output.jsonl")

# Save as CSV
dataset.to_csv("output.csv")

# Save as Parquet
dataset.to_parquet("output.parquet")

# Push to HuggingFace Hub
dataset.push_to_hub("username/dataset-name")
```

## HuggingFace TRL (Transformer Reinforcement Learning)

### SFTTrainer (Supervised Fine-Tuning)

**Basic setup**:
```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    max_seq_length=2048,
    packing=True  # Enable dataset packing
)

trainer.train()
```

**Dataset format compatibility**:
```python
# SFTTrainer accepts multiple formats
# 1. Text column
dataset_format = {"text": "..."}

# 2. Prompt-completion
dataset_format = {"prompt": "...", "completion": "..."}

# 3. Messages (ChatML)
dataset_format = {"messages": [...]}
```

### DPO Trainer (Direct Preference Optimization)

**Dataset format**:
```python
{
    "prompt": "Explain quantum computing",
    "chosen": "High-quality response...",
    "rejected": "Low-quality response..."
}
```

**Training**:
```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Reference model for comparison
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1  # KL divergence coefficient
)

trainer.train()
```

### RewardTrainer

**For training reward models**:
```python
from trl import RewardTrainer

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=512
)

trainer.train()
```

## OpenAI Fine-Tuning Tools

### Data Preparation Script

From OpenAI Cookbook:

**Format validation**:
```python
import json
import tiktoken

def validate_dataset(file_path):
    """Validate OpenAI fine-tuning dataset format"""
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    errors = []
    for i, example in enumerate(dataset):
        # Check structure
        if "messages" not in example:
            errors.append(f"Example {i}: Missing 'messages' key")
            continue

        messages = example["messages"]

        # Check message format
        for j, message in enumerate(messages):
            if "role" not in message:
                errors.append(f"Example {i}, message {j}: Missing 'role'")
            if "content" not in message:
                errors.append(f"Example {i}, message {j}: Missing 'content'")
            if message.get("role") not in ["system", "user", "assistant", "function"]:
                errors.append(f"Example {i}, message {j}: Invalid role")

        # Check for at least one assistant message
        if not any(m.get("role") == "assistant" for m in messages):
            errors.append(f"Example {i}: No assistant message found")

    return errors

# Run validation
errors = validate_dataset("train.jsonl")
if errors:
    for error in errors:
        print(error)
else:
    print("Dataset validation passed!")
```

**Token counting**:
```python
def count_tokens(file_path, model="gpt-3.5-turbo"):
    """Count tokens for cost estimation"""
    encoding = tiktoken.encoding_for_model(model)

    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    token_counts = []
    for example in dataset:
        messages = example["messages"]
        tokens = 0

        for message in messages:
            # Count role and content
            tokens += len(encoding.encode(message["role"]))
            tokens += len(encoding.encode(message["content"]))
            tokens += 3  # Message formatting tokens

        tokens += 3  # Conversation formatting tokens
        token_counts.append(tokens)

    return {
        "total_tokens": sum(token_counts),
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "examples": len(token_counts)
    }

# Calculate statistics
stats = count_tokens("train.jsonl")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Average tokens per example: {stats['avg_tokens']:.2f}")
```

### OpenAI Fine-Tuning CLI

**Using OpenAI API**:
```python
from openai import OpenAI

client = OpenAI()

# Upload training file
with open("train.jsonl", "rb") as f:
    training_file = client.files.create(
        file=f,
        purpose="fine-tune"
    )

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3
    }
)

# Monitor progress
while True:
    job_status = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job_status.status}")
    if job_status.status in ["succeeded", "failed"]:
        break
    time.sleep(60)
```

## Data Quality Tools

### fastText for Language Detection

**Installation**:
```bash
pip install fasttext
```

**Usage**:
```python
import fasttext

# Download model
# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

model = fasttext.load_model('lid.176.bin')

def detect_language(text):
    predictions = model.predict(text, k=1)
    language = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    return language, confidence

# Filter dataset
def filter_by_language(example):
    lang, conf = detect_language(example['text'])
    return lang == 'en' and conf > 0.8

dataset = dataset.filter(filter_by_language)
```

### datasketch for Deduplication

**Installation**:
```bash
pip install datasketch
```

**MinHash LSH deduplication**:
```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text"""
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def deduplicate_dataset(dataset, threshold=0.7):
    """Remove near-duplicate examples"""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    unique_indices = []
    seen_duplicates = set()

    for i, example in enumerate(dataset):
        if i in seen_duplicates:
            continue

        text = example['text']
        minhash = create_minhash(text)

        # Check for duplicates
        duplicates = lsh.query(minhash)

        if not duplicates:
            lsh.insert(f"doc_{i}", minhash)
            unique_indices.append(i)
        else:
            seen_duplicates.add(i)

    return dataset.select(unique_indices)

# Apply deduplication
dataset = deduplicate_dataset(dataset)
```

### sentence-transformers for Semantic Similarity

**Installation**:
```bash
pip install sentence-transformers
```

**Semantic deduplication**:
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_deduplicate(dataset, threshold=0.9):
    """Remove semantically similar examples"""
    texts = [ex['text'] for ex in dataset]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    # Calculate similarities
    similarities = cosine_similarity(embeddings)

    # Find unique examples
    unique_indices = []
    removed = set()

    for i in range(len(texts)):
        if i in removed:
            continue

        unique_indices.append(i)

        # Mark similar examples as duplicates
        similar = np.where(similarities[i] > threshold)[0]
        for j in similar:
            if j > i:
                removed.add(j)

    return dataset.select(unique_indices)

dataset = semantic_deduplicate(dataset)
```

## Complete Pipeline Example

### End-to-End Dataset Preparation

```python
from datasets import load_dataset
import fasttext
from datasketch import MinHash, MinHashLSH
import tiktoken

class DatasetPipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.lang_model = fasttext.load_model('lid.176.bin')

    def validate_format(self, example):
        """Ensure proper message format"""
        if 'messages' not in example:
            return False

        messages = example['messages']
        if not isinstance(messages, list):
            return False

        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                return False
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False
            if not isinstance(msg['content'], str) or len(msg['content']) == 0:
                return False

        # Ensure at least one assistant message
        has_assistant = any(m['role'] == 'assistant' for m in messages)
        return has_assistant

    def filter_language(self, example, target_lang='en', min_confidence=0.8):
        """Filter by language"""
        text = ' '.join(m['content'] for m in example['messages'])
        predictions = self.lang_model.predict(text, k=1)
        lang = predictions[0][0].replace('__label__', '')
        conf = predictions[1][0]
        return lang == target_lang and conf >= min_confidence

    def filter_length(self, example, min_tokens=10, max_tokens=2048):
        """Filter by token length"""
        text = ' '.join(m['content'] for m in example['messages'])
        formatted = self.tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False
        )
        tokens = len(self.tokenizer.encode(formatted))
        return min_tokens <= tokens <= max_tokens

    def process_dataset(self, input_file, output_file):
        """Run complete pipeline"""
        print("Loading dataset...")
        dataset = load_dataset("json", data_files=input_file)['train']
        print(f"Initial size: {len(dataset)}")

        # Step 1: Format validation
        print("Validating format...")
        dataset = dataset.filter(self.validate_format)
        print(f"After format validation: {len(dataset)}")

        # Step 2: Language filtering
        print("Filtering language...")
        dataset = dataset.filter(self.filter_language)
        print(f"After language filter: {len(dataset)}")

        # Step 3: Length filtering
        print("Filtering length...")
        dataset = dataset.filter(self.filter_length)
        print(f"After length filter: {len(dataset)}")

        # Step 4: Deduplication
        print("Deduplicating...")
        dataset = dataset.unique(column='messages')
        print(f"After deduplication: {len(dataset)}")

        # Step 5: Shuffle and split
        print("Shuffling and splitting...")
        dataset = dataset.shuffle(seed=42)
        split = dataset.train_test_split(test_size=0.1, seed=42)

        # Save
        print(f"Saving to {output_file}...")
        split['train'].to_json(f"{output_file}_train.jsonl")
        split['test'].to_json(f"{output_file}_val.jsonl")

        print("Pipeline complete!")
        return split

# Usage
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
pipeline = DatasetPipeline(tokenizer)
dataset = pipeline.process_dataset("raw_data.jsonl", "processed_data")
```

## Recommended Workflow

### 1. Initial Data Collection
- Gather raw data from sources
- Convert to common format (JSONL)
- Store in version control

### 2. Validation Phase
```python
# Run format validation
errors = validate_dataset("raw_data.jsonl")
# Fix errors before proceeding
```

### 3. Quality Filtering
```python
dataset = load_dataset("json", data_files="raw_data.jsonl")['train']
dataset = dataset.filter(validate_format)
dataset = dataset.filter(filter_language)
dataset = dataset.filter(filter_toxicity)  # If needed
```

### 4. Deduplication
```python
dataset = dataset.unique(column='text')
# Or use MinHash/semantic dedup for better results
```

### 5. Length Filtering
```python
dataset = dataset.filter(lambda x: check_token_length(x, min=10, max=2048))
```

### 6. Train/Val Split
```python
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
```

### 7. Template Application
```python
def apply_template(example):
    return {
        'text': tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
    }

split_dataset = split_dataset.map(apply_template)
```

### 8. Save and Document
```python
split_dataset.save_to_disk("final_dataset")
# Document dataset statistics, filtering decisions, etc.
```

## References

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets)
- [HuggingFace TRL Documentation](https://huggingface.co/docs/trl)
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI Data Preparation Cookbook](https://cookbook.openai.com/examples/chat_finetuning_data_prep)
- [fastText Language Identification](https://fasttext.cc/docs/en/language-identification.html)
- [datasketch Documentation](https://ekzhu.com/datasketch/)
- [sentence-transformers Documentation](https://www.sbert.net/)
