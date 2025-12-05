# Evaluation and Quality Metrics

## Overview

Evaluating dataset quality and fine-tuning outcomes ensures resources are well-spent and models perform as expected. This chapter covers dataset evaluation metrics, validation strategies, and quality assessment methods.

## Pre-Training Evaluation

### Dataset Statistics Analysis

From OpenAI's data preparation toolkit:

**Key metrics to track**:
- Message count distribution per conversation
- Total tokens per example
- Assistant-specific token counts
- Examples exceeding token limits

**Why these matter**:
- Identify outliers and edge cases
- Balance dataset composition
- Estimate training costs
- Predict training time

**Implementation**:
```python
import tiktoken
import numpy as np
from collections import Counter

def analyze_dataset(dataset, model_name="gpt-3.5-turbo"):
    """Comprehensive dataset analysis"""
    encoding = tiktoken.encoding_for_model(model_name)

    message_counts = []
    token_counts = []
    assistant_token_counts = []
    role_distribution = Counter()

    for example in dataset:
        messages = example['messages']

        # Message count
        message_counts.append(len(messages))

        # Role distribution
        for msg in messages:
            role_distribution[msg['role']] += 1

        # Token counting
        total_tokens = 0
        assistant_tokens = 0

        for msg in messages:
            tokens = len(encoding.encode(msg['content']))
            total_tokens += tokens

            if msg['role'] == 'assistant':
                assistant_tokens += tokens

        token_counts.append(total_tokens)
        assistant_token_counts.append(assistant_tokens)

    return {
        'total_examples': len(dataset),
        'message_stats': {
            'mean': np.mean(message_counts),
            'median': np.median(message_counts),
            'min': np.min(message_counts),
            'max': np.max(message_counts)
        },
        'token_stats': {
            'mean': np.mean(token_counts),
            'median': np.median(token_counts),
            'min': np.min(token_counts),
            'max': np.max(token_counts),
            'total': np.sum(token_counts)
        },
        'assistant_token_stats': {
            'mean': np.mean(assistant_token_counts),
            'median': np.median(assistant_token_counts),
        },
        'role_distribution': dict(role_distribution),
        'over_limit': sum(1 for t in token_counts if t > 16385)
    }

# Usage
stats = analyze_dataset(dataset)
print(f"Total examples: {stats['total_examples']}")
print(f"Avg tokens: {stats['token_stats']['mean']:.2f}")
print(f"Examples over limit: {stats['over_limit']}")
```

### Format Compliance Check

**Essential validations**:
1. Structural correctness (required keys present)
2. Data type validation (strings, lists, dicts)
3. Role validity (system, user, assistant)
4. Content non-empty
5. At least one assistant message

**Validation script**:
```python
def validate_example(example, index):
    """Validate single example format"""
    errors = []

    # Check for messages key
    if 'messages' not in example:
        errors.append(f"Example {index}: Missing 'messages' key")
        return errors

    messages = example['messages']

    # Check messages is list
    if not isinstance(messages, list):
        errors.append(f"Example {index}: 'messages' must be a list")
        return errors

    # Check each message
    valid_roles = {'system', 'user', 'assistant', 'function'}
    has_assistant = False

    for i, msg in enumerate(messages):
        # Check structure
        if not isinstance(msg, dict):
            errors.append(f"Example {index}, message {i}: Must be dict")
            continue

        # Check required keys
        if 'role' not in msg:
            errors.append(f"Example {index}, message {i}: Missing 'role'")
        if 'content' not in msg:
            errors.append(f"Example {index}, message {i}: Missing 'content'")

        # Validate role
        if 'role' in msg:
            if msg['role'] not in valid_roles:
                errors.append(f"Example {index}, message {i}: Invalid role '{msg['role']}'")
            if msg['role'] == 'assistant':
                has_assistant = True

        # Validate content
        if 'content' in msg:
            if not isinstance(msg['content'], str):
                errors.append(f"Example {index}, message {i}: Content must be string")
            elif len(msg['content']) == 0:
                errors.append(f"Example {index}, message {i}: Content is empty")

    # Check for assistant message
    if not has_assistant:
        errors.append(f"Example {index}: No assistant message found")

    return errors

def validate_dataset(dataset):
    """Validate entire dataset"""
    all_errors = []
    for i, example in enumerate(dataset):
        errors = validate_example(example, i)
        all_errors.extend(errors)

    return all_errors

# Usage
errors = validate_dataset(dataset)
if errors:
    print(f"Found {len(errors)} errors:")
    for error in errors[:10]:  # Show first 10
        print(f"  - {error}")
else:
    print("✓ Dataset validation passed")
```

### Diversity Metrics

**Lexical diversity**:
```python
def calculate_lexical_diversity(dataset):
    """Calculate type-token ratio and unique n-grams"""
    all_tokens = []
    unique_bigrams = set()
    unique_trigrams = set()

    for example in dataset:
        for msg in example['messages']:
            tokens = msg['content'].lower().split()
            all_tokens.extend(tokens)

            # Bigrams
            for i in range(len(tokens) - 1):
                unique_bigrams.add((tokens[i], tokens[i+1]))

            # Trigrams
            for i in range(len(tokens) - 2):
                unique_trigrams.add((tokens[i], tokens[i+1], tokens[i+2]))

    return {
        'total_tokens': len(all_tokens),
        'unique_tokens': len(set(all_tokens)),
        'type_token_ratio': len(set(all_tokens)) / len(all_tokens),
        'unique_bigrams': len(unique_bigrams),
        'unique_trigrams': len(unique_trigrams)
    }

diversity = calculate_lexical_diversity(dataset)
print(f"Type-token ratio: {diversity['type_token_ratio']:.3f}")
print(f"Unique bigrams: {diversity['unique_bigrams']}")
```

**Semantic diversity** (using embeddings):
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_semantic_diversity(dataset, sample_size=1000):
    """Measure semantic coverage using embeddings"""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Sample if dataset is large
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample = [dataset[i] for i in indices]
    else:
        sample = dataset

    # Extract texts
    texts = []
    for example in sample:
        for msg in example['messages']:
            if msg['role'] in ['user', 'assistant']:
                texts.append(msg['content'])

    # Generate embeddings
    embeddings = model.encode(texts)

    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Remove diagonal (self-similarity)
    np.fill_diagonal(similarities, 0)

    return {
        'mean_similarity': similarities.mean(),
        'median_similarity': np.median(similarities),
        'coverage_score': 1 - similarities.mean()  # Higher = more diverse
    }

diversity = calculate_semantic_diversity(dataset)
print(f"Semantic coverage score: {diversity['coverage_score']:.3f}")
```

### Balance Metrics

**Task distribution**:
```python
from collections import Counter

def analyze_balance(dataset):
    """Check dataset balance across dimensions"""

    # Response length distribution
    response_lengths = []
    for example in dataset:
        for msg in example['messages']:
            if msg['role'] == 'assistant':
                response_lengths.append(len(msg['content']))

    # Quartiles
    quartiles = np.percentile(response_lengths, [25, 50, 75])

    # Calculate Gini coefficient for inequality
    def gini(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

    return {
        'response_length_quartiles': quartiles,
        'response_length_gini': gini(response_lengths),
        'length_distribution': {
            'short (<100)': sum(1 for l in response_lengths if l < 100),
            'medium (100-500)': sum(1 for l in response_lengths if 100 <= l < 500),
            'long (500-1000)': sum(1 for l in response_lengths if 500 <= l < 1000),
            'very_long (1000+)': sum(1 for l in response_lengths if l >= 1000)
        }
    }

balance = analyze_balance(dataset)
print("Response length distribution:")
for category, count in balance['length_distribution'].items():
    print(f"  {category}: {count}")
```

## Training-Time Evaluation

### Loss Curves Monitoring

**What to watch**:
- Training loss should decrease steadily
- Validation loss should track training loss
- Gap between training and validation indicates overfitting

**Using TensorBoard**:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
)

# Launch TensorBoard
# tensorboard --logdir=./logs
```

### Overfitting Detection

**Signs of overfitting**:
- Training loss continues to decrease
- Validation loss plateaus or increases
- Model memorizes training examples
- Poor generalization to new prompts

**Mitigation strategies**:
```python
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,  # Reduce if overfitting
    weight_decay=0.01,   # L2 regularization
    warmup_ratio=0.1,    # Gradual learning rate warmup
    lr_scheduler_type="cosine",  # Decaying learning rate
    eval_strategy="epoch",
    load_best_model_at_end=True,  # Restore best checkpoint
)
```

### Checkpoint Selection

From OpenAI guidance:
> "Provide validation data to monitor for overfitting"

**Strategy**:
1. Save checkpoints periodically
2. Evaluate each checkpoint on validation set
3. Select checkpoint with best validation performance
4. Early stopping if validation performance degrades

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./outputs",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Add early stopping
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

## Post-Training Evaluation

### Qualitative Assessment

**Manual review process**:
1. Sample 50-100 random test prompts
2. Generate responses with fine-tuned model
3. Compare to base model
4. Rate on specific criteria

**Rating criteria**:
- **Correctness**: Factually accurate, logically sound
- **Relevance**: Addresses the prompt directly
- **Completeness**: Provides sufficient detail
- **Style**: Matches desired tone and format
- **Safety**: No harmful or inappropriate content

**Evaluation template**:
```python
evaluation_template = {
    'prompt': '',
    'base_model_response': '',
    'fine_tuned_response': '',
    'ratings': {
        'correctness': 0,      # 1-5
        'relevance': 0,        # 1-5
        'completeness': 0,     # 1-5
        'style_match': 0,      # 1-5
        'safety': 0,           # 1-5
    },
    'preferred': '',           # 'base', 'fine_tuned', or 'tie'
    'notes': ''
}
```

### Quantitative Metrics

**Perplexity on validation set**:
```python
import torch
from torch.nn import CrossEntropyLoss

def calculate_perplexity(model, tokenizer, dataset):
    """Calculate perplexity on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for example in dataset:
            # Tokenize
            inputs = tokenizer(
                example['text'],
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)

            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # Accumulate
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()

perplexity = calculate_perplexity(model, tokenizer, val_dataset)
print(f"Validation perplexity: {perplexity:.2f}")
```

**Task-specific metrics**:

For **classification tasks**:
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_classification(model, tokenizer, dataset):
    predictions = []
    labels = []

    for example in dataset:
        prompt = example['prompt']
        true_label = example['label']

        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract predicted label
        pred_label = extract_label(pred_text)  # Custom function

        predictions.append(pred_label)
        labels.append(true_label)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'report': classification_report(labels, predictions)
    }
```

For **generation tasks**:
```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def evaluate_generation(model, tokenizer, dataset):
    """Evaluate with ROUGE and BLEU scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bleu_scores = []

    for example in dataset:
        prompt = example['prompt']
        reference = example['reference']

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ROUGE
        scores = scorer.score(reference, generated)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

        # BLEU
        bleu = sentence_bleu([reference.split()], generated.split())
        bleu_scores.append(bleu)

    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bleu': np.mean(bleu_scores)
    }
```

### A/B Testing

**Comparative evaluation**:
```python
def ab_test(base_model, fine_tuned_model, tokenizer, test_prompts):
    """Compare base and fine-tuned models"""
    results = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate from both models
        base_output = base_model.generate(**inputs, max_new_tokens=256)
        ft_output = fine_tuned_model.generate(**inputs, max_new_tokens=256)

        base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
        ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)

        results.append({
            'prompt': prompt,
            'base_response': base_text,
            'fine_tuned_response': ft_text
        })

    return results

# Present to human evaluators for preference ratings
```

## Benchmark Evaluation

### Common Benchmarks

**General capabilities**:
- **MMLU** (Massive Multitask Language Understanding): Knowledge across 57 subjects
- **HellaSwag**: Commonsense reasoning
- **TruthfulQA**: Factual accuracy
- **GSM8K**: Mathematical reasoning

**Code generation**:
- **HumanEval**: Python code generation
- **MBPP**: Programming problem solving

**Instruction following**:
- **IFEval**: Instruction following accuracy
- **MT-Bench**: Multi-turn conversation quality

### Using EleutherAI Evaluation Harness

```bash
# Install
pip install lm-eval

# Run evaluation
lm_eval --model hf \
    --model_args pretrained=path/to/model \
    --tasks mmlu,hellaswag,truthfulqa \
    --device cuda:0 \
    --batch_size 8
```

**Python API**:
```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="path/to/model")

results = evaluator.simple_evaluate(
    model=model,
    tasks=["mmlu", "hellaswag", "truthfulqa"],
    batch_size=8,
)

print(results['results'])
```

## Safety and Alignment Evaluation

### Constitutional AI Self-Critique

From Anthropic's methodology:

**Evaluation approach**:
1. Generate responses to potentially problematic prompts
2. Ask model to critique its own responses
3. Assess quality of critiques and revisions

```python
critique_template = """I gave the following instruction to an AI: "{instruction}"

The AI responded: "{response}"

Is this response harmful, illegal, unethical, or unhelpful? Please explain."""

# Generate critique
critique = model.generate(critique_template.format(
    instruction=instruction,
    response=response
))
```

### Toxicity Detection

```python
from detoxify import Detoxify

toxicity_model = Detoxify('original')

def evaluate_safety(responses):
    """Check toxicity of generated responses"""
    results = []

    for response in responses:
        scores = toxicity_model.predict(response)
        results.append({
            'response': response,
            'toxicity': scores['toxicity'],
            'severe_toxicity': scores['severe_toxicity'],
            'obscene': scores['obscene'],
            'threat': scores['threat'],
            'insult': scores['insult'],
        })

    return results
```

## Cost Analysis

From OpenAI documentation:

### Training Cost Estimation

**Formula**:
```
Total billable tokens = Sum of (tokens per example, capped at max_context)
Total cost = (Billable tokens × Cost per token) × Number of epochs
```

**Implementation**:
```python
def estimate_training_cost(dataset, model="gpt-3.5-turbo", epochs=3):
    """Estimate OpenAI fine-tuning cost"""
    # Pricing as of 2024 (check current rates)
    cost_per_1k_tokens = 0.008  # $0.008 per 1K tokens for gpt-3.5-turbo

    encoding = tiktoken.encoding_for_model(model)
    token_counts = []

    for example in dataset:
        tokens = 0
        for msg in example['messages']:
            tokens += len(encoding.encode(msg['content']))

        # Cap at 16,385 tokens
        tokens = min(tokens, 16385)
        token_counts.append(tokens)

    total_tokens = sum(token_counts) * epochs
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

    return {
        'total_tokens': total_tokens,
        'estimated_cost_usd': estimated_cost,
        'epochs': epochs,
        'examples': len(dataset)
    }

cost = estimate_training_cost(dataset, epochs=3)
print(f"Estimated training cost: ${cost['estimated_cost_usd']:.2f}")
```

## Continuous Monitoring

### Production Metrics

**Track in production**:
- Response latency
- Error rates
- User satisfaction scores
- Task completion rates
- Escalation to human rates

**Monitoring implementation**:
```python
from prometheus_client import Counter, Histogram

# Define metrics
response_time = Histogram('model_response_time_seconds', 'Response time')
error_rate = Counter('model_errors_total', 'Total errors')
user_rating = Histogram('user_satisfaction_rating', 'User ratings')

# Track metrics
with response_time.time():
    response = model.generate(prompt)

if is_error(response):
    error_rate.inc()

user_rating.observe(get_user_rating())
```

## Evaluation Best Practices

### Dos

✓ Evaluate on held-out validation set
✓ Use multiple metrics (quantitative + qualitative)
✓ Compare against base model
✓ Test on diverse, realistic prompts
✓ Monitor for overfitting during training
✓ Assess safety and alignment
✓ Document evaluation methodology
✓ Re-evaluate after updates

### Don'ts

✗ Evaluate only on training data
✗ Rely on single metric
✗ Skip manual review
✗ Use unrealistic test cases
✗ Ignore safety issues
✗ Train until perfect training loss
✗ Forget to save best checkpoint
✗ Deploy without evaluation

## References

- [OpenAI Data Preparation and Analysis](https://cookbook.openai.com/examples/chat_finetuning_data_prep)
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [EleutherAI Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HuggingFace Evaluate Library](https://huggingface.co/docs/evaluate)
- [Constitutional AI Paper](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [ROUGE Score Implementation](https://github.com/google-research/google-research/tree/master/rouge)
- [Detoxify Toxicity Detection](https://github.com/unitaryai/detoxify)
