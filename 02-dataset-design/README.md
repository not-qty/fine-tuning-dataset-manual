# Dataset Design Principles

## Overview

Effective fine-tuning datasets balance structure, diversity, and task alignment. This chapter covers fundamental design principles drawn from industry-leading implementations.

## Core Dataset Types

### Language Modeling Format

**Purpose**: Continued pretraining or general text completion

**Structure**: Complete text sequences in plain format

```json
{"text": "The sky is blue and the grass is green."}
```

**Use cases**:
- Domain adaptation
- Style transfer
- Vocabulary expansion

### Instruction Format

**Purpose**: Teaching specific task patterns

**Structure**: Alpaca-style with instruction, optional input, and output

```json
{
  "instruction": "Translate the following English text to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**Use cases**:
- Task-specific training
- Few-shot learning enhancement
- Controlled generation

### Conversational Format

**Purpose**: Multi-turn dialogue and chat applications

**Structure**: Message sequences with roles and content

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

**Use cases**:
- Chatbots
- Interactive assistants
- Context-aware responses

### Preference Format

**Purpose**: Alignment, RLHF, and DPO training

**Structure**: Prompt with chosen and rejected responses

```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum mechanics...",
  "rejected": "Quantum stuff is just computers but weird..."
}
```

**Use cases**:
- Response quality improvement
- Safety alignment
- Style preferences

## Dataset Size Requirements

### Minimum Viable Datasets

From OpenAI's guidance:
- **Start with**: 30-50 well-curated examples
- **Production quality**: 100+ examples minimum
- **Optimal results**: 1,000+ high-quality examples

### Scaling Considerations

Performance typically scales linearly with dataset size, but:
- Quality matters more than quantity
- Diverse examples outperform repetitive data
- Longer training jobs with larger datasets

## Data Structure Best Practices

### Message Role Definitions

**system**: Sets behavior and context
- Appears first in conversations
- Defines personality, constraints, expertise
- Optional but recommended

**user**: Input from the human
- Contains queries, instructions, prompts
- Required in conversational datasets

**assistant**: Model's expected response
- Contains target output for training
- Must be present for supervision

### Multi-Turn Conversation Design

From OpenAI's recommendations:
> "If your model handles multi-turn interactions, please provide representative examples to prevent degraded performance when conversations expand."

Design principles:
- Include realistic conversation flows
- Vary conversation length
- Represent typical context switches
- Model error correction patterns

### Token Considerations

**OpenAI limits**: 4,096 tokens per example (truncated beyond)
**General practice**: Most frameworks handle 2,000-8,000 tokens

Token efficiency tips:
- Remove unnecessary whitespace
- Avoid redundant context repetition
- Use concise system messages
- Implement dataset packing (see Chapter 7)

## Task-Specific Design Patterns

### Classification Tasks

Structure responses as consistent labels:

```json
{
  "messages": [
    {"role": "user", "content": "Classify sentiment: The movie was terrible."},
    {"role": "assistant", "content": "negative"}
  ]
}
```

### Generation Tasks

Provide full-form outputs:

```json
{
  "messages": [
    {"role": "user", "content": "Write a haiku about code"},
    {"role": "assistant", "content": "Functions that compile\nVariables dance in memory\nLogic finds its way"}
  ]
}
```

### Function Calling

Include JSON schema in training data:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "{\"function\": \"get_weather\", \"args\": {\"location\": \"current\"}}"}
  ]
}
```

## Data Mixing Strategies

### Combining Multiple Sources

From HuggingFace TRL documentation:

Benefits of mixing datasets:
- Reduces overfitting
- Improves generalization
- Balances task capabilities

Mixing approaches:
- **Uniform mixing**: Equal samples from each source
- **Weighted mixing**: Proportional to dataset importance
- **Curriculum mixing**: Progress from simple to complex

### Balancing Dataset Composition

Consider:
- **Domain diversity**: Technical, conversational, creative
- **Difficulty levels**: Simple to complex examples
- **Response lengths**: Vary from concise to detailed
- **Task types**: Mix generation, classification, reasoning

## Validation Split Design

Best practices from OpenAI Cookbook:

### Validation Set Purpose
- Monitor overfitting
- Tune hyperparameters
- Evaluate generalization

### Split Recommendations
- **Small datasets** (<1000): 10-20% validation
- **Large datasets** (>10000): 5-10% validation
- Ensure validation represents training distribution

### Validation Data Characteristics
- Cover all task types in training set
- Include edge cases and difficult examples
- Maintain similar length distribution

## Common Design Pitfalls

### Data Leakage
- Validation data appears in training set
- Test set information influences training
- External knowledge not available at inference

### Format Inconsistencies
- Mixed role naming (user vs human)
- Inconsistent message structure
- Varying field names across examples

### Inadequate Diversity
- Repetitive phrasing patterns
- Limited domain coverage
- Narrow response styles

### Token Inefficiency
- Excessive padding
- Redundant context in each example
- Missed packing opportunities

## References

- [HuggingFace TRL Dataset Formats](https://huggingface.co/docs/trl/main/dataset_formats)
- [OpenAI Chat Fine-Tuning Data Prep](https://cookbook.openai.com/examples/chat_finetuning_data_prep)
- [OpenAI How to Fine-Tune Chat Models](https://cookbook.openai.com/examples/how_to_finetune_chat_models)
- [Unsloth Datasets Guide](https://docs.unsloth.ai/basics/datasets-guide)
