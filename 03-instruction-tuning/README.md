# Instruction Tuning Datasets

## Overview

Instruction tuning teaches language models to follow directives and respond appropriately to user requests. This chapter covers instruction dataset construction, formats, and best practices.

## What is Instruction Tuning?

Instruction tuning bridges the gap between pre-trained language models and user-aligned assistants by training on explicit instruction-following examples.

**Key characteristics**:
- Input contains explicit instructions or queries
- Output demonstrates desired behavior
- Format emphasizes clarity and task specification

## Instruction Dataset Formats

### Alpaca Format

Classic single-turn instruction format:

```json
{
  "instruction": "Summarize the main points",
  "input": "Long document text here...",
  "output": "Key points: 1. First point 2. Second point..."
}
```

**When to use**:
- Simple task completion
- Single-step operations
- Clear input-output mappings

**Components**:
- **instruction**: Task description or directive
- **input**: Data to process (optional)
- **output**: Expected result

### ChatML Format (OpenAI Standard)

Conversational instruction format using role-based messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert programmer."},
    {"role": "user", "content": "Write a Python function to calculate factorial"},
    {"role": "assistant", "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}
  ]
}
```

**When to use**:
- Chat-based applications
- Multi-turn interactions
- Context-dependent tasks

**Special tokens**:
Models typically use markers like:
- `<|im_start|>` and `<|im_end|>` (ChatML)
- `<|begin_of_text|>` and `<|end_of_text|>` (Llama)
- Model-specific tokens (see Chapter 5)

### ShareGPT Format

Multi-turn conversation format:

```json
{
  "conversations": [
    {"from": "human", "value": "Explain recursion"},
    {"from": "gpt", "value": "Recursion is when a function calls itself..."},
    {"from": "human", "value": "Can you give an example?"},
    {"from": "gpt", "value": "Here's a factorial example..."}
  ]
}
```

**When to use**:
- Training on existing conversations
- Multi-step problem solving
- Context accumulation

**Conversion**: Can be standardized to ChatML (see Unsloth tools in Chapter 7)

## System Message Design

### Purpose of System Messages

From HuggingFace documentation:
> System messages "define behavior, personality, constraints, and expertise"

System messages set:
- Role and personality
- Constraints and limitations
- Response style
- Domain expertise

### Effective System Prompts

**Recipe extraction example** (from OpenAI Cookbook):

```json
{
  "role": "system",
  "content": "You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."
}
```

**Best practices**:
- Be specific about the task
- Define output format expectations
- Set behavioral boundaries
- Keep concise (system messages add to every example)

### System Message vs. Instruction

**System message**: Persistent context for all interactions
**Instruction**: Specific task in individual example

Example combining both:

```json
{
  "messages": [
    {"role": "system", "content": "You are a code review assistant"},
    {"role": "user", "content": "Review this function for bugs: def add(a,b): return a+b"},
    {"role": "assistant", "content": "The function looks correct but..."}
  ]
}
```

## Instruction Quality Guidelines

### Clarity and Specificity

**Poor instruction**:
```
"Do something with this text"
```

**Good instruction**:
```
"Extract all email addresses from the following text and return them as a JSON array"
```

### Consistency in Instructions

From OpenAI's guidance:
> "Start with 30-50 well-pruned examples"

"Well-pruned" means:
- Consistent instruction phrasing
- Uniform output formatting
- Clear task boundaries
- No ambiguous requirements

### Instruction Complexity Progression

Structure datasets with varying complexity:

**Simple** (single-step):
```
"Translate this sentence to Spanish: Hello world"
```

**Medium** (multi-step):
```
"Translate the following English paragraph to Spanish, then summarize the Spanish version in one sentence"
```

**Complex** (reasoning required):
```
"Given this code snippet, identify bugs, explain why they're bugs, and provide corrected code with comments"
```

## Multi-Turn Instruction Patterns

### Context Preservation

Each turn should build on previous context:

```json
{
  "messages": [
    {"role": "user", "content": "Create a Python list with numbers 1-5"},
    {"role": "assistant", "content": "numbers = [1, 2, 3, 4, 5]"},
    {"role": "user", "content": "Now filter for even numbers"},
    {"role": "assistant", "content": "even_numbers = [n for n in numbers if n % 2 == 0]"}
  ]
}
```

### Error Correction Patterns

Include examples of clarification and correction:

```json
{
  "messages": [
    {"role": "user", "content": "Write a function to sort a list"},
    {"role": "assistant", "content": "What type of sorting would you like? Ascending, descending, or custom?"},
    {"role": "user", "content": "Ascending"},
    {"role": "assistant", "content": "def sort_ascending(lst):\n    return sorted(lst)"}
  ]
}
```

### Follow-up Question Handling

Train models to handle clarifications:

```json
{
  "messages": [
    {"role": "user", "content": "How do I deploy a web app?"},
    {"role": "assistant", "content": "What technology stack are you using? (e.g., Node.js, Python Flask, React)"},
    {"role": "user", "content": "Python Flask"},
    {"role": "assistant", "content": "Here are the steps to deploy a Flask app..."}
  ]
}
```

## Instruction Dataset Construction Methods

### Manual Curation

**Advantages**:
- High quality control
- Domain-specific accuracy
- Consistent formatting

**Process**:
1. Define task categories
2. Write clear instructions
3. Create reference outputs
4. Review and refine

### Data Annotation

**Approach**:
- Provide annotators with instruction templates
- Review for consistency
- Implement quality checks

**Tools** (from HuggingFace ecosystem):
- Label Studio
- Argilla
- Prodigy

### Synthetic Generation

**Method**: Use existing LLMs to generate training data

**Example workflow**:
1. Provide seed instructions
2. Generate variations
3. Validate outputs
4. Filter for quality

**Caution**: Avoid model collapse by mixing with human-created data

### Conversion from Existing Data

Transform non-instruction data into instruction format:

**Original**:
```
Q: What is the capital of France?
A: Paris
```

**Converted**:
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"}
  ]
}
```

## Unsloth Instruction Requirements

From Unsloth documentation:

### Required Fields

**{INPUT}**: The instruction or prompt field
**{OUTPUT}**: The model's expected response
**{SYSTEM}**: Optional system prompt

### Example Template Usage

```python
instruction_template = """Below is an instruction. Write a response.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""
```

### Conversion Tools

Unsloth provides `to_sharegpt()` function to merge columns:

```python
dataset = to_sharegpt(
    dataset,
    merged_prompt="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
    output_column_name="conversations"
)
```

## Instruction Tuning for Specific Tasks

### Code Generation

**System message**:
```
"You are an expert programmer who writes clean, efficient, well-documented code"
```

**Instruction pattern**:
```
"Write a [language] function that [specific task]. Include error handling and docstrings."
```

### Creative Writing

**System message**:
```
"You are a creative writing assistant who helps with story development and character creation"
```

**Instruction pattern**:
```
"Write a [length] [genre] story about [topic]. Include [specific elements]."
```

### Data Analysis

**System message**:
```
"You are a data analyst who interprets data and provides actionable insights"
```

**Instruction pattern**:
```
"Analyze the following dataset and provide: 1) Summary statistics, 2) Key trends, 3) Recommendations"
```

## Common Instruction Tuning Mistakes

### Vague Instructions

**Problem**: "Make this better"
**Solution**: "Improve code readability by adding comments and using descriptive variable names"

### Inconsistent Formats

**Problem**: Mixing Alpaca and ChatML in same dataset
**Solution**: Choose one format and standardize all examples

### Missing Context

**Problem**: Instructions that reference undefined entities
**Solution**: Ensure all necessary context is in the instruction or previous messages

### Output Misalignment

**Problem**: Instruction asks for JSON but output is plain text
**Solution**: Ensure outputs precisely match instruction requirements

## References

- [HuggingFace TRL Dataset Formats](https://huggingface.co/docs/trl/main/dataset_formats)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [OpenAI How to Fine-Tune Chat Models](https://cookbook.openai.com/examples/how_to_finetune_chat_models)
- [LLM Dataset Formats 101](https://huggingface.co/blog/tegridydev/llm-dataset-formats-101-hugging-face)
- [Unsloth Chat Templates](https://docs.unsloth.ai/basics/chat-templates)
