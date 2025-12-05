# Data Quality, Filtering, and Cleaning

## Overview

Dataset quality directly impacts fine-tuning performance. This chapter covers filtering techniques, deduplication methods, quality heuristics, and cleaning strategies from EleutherAI, LAION, and other industry leaders.

## Why Data Quality Matters

From EleutherAI research:
> "Aggressive filtering can in fact lead to a decrease in model quality on downstream tasks"

The balance:
- **Too little filtering**: Noise, toxicity, and irrelevant data
- **Too much filtering**: Loss of diversity and edge cases

Goal: Remove harmful and low-quality data while preserving diversity

## Quality Filtering Heuristics

### CLIP-Based Filtering (LAION Approach)

LAION uses cosine similarity between text and image embeddings:

**LAION-400M methodology**:
- Calculate CLIP cosine similarity between text and image
- Threshold: 0.3 (determined through human evaluation)
- Drop samples below threshold

**LAION-5B refined thresholds**:
- English: 0.28 (CLIP B/32)
- Multilingual: 0.26 (MCLIP)
- Thresholds based on human inspection

**Adaptation for text datasets**:
- Use sentence embeddings instead of CLIP
- Calculate semantic coherence between instruction and response
- Filter examples with low semantic alignment

### Perplexity-Based Filtering

#### Standard Perplexity Filtering (CC-100 Method)

From The Pile documentation:
> "CC-100 uses perplexity based filtering, where a language model is trained on Wikipedia and all data with a perplexity too high or too low is discarded"

**Process**:
1. Train language model on high-quality reference corpus
2. Calculate perplexity for each candidate document
3. Discard outliers (too different from reference)

**Why it works**:
- **High perplexity**: Random, garbled, or off-topic text
- **Low perplexity**: Duplicate or near-duplicate of training data

#### Inverted Perplexity for Harmful Content

From "Perplexed by Quality" paper (EleutherAI research):

**Innovation**: Train on harmful/adult content, then select documents with high perplexity

**Method**:
1. Train language model solely on adult/harmful text
2. Evaluate perplexity on candidate documents
3. Select documents above threshold (dissimilar to harmful content)

**Advantage**:
> "Documents separate into two distinct groups, making threshold selection more straightforward and improving precision over traditional classification approaches"

**Application**:
- Handles multilingual, heterogeneous web data
- More robust than traditional classifiers on noisy data
- Effective at web scale

### FastText Classification

From EleutherAI Pile-CC filtering:

**Approach**:
> "Train a fasttext classifier to classify between Pile data and CC data"

**Process**:
1. Create positive class from high-quality data (e.g., OpenWebText2)
2. Create negative class from raw Common Crawl
3. Train FastText classifier
4. Apply to candidate documents

**Benefits**:
- Fast inference (critical for large datasets)
- Low memory footprint
- Effective with minimal features

## Deduplication Methods

### MinHash LSH (The Pile Approach)

From The Pile paper:
> "The dataset is processed to ensure minimal duplication using MinHashLSH techniques"

**How MinHash LSH works**:
1. Convert documents to character n-grams
2. Generate MinHash signatures
3. Group similar signatures using Locality-Sensitive Hashing
4. Remove near-duplicates within groups

**Parameters to tune**:
- N-gram size (typically 5-13)
- Number of hash functions
- Similarity threshold (0.7-0.9 common)

### Bloom Filter Deduplication (LAION)

LAION methodology:
> "LAION uses continuously updated bloom filters to drop samples that are already in the dataset"

**LAION-5B approach**:
- Deduplication criteria: URL-based
- Bloom filter tracks seen URLs
- Same image with different captions not considered duplicate

**Limitation**:
> "Same image with the same caption may sit at different URLs, causing duplicates"

**For text datasets**:
- Use content hash instead of URL
- Concatenate instruction + response for hash
- Update bloom filter continuously during dataset creation

### Embedding-Based Deduplication (SemDeDup)

Advanced technique from LAION research:

**Process**:
1. Generate embeddings for all samples (e.g., with CLIP or sentence transformers)
2. Cluster embeddings
3. Remove samples within similarity threshold
4. Keep representative sample from each cluster

**When to use**:
- Semantic duplicates (different wording, same meaning)
- High-quality datasets where exact matching insufficient
- Computational resources available for embedding generation

## Language Detection and Filtering

### FastText Language Identification

From EleutherAI tooling:

**Implementation**:
```python
import fasttext
model = fasttext.load_model('lid.176.bin')
predictions = model.predict(text)
language, confidence = predictions
```

**Best practices**:
- Set confidence threshold (e.g., 0.8)
- Handle code-switching in multilingual data
- Validate on mixed-language examples

### Language-Specific Quality Thresholds

LAION approach: Different thresholds per language

**Why**:
- Language representation varies
- Cultural context differs
- Script characteristics (e.g., logographic vs. alphabetic)

**Implementation**:
- Train separate quality models per language
- Adjust filtering thresholds based on validation
- Monitor distribution balance

## Toxicity and Safety Filtering

### Content Warning Detection

LAION datasets include detection scores for:
- NSFW content
- Toxic content
- Watermarks

**For instruction datasets**:
- Filter harmful instructions
- Remove unsafe response patterns
- Check for manipulation attempts

### Safety Classification

**Approach**:
1. Define safety categories (violence, hate speech, self-harm, etc.)
2. Use pre-trained safety classifiers (e.g., OpenAI Moderation API)
3. Apply thresholds per category
4. Human review borderline cases

**Balance considerations**:
- Remove clearly harmful content
- Preserve educational/medical content
- Allow refusals and safety explanations

## Data Validation and Format Checking

### Format Compliance (OpenAI Approach)

From OpenAI data prep toolkit:

**Essential validation checks**:
1. Each entry contains "messages" list
2. Every message has "role" and "content" keys
3. Valid roles: "system", "user", "assistant", "function"
4. Content is non-empty string
5. At least one assistant message per conversation

**Allowed message keys**:
> "role," "content," "weight," "function_call," and "name" only

**Python validation example**:
```python
def validate_message(msg):
    assert isinstance(msg, dict), "Message must be dict"
    assert "role" in msg, "Missing role"
    assert "content" in msg, "Missing content"
    assert msg["role"] in ["system", "user", "assistant", "function"]
    assert isinstance(msg["content"], str) and len(msg["content"]) > 0
    return True
```

### Token Count Validation

**OpenAI limits**: 16,385 tokens (examples truncated beyond this)

**Validation process**:
1. Count tokens using appropriate tokenizer (e.g., tiktoken)
2. Flag examples exceeding limit
3. Either truncate or split into multiple examples
4. Track token distribution statistics

### Statistical Analysis

From OpenAI data prep notebook:

**Key metrics to track**:
- Message count distribution per conversation
- Total tokens per example
- Assistant-specific token counts
- Examples exceeding token limits

**Why these matter**:
- Cost estimation (billable tokens)
- Training time prediction (based on dataset size)
- Balance checking (avoid skewed distributions)

## Contamination Prevention

### Test Set Leakage Detection

**Methods**:
1. N-gram overlap detection
2. Embedding similarity
3. Exact substring matching

**Best practices**:
- Check against common benchmarks (MMLU, HumanEval, etc.)
- Remove examples too similar to evaluation sets
- Document any potential overlaps

### Training Data Memorization

**Risks**:
- Model reproduces training examples verbatim
- Privacy concerns with personal information
- Copyright issues with reproduced content

**Mitigation**:
- Deduplicate aggressively
- Remove examples with PII
- Check for copyrighted content
- Use differential privacy techniques (advanced)

## Quality Metrics and Monitoring

### Diversity Metrics

**Lexical diversity**:
- Unique n-gram ratio
- Type-token ratio
- Vocabulary size

**Semantic diversity**:
- Embedding space coverage
- Topic distribution
- Domain representation

### Balance Metrics

**Check for**:
- Task type distribution
- Response length distribution
- Complexity levels
- Domain coverage

**Tools**:
```python
from collections import Counter
import numpy as np

def calculate_balance(dataset, field):
    counts = Counter([item[field] for item in dataset])
    values = list(counts.values())
    return {
        'gini': calculate_gini(values),
        'entropy': calculate_entropy(values),
        'max_ratio': max(values) / sum(values)
    }
```

### Quality Sampling

**Process**:
1. Sample 100-200 random examples
2. Manual review for quality
3. Track issues (format errors, poor responses, etc.)
4. Calculate quality score
5. Iterate filtering if below threshold

## Common Crawl Specific Filtering

From EleutherAI pile-cc-filtering:

**GPT-3 Paper methodology**:
1. Train classifier on high-quality vs. Common Crawl data
2. Apply to Common Crawl
3. Use jusText for HTML content extraction
4. Remove boilerplate and navigation

**Additional filters**:
- Minimum document length
- Maximum document length
- Language detection
- URL blacklists (adult sites, spam domains)

## Preprocessing Pipeline Example

**Recommended order**:

1. **Format validation**: Ensure structural correctness
2. **Language detection**: Filter to target languages
3. **Deduplication**: Remove exact and near duplicates
4. **Toxicity filtering**: Remove harmful content
5. **Quality filtering**: Apply perplexity/classifier
6. **Length filtering**: Remove too short/long examples
7. **Token validation**: Check token limits
8. **Final review**: Sample and manual check

## Tools and Libraries

### Data Quality Tools

- **fastText**: Language detection, classification
- **datasketch**: MinHash LSH implementation
- **sentence-transformers**: Semantic similarity
- **tiktoken**: Token counting (OpenAI)
- **jusText**: Content extraction from HTML

### HuggingFace Datasets Utilities

```python
from datasets import load_dataset

# Load and filter
dataset = load_dataset("your/dataset")
dataset = dataset.filter(lambda x: len(x['text']) > 100)
dataset = dataset.filter(lambda x: x['language'] == 'en')

# Deduplicate
dataset = dataset.unique(column='text')
```

## When to Filter vs. When to Preserve

### Preserve for diversity:
- Edge cases and rare patterns
- Domain-specific jargon
- Creative or unconventional responses
- Multi-lingual mixing (if intentional)

### Filter for quality:
- Garbled or corrupted text
- Clear spam or advertising
- Harmful or toxic content
- Exact duplicates

### Context matters:
- Medical/legal domains: Higher accuracy standards
- Creative writing: More permissive filtering
- Code generation: Validate syntax
- Chatbots: Natural conversation flow

## References

- [LAION-400M Dataset](https://laion.ai/blog/laion-400-open-dataset/)
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)
- [On the De-duplication of LAION-2B](https://arxiv.org/abs/2303.12733)
- [The Pile: An 800GB Dataset](https://arxiv.org/abs/2101.00027)
- [Perplexity-based Quality Filtering](https://arxiv.org/abs/2212.10440)
- [EleutherAI Pile-CC Filtering](https://github.com/EleutherAI/pile-cc-filtering)
- [OpenAI Data Preparation Tool](https://cookbook.openai.com/examples/chat_finetuning_data_prep)
