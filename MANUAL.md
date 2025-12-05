# LLM Fine-Tuning Dataset Engineering Manual

**Comprehensive Guide to Creating, Preparing, and Validating Datasets for Large Language Model Fine-Tuning**

Compiled from official documentation by HuggingFace, OpenAI, Anthropic, EleutherAI, LAION, and Unsloth

---

## Table of Contents

### [Chapter 1: Introduction](01-introduction/README.md)
**What This Manual Covers**
- Purpose and scope of the manual
- Who this guide is for
- How to navigate the content
- Prerequisites and assumptions

**Key Topics:**
- Dataset engineering fundamentals
- Why quality matters in fine-tuning
- Overview of authoritative sources
- Quick navigation guide

---

### [Chapter 2: Dataset Design](02-dataset-design/README.md)
**Core Dataset Types and Structures**
- Language modeling format
- Instruction format (Alpaca-style)
- Conversational format (ChatML, ShareGPT)
- Preference format (for RLHF/DPO)

**Design Principles:**
- Dataset size requirements (minimum, optimal)
- Message role definitions (system, user, assistant)
- Multi-turn conversation patterns
- Token efficiency considerations
- Data mixing strategies
- Validation split design

**Key Insights:**
- Start with 30-50 well-curated examples (OpenAI)
- Quality matters more than quantity
- 1,000+ examples for production (Unsloth)
- Balance diversity with consistency

---

### [Chapter 3: Instruction Tuning](03-instruction-tuning/README.md)
**Instruction Dataset Construction**
- Alpaca format specifications
- ChatML format (OpenAI standard)
- ShareGPT format for conversations
- System message design patterns

**Instruction Quality:**
- Clarity and specificity guidelines
- Consistency in phrasing
- Complexity progression (simple to complex)
- Multi-turn instruction patterns

**Construction Methods:**
- Manual curation workflows
- Data annotation processes
- Synthetic generation techniques
- Converting existing data

**Unsloth Requirements:**
- {INPUT}, {OUTPUT}, {SYSTEM} placeholders
- Template usage and syntax
- Conversion tools (to_sharegpt)

---

### [Chapter 4: Data Quality](04-data-quality/README.md)
**Quality Filtering Heuristics**
- CLIP-based filtering (LAION approach)
- Perplexity-based filtering (EleutherAI)
- Inverted perplexity for harmful content
- FastText classification

**Deduplication Methods:**
- MinHash LSH (The Pile approach)
- Bloom filter deduplication (LAION)
- Embedding-based semantic deduplication
- URL and content-based strategies

**Language and Safety:**
- FastText language identification
- Language-specific quality thresholds
- Toxicity and safety filtering
- Content warning detection

**Validation and Metrics:**
- Format compliance checking (OpenAI toolkit)
- Token count validation
- Statistical analysis
- Contamination prevention

**Key Thresholds:**
- LAION-400M: CLIP similarity > 0.3
- LAION-5B: English > 0.28, Multilingual > 0.26
- Confidence thresholds based on human evaluation

---

### [Chapter 5: Formatting Schemas](05-formatting-schemas/README.md)
**JSONL (JSON Lines) Format**
- Structure and syntax
- When to use JSONL
- Creating JSONL files
- Common errors to avoid

**ChatML Format (OpenAI)**
- Role/content structure
- Special tokens (<|im_start|>, <|im_end|>)
- Function calling extensions
- Validation checklist

**ShareGPT Format**
- Conversations array structure
- from/value key specifications
- Converting to ChatML
- Multi-turn patterns

**Model-Specific Templates:**
- Llama 3 format with special tokens
- Mistral Instruct format
- Alpaca template (legacy)
- Vicuna format

**HuggingFace Integration:**
- apply_chat_template() usage
- tokenize parameter
- add_generation_prompt parameter
- Avoiding token duplication

**Dataset Packing:**
- What is packing and why use it
- Memory efficiency benefits
- When to enable/disable
- Implementation examples

---

### [Chapter 6: Tools and Pipelines](06-tools-and-pipelines/README.md)
**HuggingFace Datasets Library**
- Loading datasets (local, Hub, streaming)
- Operations (filtering, mapping, selecting)
- Shuffling and splitting
- Deduplication and saving

**HuggingFace TRL:**
- SFTTrainer for supervised fine-tuning
- DPOTrainer for preference optimization
- RewardTrainer for reward modeling
- Dataset format compatibility

**OpenAI Tools:**
- Data preparation scripts
- Format validation functions
- Token counting utilities
- Cost estimation methods

**Data Quality Tools:**
- fastText for language detection
- datasketch for MinHash deduplication
- sentence-transformers for semantic similarity
- Implementation examples

**Complete Pipeline Example:**
- End-to-end dataset preparation
- Validation → Filtering → Deduplication
- Length filtering → Splitting → Template application
- Saving and documentation

---

### [Chapter 7: Unsloth-Specific Requirements](07-unsloth-specific/README.md)
**Why Unsloth Formatting Matters**
- 2x faster training, 60% less memory
- Specific format requirements
- Optimization dependencies

**Required Field Placeholders:**
- {INPUT}: Instruction/prompt (required)
- {OUTPUT}: Expected response (required)
- {SYSTEM}: System prompt (optional)

**Supported Formats:**
- Raw corpus for pretraining
- Instruction format (Alpaca-style)
- ShareGPT for conversations
- ChatML (OpenAI standard)

**Chat Template Application:**
- Checking available templates
- Applying templates to tokenizer
- Template selection guide (llama-3, chatml, alpaca, mistral)

**ShareGPT to ChatML Conversion:**
- standardize_sharegpt() function
- Automatic role/key conversion
- Complete workflow example

**Multi-Turn Conversation Extension:**
- conversation_extension parameter
- Merging single-turn into multi-turn
- Benefits and use cases
- When to enable/disable

**Dataset Packing in Unsloth:**
- How packing improves efficiency
- Memory and speed benefits
- Implementation with SFTTrainer
- Considerations and cautions

**Complete Training Pipeline:**
- Model loading with 4-bit quantization
- LoRA adapter application
- Dataset preparation and formatting
- Trainer configuration
- Training and saving

**VRAM-Aware Design:**
- Memory considerations by GPU
- Recommendations for 24GB/16GB/12GB VRAM
- Optimizing dataset for VRAM constraints

---

### [Chapter 8: Evaluation](08-evaluation/README.md)
**Pre-Training Evaluation**
- Dataset statistics analysis
- Format compliance checking
- Diversity metrics (lexical, semantic)
- Balance metrics across dimensions

**Training-Time Evaluation:**
- Loss curve monitoring
- Overfitting detection
- Checkpoint selection strategies
- Early stopping implementation

**Post-Training Evaluation:**
- Qualitative assessment (manual review)
- Quantitative metrics (perplexity, ROUGE, BLEU)
- Task-specific evaluation
- A/B testing methodology

**Benchmark Evaluation:**
- Common benchmarks (MMLU, HellaSwag, TruthfulQA, GSM8K)
- Code generation (HumanEval, MBPP)
- Instruction following (IFEval, MT-Bench)
- EleutherAI Evaluation Harness usage

**Safety and Alignment:**
- Constitutional AI self-critique
- Toxicity detection (Detoxify)
- Content moderation
- Alignment evaluation

**Cost Analysis:**
- Training cost estimation (OpenAI)
- Token counting for pricing
- Budget optimization strategies

**Continuous Monitoring:**
- Production metrics tracking
- Response latency, error rates
- User satisfaction scores
- Prometheus integration

**Best Practices:**
- Dos and Don'ts checklist
- Multi-metric evaluation
- Held-out validation
- Documentation requirements

---

### [References: Complete Source List](references/SOURCES.md)
**HuggingFace Documentation**
- TRL, Transformers, Datasets library
- Chat templates, format guides
- Community tutorials and blogs

**OpenAI Documentation**
- Fine-tuning guides and cookbooks
- Data preparation tools
- API documentation

**Anthropic Research**
- Constitutional AI paper
- RLAIF methodology
- HH-RLHF dataset

**EleutherAI & LAION**
- The Pile dataset paper
- Perplexity-based filtering
- Deduplication techniques
- LAION-400M and LAION-5B

**Unsloth Documentation**
- Datasets guide
- Chat templates
- Fine-tuning tutorials

**Tools and Libraries**
- fastText, datasketch, sentence-transformers
- tiktoken, Detoxify, ROUGE
- Evaluation frameworks

**Research Papers**
- Foundational papers (Transformers, BERT, GPT-3)
- Fine-tuning methods (LoRA, QLoRA, DPO)
- Dataset quality research

---

## Quick Reference

### Format Decision Tree

**Starting from scratch?** → Chapter 2 (Dataset Design)

**Have data, need format?** → Chapter 5 (Formatting Schemas)

**Using Unsloth?** → Chapter 7 (Unsloth-Specific)

**Need to clean data?** → Chapter 4 (Data Quality)

**Setting up pipeline?** → Chapter 6 (Tools and Pipelines)

**Ready to evaluate?** → Chapter 8 (Evaluation)

### Common Tasks Quick Links

**Create instruction dataset:** → Chapter 3
- Alpaca format: Section "Alpaca Format"
- ChatML format: Section "ChatML Format"
- Multi-turn: Section "Multi-Turn Instruction Patterns"

**Apply chat template:** → Chapter 5
- HuggingFace: Section "HuggingFace Integration"
- Unsloth: Chapter 7, Section "Chat Template Application"

**Filter and clean:** → Chapter 4
- Language detection: Section "Language Detection and Filtering"
- Deduplication: Section "Deduplication Methods"
- Quality filtering: Section "Quality Filtering Heuristics"

**Validate format:** → Chapter 4, Chapter 8
- Format checking: Chapter 4, Section "Data Validation"
- Statistical analysis: Chapter 8, Section "Pre-Training Evaluation"

**Estimate costs:** → Chapter 8
- OpenAI pricing: Section "Cost Analysis"
- Token counting: Chapter 6, "OpenAI Tools"

### Key Takeaways by Source

**HuggingFace:**
- Use JSONL for fine-tuning datasets
- apply_chat_template() with proper parameters
- Set add_generation_prompt=False for training
- Enable dataset packing for efficiency

**OpenAI:**
- Start with 30-50 high-quality examples
- Validate format before training
- Use validation set to detect overfitting
- Examples capped at 16,385 tokens (truncated beyond)

**Anthropic:**
- Constitutional AI for preference datasets
- RLAIF (RL from AI Feedback)
- Self-critique and revision methodology
- AI-generated comparison data

**EleutherAI/LAION:**
- Perplexity filtering for quality control
- MinHash LSH for deduplication at scale
- Bloom filters for incremental dedup
- CLIP similarity for multimodal filtering

**Unsloth:**
- Minimum 100 examples, optimal 1,000+
- Use {INPUT}, {OUTPUT}, {SYSTEM} placeholders
- standardize_sharegpt() for format conversion
- conversation_extension for multi-turn training
- Dataset packing improves efficiency 2-3x

### Format Comparison Table

| Format | Use Case | Structure | Tools |
|--------|----------|-----------|-------|
| **JSONL** | All fine-tuning | One JSON per line | HuggingFace, OpenAI, Unsloth |
| **ChatML** | Chat models | role/content messages | OpenAI, HuggingFace |
| **ShareGPT** | Conversations | from/value messages | Unsloth (with conversion) |
| **Alpaca** | Instructions | instruction/input/output | Unsloth, legacy tools |
| **Preference** | RLHF/DPO | prompt/chosen/rejected | TRL DPOTrainer |

### Minimum Dataset Sizes

- **Proof of concept:** 30-50 examples (OpenAI)
- **Minimum viable:** 100 examples (Unsloth)
- **Production quality:** 1,000+ examples (Unsloth)
- **Optimal results:** 10,000+ examples

### Token Limits

- **OpenAI:** 16,385 tokens per example (truncated)
- **Common practice:** 2,048-4,096 tokens
- **Unsloth recommended:** 1,024-2,048 (VRAM dependent)

### Quality Thresholds

- **CLIP similarity:** 0.28-0.30 (LAION)
- **Language confidence:** >0.8 (fastText)
- **Perplexity:** Model-specific, validate on sample

---

## Getting Started

### First-Time Users

1. **Read Chapter 1** to understand scope and sources
2. **Review Chapter 2** for design principles
3. **Choose format** from Chapter 5 based on your use case
4. **Follow Chapter 6** to set up your pipeline
5. **Validate** using Chapter 4 and Chapter 8 tools

### Experienced Users

- Use **Table of Contents** to jump to specific topics
- Reference **Quick Reference** for common tasks
- Consult **References** for original source materials
- Use **Search** (Ctrl+F) for specific terms or concepts

### Integration with CommunityLLM/Kestra

This manual serves as reference documentation for dataset preparation in Kestra workflows:

1. **Reference chapters** in task documentation
2. **Link specific sections** for validation steps
3. **Use code examples** from Chapter 6 for pipeline tasks
4. **Apply quality checks** from Chapter 4 in automated flows
5. **Implement evaluation** from Chapter 8 in CI/CD

---

## Contributing and Updates

This manual is compiled from publicly available documentation as of December 2025.

**To stay current:**
- Bookmark official documentation links (References chapter)
- Follow source organizations' blogs and updates
- Check for model-specific format changes
- Validate examples against latest library versions

**Found an issue?**
- Verify against original sources (links provided)
- Check for version updates in libraries
- Consult community forums for clarifications

---

## License and Attribution

This manual aggregates and organizes publicly available documentation from:
- **HuggingFace** (Apache 2.0 License)
- **OpenAI** (MIT License for cookbook examples)
- **Anthropic** (Published research papers)
- **EleutherAI** (Apache 2.0 License)
- **LAION** (Public datasets and papers)
- **Unsloth** (Apache 2.0 License)

All credit for methodologies, techniques, and original content goes to the respective organizations.

This compilation is intended for educational and reference purposes. Always consult original sources for the most current and authoritative information.

---

## Version Information

**Manual Version:** 1.0
**Compilation Date:** December 2025
**Content Sources:** December 2025 and earlier

**Covered Frameworks:**
- HuggingFace Transformers, TRL, Datasets (latest)
- OpenAI API (gpt-3.5-turbo, gpt-4 fine-tuning)
- Unsloth (latest optimizations for Llama 3/3.1)
- EleutherAI/LAION (research publications through 2024)

**Model Coverage:**
- Llama 3, Llama 3.1
- Mistral, Mixtral
- GPT-3.5-turbo, GPT-4
- Phi, Gemma, Qwen
- General fine-tuning principles applicable to all modern LLMs

---

**Ready to start? Begin with [Chapter 1: Introduction](01-introduction/README.md)**

For quick tasks, see the **Quick Reference** section above.

For specific questions, use Ctrl+F to search or navigate via the **Table of Contents**.
