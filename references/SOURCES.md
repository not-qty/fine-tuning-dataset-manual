# Complete Source References

## HuggingFace Documentation

### Core Resources

**TRL Dataset Formats**
- URL: https://huggingface.co/docs/trl/main/dataset_formats
- Content: Dataset format specifications for SFT, DPO, and other training methods
- Topics: Language modeling, prompt-completion, preference, conversational formats

**Chat Templates**
- URL: https://huggingface.co/docs/transformers/main/en/chat_templating
- Content: Official chat template documentation
- Topics: apply_chat_template(), role/content structure, tokenization best practices

**LLM Dataset Formats 101**
- URL: https://huggingface.co/blog/tegridydev/llm-dataset-formats-101-hugging-face
- Content: Comprehensive guide to dataset formats
- Topics: CSV/TSV, JSON/JSONL, Parquet, raw text, format selection

**Fine-Tuning Guide (2024)**
- URL: https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
- Content: Practical fine-tuning tutorial with TRL
- Topics: Dataset preparation, trainer configuration, best practices

**Fine-Tuning Guide (2025)**
- URL: https://www.philschmid.de/fine-tune-llms-in-2025
- Content: Updated fine-tuning methods
- Topics: Latest techniques, optimizations, format updates

### Related Resources

**Datasets Library Documentation**
- URL: https://huggingface.co/docs/datasets
- Content: Complete datasets library reference
- Topics: Loading, processing, streaming, filtering

**TRL Documentation**
- URL: https://huggingface.co/docs/trl
- Content: Transformer Reinforcement Learning library
- Topics: SFTTrainer, DPOTrainer, RewardTrainer

**Evaluate Library**
- URL: https://huggingface.co/docs/evaluate
- Content: Model evaluation framework
- Topics: Metrics, benchmarks, evaluation methods

## OpenAI Documentation

### Cookbooks

**Chat Fine-Tuning Data Preparation**
- URL: https://cookbook.openai.com/examples/chat_finetuning_data_prep
- Content: Dataset validation and analysis tools
- Topics: Format checking, token counting, statistics, cost estimation

**How to Fine-Tune Chat Models**
- URL: https://cookbook.openai.com/examples/how_to_finetune_chat_models
- Content: Complete fine-tuning workflow
- Topics: Message structure, JSONL format, training guidelines

**DPO Fine-Tuning Guide**
- URL: https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide
- Content: Direct Preference Optimization methods
- Topics: Preference datasets, SFT vs DPO vs RFT comparison

**Function Calling Fine-Tuning**
- URL: https://cookbook.openai.com/examples/fine_tuning_for_function_calling
- Content: Training models for function calls
- Topics: Function schema, dataset format, validation

### Official Documentation

**Fine-Tuning Guide**
- URL: https://platform.openai.com/docs/guides/fine-tuning
- Content: Official fine-tuning documentation
- Topics: API usage, pricing, best practices

**OpenAI Cookbook Repository**
- URL: https://cookbook.openai.com/
- Content: Complete cookbook collection
- Topics: Examples, tutorials, advanced techniques

**GitHub Repository**
- URL: https://github.com/openai/openai-cookbook
- Content: Source code and notebooks
- Topics: Working examples, Jupyter notebooks

## Anthropic Research

### Constitutional AI

**Constitutional AI: Harmlessness from AI Feedback**
- URL: https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
- Content: Official research page
- Topics: AI feedback, self-critique, preference datasets

**arXiv Paper**
- URL: https://arxiv.org/abs/2212.08073
- Content: Full research paper
- Topics: RLAIF methodology, dataset construction, evaluation

**PDF Version**
- URL: https://www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/Anthropic_ConstitutionalAI_v2.pdf
- Content: Downloadable paper
- Topics: Complete methodology, results, appendices

### Related Resources

**HH-RLHF Dataset**
- URL: https://github.com/anthropics/hh-rlhf
- Content: Human preference dataset
- Topics: Helpful and harmless assistant training data

**RLHF Book Chapter on Constitutional AI**
- URL: https://rlhfbook.com/c/13-cai.html
- Content: Educational resource on Constitutional AI
- Topics: Explanation, context, applications

## EleutherAI & LAION

### Dataset Papers

**The Pile: An 800GB Dataset**
- URL: https://arxiv.org/abs/2101.00027
- Content: Massive language modeling dataset paper
- Topics: Dataset composition, quality metrics, preprocessing

**The Pile Paper PDF**
- URL: https://pile.eleuther.ai/paper.pdf
- Content: Complete research paper
- Topics: Filtering methods, deduplication, quality heuristics

**Perplexity-Based Quality Filtering**
- URL: https://arxiv.org/abs/2212.10440
- Content: Novel filtering methodology
- Topics: Inverted perplexity, adult content detection, multilingual filtering

**On the De-duplication of LAION-2B**
- URL: https://arxiv.org/abs/2303.12733
- Content: Deduplication techniques at scale
- Topics: CLIP feature compression, duplicate detection, efficiency

### LAION Resources

**LAION-400M Dataset**
- URL: https://laion.ai/blog/laion-400-open-dataset/
- Content: 400 million image-text pairs
- Topics: CLIP filtering, cosine similarity thresholds, bloom filters

**LAION-5B Dataset**
- URL: https://laion.ai/blog/laion-5b/
- Content: 5 billion image-text pairs
- Topics: Refined filtering, multilingual considerations, quality detection

### EleutherAI Tools

**Pile-CC Filtering Repository**
- URL: https://github.com/EleutherAI/pile-cc-filtering
- Content: Common Crawl filtering code
- Topics: FastText classification, quality filtering, implementation

**The Pile Repository**
- URL: https://github.com/EleutherAI/the-pile
- Content: Dataset construction code
- Topics: Preprocessing, filtering, component datasets

**LM Evaluation Harness**
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Content: Model evaluation framework
- Topics: Benchmarks, metrics, evaluation pipeline

**EleutherAI Papers & Blog**
- URL: https://www.eleuther.ai/papers-blog/tag/Datasets
- Content: Dataset-related research
- Topics: Various dataset papers and blog posts

## Unsloth Documentation

### Official Docs

**Datasets Guide**
- URL: https://docs.unsloth.ai/basics/datasets-guide
- Content: Complete dataset formatting guide
- Topics: Format types, packing, conversion, multi-turn

**Chat Templates**
- URL: https://docs.unsloth.ai/basics/chat-templates
- Content: Template application documentation
- Topics: {INPUT}/{OUTPUT} syntax, supported templates, standardize_sharegpt

**Fine-Tuning LLMs Guide**
- URL: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
- Content: End-to-end fine-tuning tutorial
- Topics: Setup, training, optimization, saving

### Community Resources

**Fine-Tune Llama 3.1 with Unsloth (HuggingFace Blog)**
- URL: https://huggingface.co/blog/mlabonne/sft-llama3
- Content: Practical tutorial
- Topics: Dataset preparation, training configuration, results

**Fine-Tune Llama 3.1 (Towards Data Science)**
- URL: https://towardsdatascience.com/fine-tune-llama-3-1-ultra-efficiently-with-unsloth-7196c7165bab/
- Content: In-depth tutorial
- Topics: VRAM optimization, packing, performance

**Unsloth GitHub Repository**
- URL: https://github.com/unslothai/unsloth
- Content: Source code and examples
- Topics: Implementation, chat templates, utilities

**Unsloth Chat Templates Source**
- URL: https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py
- Content: Template definitions
- Topics: All supported templates, syntax, mappings

## Additional Tools & Libraries

### Data Quality

**fastText Language Identification**
- URL: https://fasttext.cc/docs/en/language-identification.html
- Content: Language detection documentation
- Topics: Pre-trained models, usage, accuracy

**datasketch Library**
- URL: https://ekzhu.com/datasketch/
- Content: MinHash LSH implementation
- Topics: Deduplication, similarity search, scaling

**sentence-transformers**
- URL: https://www.sbert.net/
- Content: Sentence embeddings library
- Topics: Semantic similarity, clustering, applications

**Detoxify**
- URL: https://github.com/unitaryai/detoxify
- Content: Toxicity detection
- Topics: Content moderation, safety scoring

### Evaluation

**ROUGE Score**
- URL: https://github.com/google-research/google-research/tree/master/rouge
- Content: ROUGE metric implementation
- Topics: Summarization evaluation, scoring

**tiktoken**
- URL: https://github.com/openai/tiktoken
- Content: Fast BPE tokenizer
- Topics: Token counting, encoding, OpenAI models

**EleutherAI LM Evaluation Harness**
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Documentation: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/README.md
- Content: Comprehensive evaluation framework
- Topics: Benchmarks (MMLU, HellaSwag, etc.), metrics, automation

## Research Papers

### Foundational Papers

**Attention Is All You Need** (Transformers)
- URL: https://arxiv.org/abs/1706.03762
- Topics: Transformer architecture, attention mechanisms

**BERT: Pre-training of Deep Bidirectional Transformers**
- URL: https://arxiv.org/abs/1810.04805
- Topics: Pre-training methods, masked language modeling

**Language Models are Few-Shot Learners** (GPT-3)
- URL: https://arxiv.org/abs/2005.14165
- Topics: In-context learning, scaling laws

### Fine-Tuning Methods

**LoRA: Low-Rank Adaptation of Large Language Models**
- URL: https://arxiv.org/abs/2106.09685
- Topics: Parameter-efficient fine-tuning, LoRA methodology

**QLoRA: Efficient Finetuning of Quantized LLMs**
- URL: https://arxiv.org/abs/2305.14314
- Topics: 4-bit quantization, memory optimization

**Direct Preference Optimization** (DPO)
- URL: https://arxiv.org/abs/2305.18290
- Topics: Preference learning, alignment without RL

### Dataset Quality

**Scaling Laws for Neural Language Models**
- URL: https://arxiv.org/abs/2001.08361
- Topics: Data quality vs quantity, optimal allocation

**Data Selection for Language Models via Importance Resampling**
- URL: https://arxiv.org/abs/2302.03169
- Topics: Data selection, quality metrics

## Community Resources

### Discussion Forums

**HuggingFace Forums - Datasets Section**
- URL: https://discuss.huggingface.co/c/datasets/10
- Content: Community discussions
- Topics: Format questions, troubleshooting, best practices

**EleutherAI Discord**
- URL: https://www.eleuther.ai/ (link on website)
- Content: Active community
- Topics: Technical discussions, research collaboration

### Educational Content

**Common Pile Blog**
- URL: https://blog.eleuther.ai/common-pile/
- Content: Dataset construction blog post
- Topics: Modern approaches, lessons learned

**Lil'Log - Reducing Toxicity in Language Models**
- URL: https://lilianweng.github.io/posts/2021-03-21-lm-toxicity/
- Content: Toxicity mitigation techniques
- Topics: Filtering methods, detoxification approaches

## Dataset Examples

### Available on HuggingFace Hub

**Alpaca Dataset**
- URL: https://huggingface.co/datasets/tatsu-lab/alpaca
- Content: 52K instruction-following examples
- Format: Instruction-input-output

**ShareGPT Datasets**
- Various URLs on HuggingFace Hub
- Content: Real conversation data
- Format: ShareGPT (from/value)

**OpenAssistant Conversations**
- URL: https://huggingface.co/datasets/OpenAssistant/oasst1
- Content: Human-generated conversations
- Format: Message trees with rankings

**Anthropic HH-RLHF**
- URL: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Content: Helpful and harmless preferences
- Format: Chosen/rejected pairs

## Stay Updated

### Official Blogs

- **HuggingFace Blog**: https://huggingface.co/blog
- **OpenAI Blog**: https://openai.com/blog
- **Anthropic News**: https://www.anthropic.com/news
- **EleutherAI Blog**: https://blog.eleuther.ai/

### Paper Tracking

- **arXiv AI Section**: https://arxiv.org/list/cs.AI/recent
- **Papers with Code**: https://paperswithcode.com/
- **AI Alignment Forum**: https://www.alignmentforum.org/

### Social Media

- Follow key researchers on Twitter/X
- Join ML Discord communities
- Subscribe to newsletter aggregators (e.g., The Batch, Import AI)

---

**Note**: URLs and content were accurate as of December 2025. Always check official sources for the most current information.
