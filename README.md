# LLM Fine-Tuning Dataset Engineering Manual

**Comprehensive manual for LLM fine-tuning dataset creation - compiled documentation from HuggingFace, OpenAI, Anthropic, EleutherAI, and Unsloth.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a comprehensive manual on creating, preparing, and validating datasets for Large Language Model (LLM) fine-tuning. Unlike general machine learning resources, this guide focuses specifically on **dataset engineering** - the practical aspects of building high-quality training data.

### What Makes This Different?

Rather than offering generic advice, this manual **compiles and organizes authoritative documentation** from the leading organizations in AI:

- **HuggingFace** - Open-source fine-tuning platform
- **OpenAI** - ChatGPT and GPT-4 creators
- **Anthropic** - Constitutional AI and Claude developers
- **EleutherAI/LAION** - Large-scale dataset research
- **Unsloth** - Optimized fine-tuning framework

All content is sourced directly from official documentation, research papers, and published guides with full attribution and links.

## Quick Start

### Read Online

📖 **[Start with the Manual](MANUAL.md)** - Complete table of contents and navigation

### Jump to Chapters

1. **[Introduction](01-introduction/README.md)** - Overview and how to use this manual
2. **[Dataset Design](02-dataset-design/README.md)** - Core structures and principles
3. **[Instruction Tuning](03-instruction-tuning/README.md)** - Instruction dataset construction
4. **[Data Quality](04-data-quality/README.md)** - Filtering, cleaning, and deduplication
5. **[Formatting Schemas](05-formatting-schemas/README.md)** - JSONL, ChatML, ShareGPT formats
6. **[Tools and Pipelines](06-tools-and-pipelines/README.md)** - Practical implementations
7. **[Unsloth-Specific](07-unsloth-specific/README.md)** - Unsloth requirements and optimizations
8. **[Evaluation](08-evaluation/README.md)** - Quality metrics and validation

📚 **[Complete Source References](references/SOURCES.md)** - All original documentation links

## Who Is This For?

- **ML Engineers** implementing fine-tuning pipelines
- **Data Engineers** preparing training datasets
- **Researchers** designing custom training data
- **Teams** building production LLM applications

## What You'll Learn

### Format Specifications
- JSONL structure for fine-tuning
- ChatML format (OpenAI standard)
- ShareGPT conversation format
- Model-specific templates (Llama 3, Mistral, etc.)

### Quality Control
- Perplexity-based filtering (EleutherAI)
- CLIP similarity thresholds (LAION)
- Deduplication methods (MinHash LSH, Bloom filters)
- Language detection and toxicity filtering

### Instruction Tuning
- Alpaca format specifications
- System message design
- Multi-turn conversation patterns
- Synthetic data generation

### Tools and Implementation
- HuggingFace Datasets library
- TRL (Transformer Reinforcement Learning)
- OpenAI data preparation toolkit
- Unsloth optimization techniques

### Evaluation Methods
- Pre-training statistics
- Loss curve monitoring
- Benchmark evaluation (MMLU, HellaSwag, etc.)
- Cost estimation

## Key Insights from Sources

### From HuggingFace
> "JSONL is widely adopted in modern LLM pipelines because each line is a separate JSON object—ideal for nested and multi-field data"

### From OpenAI
> "You can begin with even 30-50 well-pruned examples, and you should see performance continue to scale linearly as you increase the size of the training set"

### From Anthropic
> Constitutional AI uses "a list of rules or principles" to guide self-improvement, enabling scalable dataset creation without human labels

### From EleutherAI
> "Train solely with adult and harmful textual data, then select documents having a perplexity value above a given threshold" - inverted perplexity filtering

### From Unsloth
> "Effective fine-tuning requires at least 100 rows; 1,000+ rows produces optimal results. Dataset quality matters more than quantity."

## Project Structure

```
fine-tuning-dataset-manual/
├── MANUAL.md                 # Main entry point with full table of contents
├── README.md                 # This file
├── LICENSE                   # MIT License
├── 01-introduction/          # Overview and getting started
├── 02-dataset-design/        # Core principles and structures
├── 03-instruction-tuning/    # Instruction dataset creation
├── 04-data-quality/          # Filtering and cleaning methods
├── 05-formatting-schemas/    # Format specifications
├── 06-tools-and-pipelines/   # Implementation guides
├── 07-unsloth-specific/      # Unsloth requirements
├── 08-evaluation/            # Quality metrics and validation
└── references/               # Complete source attribution
    └── SOURCES.md
```

## Use Cases

### For Individual Projects
- Reference while building datasets
- Validate format compliance
- Implement quality filtering
- Choose appropriate formats

### For Team Workflows
- Standardize dataset creation processes
- Document format requirements
- Share best practices
- Integrate with CI/CD pipelines

### For Kestra/Workflow Automation
- Link specific sections in task documentation
- Use code examples in pipeline tasks
- Implement validation checks
- Reference evaluation methods

## Contributing

This manual compiles publicly available documentation. To improve it:

1. **Report issues** - Found outdated info or broken links? Open an issue
2. **Suggest additions** - New authoritative sources? Let us know
3. **Update content** - Official docs changed? Submit a PR

All additions must cite authoritative sources with full attribution.

## Sources and Attribution

This manual aggregates documentation from:

- [HuggingFace](https://huggingface.co/docs) - Datasets, TRL, Transformers
- [OpenAI](https://platform.openai.com/docs) - Fine-tuning guides and cookbook
- [Anthropic Research](https://www.anthropic.com/research) - Constitutional AI
- [EleutherAI](https://www.eleuther.ai/) - The Pile, evaluation harness
- [LAION](https://laion.ai/) - Large-scale dataset filtering
- [Unsloth](https://docs.unsloth.ai/) - Optimized fine-tuning

See [SOURCES.md](references/SOURCES.md) for complete attribution and links.

## License

This compilation is released under the MIT License. See [LICENSE](LICENSE) for details.

**Important**: Individual source materials maintain their original licenses:
- HuggingFace (Apache 2.0)
- OpenAI Cookbook (MIT)
- Research papers (as published)

Always consult original sources for authoritative and current information.

## Staying Updated

Fine-tuning practices evolve rapidly. To stay current:

- ⭐ **Star this repo** for updates
- 📖 **Bookmark source docs** (linked in each chapter)
- 🔔 **Follow official blogs** (HuggingFace, OpenAI, Anthropic)
- 💬 **Join communities** (HuggingFace Forums, Discord servers)

## Citation

If you use this manual in your work, please cite:

```bibtex
@misc{llm_finetuning_manual_2025,
  title={LLM Fine-Tuning Dataset Engineering Manual},
  author={Compiled from HuggingFace, OpenAI, Anthropic, EleutherAI, LAION, Unsloth},
  year={2025},
  url={https://github.com/not-qty/fine-tuning-dataset-manual}
}
```

And cite the original sources for specific methodologies (links provided throughout).

## Support

- 📖 **Read the docs**: Start with [MANUAL.md](MANUAL.md)
- 🔍 **Search**: Use Ctrl+F or GitHub search
- 💬 **Discuss**: Open an issue for questions
- 🐛 **Report bugs**: Issues with content or formatting

---

**Ready to start?** → [Open the Manual](MANUAL.md)

**Need something specific?** → Use the [Quick Reference](MANUAL.md#quick-reference) section

**Want to dive deep?** → Browse by [chapter](MANUAL.md#table-of-contents)
