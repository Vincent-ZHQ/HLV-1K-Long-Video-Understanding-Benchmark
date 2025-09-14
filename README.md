# 🎬 HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ICME%202025-blue)](https://arxiv.org/abs/2501.01645)
[![Dataset](https://img.shields.io/badge/Dataset-1009%20Videos-green)](https://github.com/Vincent-ZHQ/HLV-1K-Long-Video-Understanding-Benchmark)
[![QA Pairs](https://img.shields.io/badge/QA%20Pairs-14847-orange)](https://github.com/Vincent-ZHQ/HLV_1K)

</div>

## 📖 Introduction

HLV-1K is a comprehensive benchmark designed to evaluate the capabilities of multimodal large language models (MLLMs) in understanding hour-long videos with **time-specific queries**. Unlike existing video understanding benchmarks that focus on short clips, HLV-1K addresses the critical challenge of long-term video comprehension by providing:

- **🕐 Hour-long Videos**: 1,009 videos with an average duration of 1 hour
- **📊 Diverse Reasoning Tasks**: 14,847 QA and MCQA pairs across multiple reasoning levels
- **⏰ Time-specific Queries**: Questions that require understanding of specific temporal segments
- **🎯 Multi-level Evaluation**: Frame-level, within-event, cross-event, and long-term reasoning

As video content becomes increasingly prevalent and lengthy, HLV-1K provides a robust evaluation framework for assessing models' ability to comprehend and reason about extended video sequences with precise temporal understanding.

## Leaderboard

Accuracy scores on HLV-1K are presented on frame-level, within-event-level, cross-event-level and long-term-level.

| **#** | **Model** | **LLM  <br>Params** | **Frames** | **Date** | **Frame-level** | **Within-event-level** | **Cross-event-level** | **Long-term-level** | **Overall** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3   | **[LLaVA-Video](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2)** | 72B | 120 | 2025-01-03 | **84.41** | **78.43** | 80.10 | 75.65 | 78.93 |
| 2   | **[LLaVA-OneVision](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-sft)** | 72B | 120 | 2025-01-03 | **80.33** | **75.06** | 77.25 | 68.74 | 74.01 |
| 1   | **[Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct)** | 72B | 120 | 2025-01-03 | **61.44** | **66.83** | 66.96 | 67.17 | 65.78 |
| 4   | **[Kangaroo](https://huggingface.co/KangarooGroup/kangaroo)** | 8B  | 120 | 2025-01-03 | **75.23** | **63.57** | 65.04 | 54.60 | 62.71 |
| 6   | **[Gemini 1.5 Pro](https://deepmind.google/technologies/gemini/pro/)** | \-  | 120 | 2025-01-03 | **60.39** | **64.46** | 63.08 | 62.37 | 62.41 |
| 2   | **[LongVA](https://huggingface.co/lmms-lab/LongVA-7B)** | 7B  | 120 | 2025-01-03 | **67.89** | **59.12** | 61.37 | 59.67 | 61.74 |
| 1   | **[InternVL2.5](https://huggingface.co/OpenGVLab/InternVL2_5-8B)** | 8B  | 120 | 2025-01-03 | **60.72** | **65.02** | 62.73 | 59.34 | 61.24 |
| 5   | **[GPT-4o](https://openai.com/index/hello-gpt-4o/)** | \-  | 120 | 2025-01-03 | **53.88** | **59.08** | 56.64 | 54.37 | 55.48 |
| 4   | **[Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)** | \-  | 20  | 2025-01-03 | **26.21** | **23.98** | 27.73 | 28.89 | 27.24 |


## 📊 Benchmark Details

### 🎯 Key Features

- **📹 Video Scale**: 1,009 hour-long videos (average duration: ~1 hour)
- **❓ Question Diversity**: 14,847 QA and MCQA pairs with time-specific queries
- **🔍 Multi-level Reasoning**: Four distinct reasoning levels for comprehensive evaluation
- **⏱️ Temporal Precision**: Questions anchored to specific time segments within videos

### 📈 Dataset Statistics

| **Metric** | **Count** | **Percentage** |
|------------|-----------|----------------|
| **Total Videos** | 1,009 | 100% |
| **Total QA Pairs** | 14,847 | 100% |
| **QA Type** | | |
| - Multiple Choice (MCQA) | 10,533 | 70.9% |
| - Open-ended (QA) | 4,314 | 29.1% |
| **Reasoning Level** | | |
| - Long-term | 6,213 | 41.8% |
| - Frame-level | 3,335 | 22.5% |
| - Cross-event | 2,809 | 18.9% |
| - Within-event | 2,490 | 16.8% |

### 🎭 Task Distribution

| **Task Type** | **Count** | **Percentage** |
|---------------|-----------|----------------|
| Object Understanding | 2,396 | 16.1% |
| Character Understanding | 2,191 | 14.8% |
| Speed Analysis | 1,701 | 11.5% |
| Camera Direction | 1,275 | 8.6% |
| Spatial Relationship | 1,255 | 8.5% |
| Attribute Change | 1,159 | 7.8% |
| Descriptive Scene | 964 | 6.5% |
| Action Understanding | 826 | 5.6% |
| Time Order | 730 | 4.9% |
| Plot Understanding | 649 | 4.4% |
| Temporal Relationship | 641 | 4.3% |
| Object Direction | 429 | 2.9% |
| Causal Reasoning | 322 | 2.2% |
| Scene Understanding | 212 | 1.4% |
| Counting | 97 | 0.7% |

### Data Examples
<img src="static/images/HLV_1K_F01.jpg" alt="HLV-1K" style="width:900px;height:700px;"> 

Benchmark construction and examples.

### Benchmark Statistics
<img src="static/images/HLV_1K_F00.jpg" alt="HLV-1K" style="width:900px;height:380px;"> 

HLV-1K: (a) Video category distribution, (b) Video duration distribution, and (c) Duration distribution of time-specific query.


<img src="static/images/HLV_1K_F02.jpg" alt="HLV-1K" style="width:900px;height:310px;"> 

HLV-1K: Distribution of benchmark annotations.

## 🔧 Dataset Construction

### 📝 Annotation Pipeline

HLV-1K employs a sophisticated annotation pipeline using GPT-4o for high-quality question generation:

1. **Frame Description Extraction**: Detailed descriptions of video frames at specific timestamps
2. **Event Summarization**: Coherent event descriptions spanning ~60 seconds with precise temporal boundaries
3. **Question Generation**: Time-specific questions across four reasoning levels
4. **Quality Assurance**: Multi-round validation to ensure question accuracy and temporal precision

### 🎯 Reasoning Levels

| **Level** | **Description** | **Example** |
|-----------|-----------------|-------------|
| **Frame-level** | Questions about specific frames | "What object is visible at 1290.0 seconds?" |
| **Within-event** | Questions within single events | "Are the individuals working at a fast pace between 1290.0-1350.0 seconds?" |
| **Cross-event** | Questions spanning multiple events | "What activity follows the circuit board assembly?" |
| **Long-term** | Questions requiring full video understanding | "What is the overall project being completed in this video?" |

### 📊 Evaluation Metrics

- **Accuracy**: Overall correctness across all question types
- **Level-wise Performance**: Accuracy breakdown by reasoning level
- **Task-specific Metrics**: Performance on different cognitive tasks
- **Temporal Understanding**: Accuracy on time-specific queries

## 🔍 Benchmark Comparison

<img src="static/images/datasets.png" alt="HLV-1K" style="width:900px;height:160px;"> 

## Experiment Results

### Different Question Types

<img src="static/images/HLV_1K_F03.jpg" alt="HLV-1K" style="width:900px;height:580px;"> 

Evaluation results of four representative MLLMs.

## Related Wrok

[Comprehensive-Long-Video-Understanding-Survey](https://github.com/Vincent-ZHQ/LV-LLMs)


## 🚀 Getting Started

### 📥 Dataset Download

The HLV-1K dataset is available for research purposes. Please follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vincent-ZHQ/HLV_1K.git
   cd HLV_1K
   ```

2. **Dataset structure**:
   ```
   HLV_1K/
   ├── dataset/           # 1,009 JSON files with QA pairs
   ├── static/           # Web interface assets
   ├── gpt_evaluation.py # Evaluation script
   └── index.html        # Web interface
   ```

### 🔧 Usage

1. **Load dataset**:
   ```python
   import json
   
   # Load a single video's QA pairs
   with open('dataset/video_id.json', 'r') as f:
       qa_pairs = json.load(f)
   
   for qa in qa_pairs:
       print(f"Question: {qa['question']}")
       print(f"Answer: {qa['answer']}")
       print(f"Level: {qa['qa_level']}")
       print(f"Task: {qa['qa_task']}")
   ```

2. **Evaluation**:
   ```bash
   python gpt_evaluation.py --model_name your_model --results_file your_results.json
   ```

### 📋 Data Format

Each JSON file contains QA pairs with the following structure:
```json
{
  "qa_idx": 1,
  "qa_type": "mcqa",
  "qa_level": "within_event",
  "qa_task": "speed",
  "question": "Are the individuals working at a fast pace between 1290.0 and 1350.0 seconds?",
  "answer": "No",
  "options": ["A. Yes", "B. No"]  // For MCQA only
}
```

## 🤝 Contributing

We welcome contributions to improve HLV-1K! Please feel free to:
- Report issues or bugs
- Suggest new features or improvements
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{zou2025hlv,
  title={Hlv-1k: A large-scale hour-long video benchmark for time-specific long video understanding},
  author={Zou, Heqing and Luo, Tianze and Xie, Guiyang and Zhang, Victor Xiao Jie and Lv, Fengmao and Wang, Guangcong and Chen, Junyang and Wang, Zhuochen and Zhang, Hansheng and Zhang, Huaijian},
  journal={arXiv preprint arXiv:2501.01645},
  year={2025}
}
```

## 🙏 Acknowledgments

We thank all contributors and the research community for their valuable feedback and support in developing HLV-1K.
