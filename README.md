# <img src="static/images/hlv-1k.png" alt="HLV-1K" style="width:40px;height:40px;"> HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding

[arXiv](https://arxiv.org/submit/6108440/view)    [Code](https://github.com/Vincent-ZHQ/HLV-1K)


## Introduction

Multimodal large language models have become a popular topic in deep visual understanding due to many promising real-world applications. However, hour-long video understanding, spanning over one hour and containing tens of thousands of visual frames, remains under-explored because of 1) challenging long-term video analyses, 2) inefficient large-model approaches, and 3) lack of large-scale benchmark datasets. Among them, in this paper, we focus on building a large-scale hour-long long video benchmark, HLV-1K, designed to evaluate long video understanding models. HLV-1K comprises 1009 hour-long videos with 14,847 high-quality question answering (QA) and multi-choice question asnwering (MCQA) pairs with time-aware query and diverse annotations, covering frame-level, within-event-level, cross-event-level, and long-term reasoning tasks. We evaluate our benchmark using existing state-of-the-art methods and demonstrate its value for testing deep long video understanding capabilities at different levels and for various tasks. This includes promoting future long video understanding tasks at a granular level, such as deep understanding of long live videos, meeting recordings, and movies.

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


## Benchmark

### Data Examples
<img src="static/images/HLV_1K_F01.jpg" alt="HLV-1K" style="width:900px;height:700px;"> 

Benchmark construction and examples.

### Benchmark Statistics
<img src="static/images/HLV_1K_F00.jpg" alt="HLV-1K" style="width:900px;height:380px;"> 

HLV-1K: (a) Video category distribution, (b) Video duration distribution, and (c) Duration distribution of time-specific query.


<img src="static/images/HLV_1K_F02.jpg" alt="HLV-1K" style="width:900px;height:310px;"> 

HLV-1K: Distribution of benchmark annotations.

### Benchmark Comparison

<img src="static/images/datasets.png" alt="HLV-1K" style="width:900px;height:160px;"> 

## Experiment Results

### Different Question Types

<img src="static/images/HLV_1K_F03.jpg" alt="HLV-1K" style="width:900px;height:580px;"> 

Evaluation results of four representative MLLMs.

## Related Wrok

[Awesome-LV-LLMs](https://github.com/Vincent-ZHQ/LV-LLMs)


### Citation

```

    @article{zou2025hlv1k,
      title={HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding},
      author={Heqing Zou, Tianze Luo, Guiyang Xie, Victor (Xiao Jie) Zhang, Fengmao Lv, Guangcong Wang, Junyang Chen, Zhuochen Wang, Hansheng Zhang and Huaijian Zhang },
      year={2024}
    }
```

This website is adapted from [Video-MME](https://video-mme.github.io/), licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
