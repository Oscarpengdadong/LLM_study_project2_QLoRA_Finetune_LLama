# LLM_study_project2_QLoRA_Finetune_LLama
QLoRA fine-tuning on llama3-8b-instruct model. Train on dialog summary dataset https://huggingface.co/datasets/neil-code/dialogsum-test 
Workflow:
Target:
finetune LLama model version: on dataset: specific on area: dialog summary

Dataset:
huggingface

Base model:
setup Libraries
Download LLama model
Base model Config setup(quantization setting step is here)
import pre-trained model
setup tokenizer\

Data preprocess:
1: formatting text and summary to "prompt+input:text + output:summary"for model
2: tokenization, transfet text to token_ids(with attention_mask) 3: use base model to generate 'raw' summary for feeling.

Finetune process:
1:trainning setup:
1.1: Set up QLora Moudle(actually LoraConfig, base parameters W parameters NF4 setting + quantization is in earlier base model Config) 1.2: Set up peft_trainer 1.3: .train()

2:check training process:
trainable model parameters
all model parameters
percentage of trainable model parameters

3: train model\

post-training:
1.upload Peft_model from based model + Lora tuend model

test its ability of generating dialog summary. Compare its results with base model's.
Important comments:
1.I don't have good local GPU, and that's main reason I put this project on colab. One biggest debug issue I constantly have is error of torch.device(None). 我在peft_model.train(cuda)时触发 torch.device(None) 的报错，debug去掉了peft_model.to('cuda') 以及原模型 device—map=‘cuda’ 改成‘auto’ 后没问题了。可能是某些子模块或者参数（比如lora参数）没有放到GPU上，引发这个 “device index None” 的错误。 然后改成device_map='auto'，Accelerate就会动为模型的不同子模块分配设备，避免设备不一致带来的问题。 也可以参考这个回复https://stackoverflow.com/questions/78008119/huggingface-transformer-train-function-throwing-device-received-an-invalid-com?utm_source=chatgpt.com

2.If you want to run this file, pls change hugging face access token with your own, and before this step you need to apply premission from Meta Llama 3 on hugging face. The access token shown in this file is changed by randomly generated one, and you will get login error with it.

3.On github you can see modified version of this project. That evaluate generated summaries in more details.

4.关于checkpointing 主要作用在训练的 前向／反向传播中的激活保存与梯度重计算；而 use_cache 主要作用在推理阶段的 Key/Value 缓存复用，这俩不是一个阶段的事情，开启Key/Value 缓存复用不会加速训练速度。另外我看到有个帖子讨论这件事，https://discuss.huggingface.co/t/why-is-use-cache-incompatible-with-gradient-checkpointing/18811，我的理解是在训练中，通常是一次处理完整序列，模型并不逐 token 生成、累积 past key values。此时 use_cache=True 会让模型保存／返回 past_key_values，但这些缓存可能会干扰后续梯度计算或被视为“状态”而非激活，从而与 checkpointing 的重新计算逻辑冲突。
