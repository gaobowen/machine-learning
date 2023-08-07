```py
# sudo apt install nvidia-cuda-toolkit

# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# pip install bitsandbytes transformers peft accelerate trl scipy sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

```py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, BitsAndBytesConfig

model_path = '/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/05d7cc02d0d1cfd518dc98a9a16be2708e4a9043'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, device_map="auto")

print(tokenizer) #打印 special_tokens 发现没有 pad_token

# 终止符作为填充符号
tokenizer.pad_token = tokenizer.eos_token

#量化加载，减小显存占用，4bit 默认 float16 dtype, 量化模式LoRA需要最新版本支持
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
```
LlamaTokenizer(name_or_path='/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/05d7cc02d0d1cfd518dc98a9a16be2708e4a9043', vocab_size=32000, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("\<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("\</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False)}, clean_up_tokenization_spaces=False)

```py
PROMPT_DICT = {
    "prompt_input":
        """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        {instruction} 
        {input}
        <</SYS>> [/INST]
        """,
    "prompt_no_input": 
        """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        {instruction} 
        <</SYS>> [/INST]
        """
}

prompt_template = PROMPT_DICT["prompt_no_input"].format(instruction='你叫什么名字')

# 手动处理编解码
# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=64)
# print('output tensor', output.shape)
# outstring = tokenizer.decode(output[0], skip_special_tokens=True)
# print('outstring', outstring)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
```
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        你叫什么名字 
        <</SYS>> [/INST]
        我叫小明。 (wǒ jiào xiǎo míng)



```py
import os
import io
import json

def jload(path):
    f = open(path, mode='r')
    jdict = json.load(f)
    f.close()
    return jdict


def get_dataset_from_jsonl(jsonl_file, tokenizer=None):
    list_data_dict = jload(jsonl_file)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    #问题
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    #回答，回答加上起始终止符
    targets = [f"{tokenizer.bos_token}{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

    return zip(sources, targets)


class SFTDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=2048, max_data_size=0, valid_ratio=0.01):
        self.post_list = []
        dataset = get_dataset_from_jsonl(train_path, tokenizer=tokenizer)
        self.post_list = [s + t for s, t in dataset]
        
        if max_data_size != 0:
            self.post_list = self.post_list[:max_data_size]

        if "valid" in split:
            self.post_list = self.post_list[0:10]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }



max_input_length = 512 #llama2最高支持4k上下文，因为本机显存太小就取了512
data_path = "./traindata/wukong.txt"

train_dataset = SFTDataset(
        data_path,
        tokenizer,
        "train",
        max_length=max_input_length,
)
print('train_dataset', len(train_dataset))
```
train_dataset 30

```py
from peft import prepare_model_for_kbit_training, prepare_model_for_int8_training

# 配置lora参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out = False,
    # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'], #'down_proj', 'gate_proj', 'up_proj'
)

# model.resize_token_embeddings(len(tokenizer)) 词表变化 len(tokenizer) != model.get_input_embeddings().weight.shape[0]
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
```py
import transformers
trainer = Trainer(
    model=model, 
    train_dataset=train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        warmup_steps=50, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=10, 
        output_dir='mymodel'
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train(resume_from_checkpoint=False)

```
```
Step	Training Loss
10	16.302500
20	11.794800
30	2.497100
40	0.559500
50	0.294600
60	0.088300
70	0.050700
80	0.041800
90	0.033600
100	0.015100
110	0.004500
120	0.003000
130	0.002400
140	0.002500
150	0.002500
160	0.002400
170	0.002400
180	0.002300
190	0.002200
200	0.002300
```
```py
input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=64)
print('output tensor', output.shape)
outstring = tokenizer.decode(output[0], skip_special_tokens=True)
print('outstring', outstring)

```
```
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        你叫什么名字 
        <</SYS>> [/INST]
        我是孙悟空
```
```py
# peftModel重写了原始model的save_pretrained函数，只把lora层的权重进行存储，因此model.save_pretrained只会存储lora权重。
model.save_pretrained("outputs/wukong/")
```
```py
# 重启内核，重新加载模型，测试lora 效果

# if 'model' in globals():
#     import gc
#     del model
#     gc.collect()
# if 'base_model' in globals():
#     del base_model
#     import gc
#     gc.collect()
# torch.cuda.empty_cache()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


base_path = '/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/05d7cc02d0d1cfd518dc98a9a16be2708e4a9043'
lora_path = 'outputs/wukong'

base_model = AutoModelForCausalLM.from_pretrained(base_path, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, device_map="auto")

lora_model = PeftModel.from_pretrained(base_model, lora_path).eval()
```
```py
PROMPT_DICT = {
    "prompt_input":
        """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        {instruction} 
        {input}
        <</SYS>> [/INST]
        """,
    "prompt_no_input": 
        """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        {instruction} 
        <</SYS>> [/INST]
        """
}
prompt_template = PROMPT_DICT["prompt_no_input"].format(instruction='你叫什么名字')

# PeftModelForCausalLM 不支持 pipeline text-generation
# 经测试8bit的lora推理成功，bit4应该是bug
input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
print('input_ids', input_ids.shape)
output = lora_model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print('output tensor', output.shape)
outstring = tokenizer.decode(output[0], skip_special_tokens=True)
print('outstring', outstring)
```
```
input_ids torch.Size([1, 66])
output tensor torch.Size([1, 77])
outstring 
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
        answer the question use Chinese.
        你叫什么名字 
        <</SYS>> [/INST]
        我是孙悟空
```

