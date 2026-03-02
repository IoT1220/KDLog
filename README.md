# HermesLog

HermesLog is a novel federated split learning (FSL) framework. It enables collaborative training while ensuring data privacy and reducing the computational load on the client. It not only significantly lowers local computational load and privacy leakage but also preserves high diagnostic accuracy.

## 🔍 Key Features

- **FL + SL Integration**: Collaborative training while ensuring data privacy and reducing the computational load on the client.
- **Multilayer privacy protection mechanism**: Combining feature filtering and irreversible protection techniques to effectively reduce reconstruction success and safeguard data privacy.
- **A model-splitting (SL-BERT）**: With the server sharing the backbone of the resource-intensive transformer and clients maintaining lightweight embeddings and personalized heads to reduce local computing power load.
- **The personalized local model via FedAvg and EMA (FL-EMA)**: Integrate component-driven log clustering with personalized fine-tuning to enhances model’s generalization and adaptability in heterogeneous environments.
- **Modular & Extensible**: Containerized deployment demonstrates enables rapid fault localization and recovery, effectively protecting enterprise data.

## 📄 Dataset Description
### This study evaluates two datasets:

  - **The publicly available Aliyun dataset:** link at https://tianchi.aliyun.com/competition/entrance/531947/information.  

  - **The Privacy unavailable ZTE datasets:** a proprietary dataset licensed from an industry partner, which cannot be publicly released. Access to the proprietary dataset requires authorization from the provider and a signed data‑use agreement.  

### Data storage and load:
  **dataset is divided into five parts, each representing the log data of an client-server, as shown in the following three files:**

- **1.The result of the log sequence after being vectorized by BERT**
```bash
data_{}.npy
```
- **2.Semi-supervised labels, where -1 indicates no label**
```bash
 semi_label_{}.npy
```
- **3.The label of the original data source**
```bash
 label_{}.npy
```

## 📁 Icore code 

1. **Prompt-tuning**   

```bash
claude_zeroshot-cot.py
mistral_fewshot-cot.py
```

2. **Preference-tuning**
```bash
config.py
vllm_sample_offline.py
make_preference.py
run_train.py
```

3. **Knowledge Distillation**
```bash
XXX.py
```

## 📦 Installation

```
conda create --name <env> --file requirements.txt
```




## 📁 Project Structure
```
KDLog/
├── code/               # Icore code (SL-Bert, FL-EMA, docker)
├── data/               # Input logs
├── requirements/       # Create an environment
└── README.md           # Project description
```

```  
 Prompt-tuning/
├── claude_zeroshot-cot-stage1.py
├── data
│   └── output.json
├── mistral_fewshot-cot-stage2.py
├── monitor_gpu.sh
├── output-stage1
│   ├── claude.log
│   └── claude_results.json
├── output-stage2-fewshot-cot
│   ├── mistral.log
│   └── mistral_results.json
├── README.md
├── test_case_id.txt
└── tree.txt
```

```
(base) ➜  dataset ls
claude_zeroshot-cot.py            output-stage2-fewshot-cot
data                              output-stage2-fewshot-nocot
mistral_fewshot-cot-stage2.py     output-stage2-zeroshot-cot
mistral_fewshot-nocot-stage2.py   output-stage2-zeroshot-nocot
mistral_zeroshot-cot-stage2.py    README.md
mistral_zeroshot-nocot-stage2.py  test_case_id.txt
monitor_gpu.sh                    tree.txt
output-stage1
```

```  
Preference-tuning/
├── config.py
├── vllm_sample_offline.py
├── make_preference.py
├── run_train.py
└── vllm_sample_offline.py
```

```
Knowledge Distillation/
``` 

## 🔗 Links
- [Code](https://github.com/IoT1220/KDLog)








## 重要文件


1. 代码

   ```
   cal_acc.py  make_preference.py  openai_multi_client.py  prompts.py         run_train.py  vllm_sample_offline.py
   config.py   make_sft.py         openai_rank.py          run_lora_merge.py  test.py       vllm_sample.py
   ```

2. 中间运行所需配置文件

   ```
   dpo.config.yaml  ipo.config.yaml  lora.merge.yaml  orpo.config.yaml  sft.config.yaml  temp.lora.merge.yaml  temp.yaml
   ```

   以及`data/dataset_info.json`文件

3. 批量运行跑实验的脚本在`scripts`文件夹下

   ```
   sample_aliyun.sh  sample_zte.sh  test_aliyun.sh  test_zte.sh  train_aliyun_sft.sh  train_aliyun.sh  train_zte.sh
   ```

## 运行说明

1. 设置dataset对应的文件夹路径，请编辑`config.py`文件，比如zte对应`ratio1224/ZTE/uncorrected`，请确保该文件夹下有类似于`train1+test1`的文件夹，在`ratio1224/ZTE/uncorrected/train1+test1`文件夹还应该有`train.json`和`test.json`文件

2. 采样模型回复，`vllm_sample_offline.py`

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python vllm_sample_offline.py --model xxxx_path_to_model_folder --dataset zte --fewshot no --sample_n 5 --split train --run_split 1
   ```

   这里面可以指定是否使用in-context learning，`--fewshot no`就是不使用，如果`yes`的话，还可以指定使用哪些fewshot examples：`--fewshot_path xxx_path_to_file`；`--split train`参数指定了使用`train.json`，`--run_split 1`指的是使用`train1+test1`，相似地，如果是`--run_split 2`，那么就是`train2+test2`

3. 模型回复采样结束后，我们需要制作preference data，`make_preference.py`

   - 首先需要把刚才采样得到的文件`xxx.json`放到一个单独的文件夹中，比如`preference_data_folder`

   - 运行命令会得到在同文件夹下的一个`preference_data.train.json`

     ```
     python make_preference.py --dataset zte --input_folder xxx_path_to_preference_data_folder
     ```

   - 这份数据将用来偏好训练模型，我们需要编辑`data/dataset_info.json`，给该数据集命名增加一条：

     ```
         "test_dataset_name": {
             "file_name": "xxxxx_path_to_file_preference_data.train.json",
             "ranking": true,
             "columns": {
                 "prompt": "instruction",
                 "query": "input",
                 "chosen": "chosen",
                 "rejected": "rejected"
             }
         }
     ```

4. 模型偏好训练，`run_train.py`

   ```
   python run_train.py \
           --train_method ipo \
           --dataset test_dataset_name \
           --output output_folder/xxxxxxxx \
           --mode "dpo"
   ```

   其中，`--train_method ipo`可以替换为`dpo`或者`orpo`，注意修改`--output`到你想要的模型保存的位置；训练结束后在output文件夹中你会找到一个`xxx-merged`文件夹，这个就是最终保存的模型ckpt结果

5. 对训练好的模型进行测试，`vllm_sample_offline.py`

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python vllm_sample_offline.py --model xxxx_path_to_merged_model --dataset zte --fewshot yes --sample_n 1 --split test --run_split 1 --fewshot_path xxxx_fewshot.json
   ```

   注意修改`--model`参数为刚才得到的`xxx-merged`文件夹路径，如果使用fewshot，你也可以指定`--fewshot_path xxxx_fewshot.json`参数


   # Knowledge Distillation







