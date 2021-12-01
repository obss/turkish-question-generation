<div align="center">
<h1>
  Turkish Question Generation
</h1>

<h4>
  Offical source code for 

  <a href="https://arxiv.org/abs/2111.06476">"Automated question generation &amp; question answering from Turkish texts using text-to-text transformers"</a>
</h4>
</div>

<details open>
<summary>
  <big><b>citation</b></big>
</summary>
  
If you use this software in your work, please cite as:
  
```
@article{akyon2021automated,
  title={Automated question generation and question answering from Turkish texts using text-to-text transformers},
  author={Akyon, Fatih Cagatay and Cavusoglu, Devrim and Cengiz, Cemil and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={arXiv preprint arXiv:2111.06476},
  year={2021}
}
```
  
</details>

<details closed>
<summary>
<big><b>install</b></big>
</summary>

```bash
git clone https://github.com/obss/turkish-question-generation.git
cd turkish-question-generation
pip install -r requirements.txt
```
</details>

<details closed>
<summary>
<big><b>train</b></big>
</summary>

- start a training using args:

```bash
python run.py --model_name_or_path google/mt5-small  --output_dir runs/exp1 --do_train --do_eval --tokenizer_name_or_path mt5_qg_tokenizer --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 1e-4 --seed 42 --save_total_limit 1
```

- download [json config](configs/default/config.json) file and start a training:

```bash
python run.py config.json
```

- downlaod [yaml config](configs/default/config.yaml) file and start a training:

```bash
python run.py config.yaml
```

</details>

<details closed>
<summary>
<big><b>evaluate</b></big>
</summary>

- arrange related params in config:

```yaml
do_train: false
do_eval: true
eval_dataset_list: ["tquad2-valid", "xquad.tr"]
prepare_data: true
mt5_task_list: ["qa", "qg", "ans_ext"]
mt5_qg_format: "both"
no_cuda: false
```

- start an evaluation:

```bash
python run.py config.yaml
```

</details>

<details closed>
<summary>
<big><b>neptune</b></big>
</summary>

- install neptune:

```bash
pip install neptune-client
```

- download [config](configs/default/config.yaml) file and arrange neptune params:

```yaml
run_name: 'exp1'
neptune_project: 'name/project'
neptune_api_token: 'YOUR_API_TOKEN'
```

- start a training:

```bash
python train.py config.yaml
```

</details>

<details closed>
<summary>
<big><b>wandb</b></big>
</summary>

- install wandb:

```bash
pip install wandb
```

- download [config](configs/default/config.yaml) file and arrange wandb params:

```yaml
run_name: 'exp1'
wandb_project: 'turque'
```

- start a training:

```bash
python train.py config.yaml
```

</details>

<details closed>
<summary>
<big><b>finetuned checkpoints</b></big>
</summary>

[model_url1]: https://drive.google.com/uc?id=10hHFuavHCofDczGSzsH1xPHgTgAocOl1
[model_url2]: https://huggingface.co/google/mt5-small
[model_url3]: https://huggingface.co/google/mt5-base
[model_url4]: https://drive.google.com/uc?id=17MTMDhhEtQ9AP-y3mQl0QV0T8SvT_OZF
[model_url5]: https://drive.google.com/uc?id=1LOaZvQFwVGk9WFXU1bB8MsgjEsmN__Ex
[model_url6]: https://drive.google.com/uc?id=10M2sukxLiat2M4j_nIJycn4X5gtij-K9
[model_url7]: https://drive.google.com/uc?id=1WMQ_Iis4eYvD52wt2zXaFSzpU8umFNd2
[model_url8]: https://drive.google.com/uc?id=10M2sukxLiat2M4j_nIJycn4X5gtij-K9
[data_url1]: https://github.com/obss/turkish-question-generation/releases/download/0.0.1/tquad_train_data_v2.json
[data_url2]: https://github.com/obss/turkish-question-generation/releases/download/0.0.1/tquad_dev_data_v2.json
[data_url3]: https://github.com/deepmind/xquad/blob/master/xquad.tr.json


|name |model |training <br> data |trained <br> tasks |model size<br><sup>(GB) |
|--- |--- |--- |--- |--- |
|[mt5-small-3task-both-tquad2][model_url4] |[mt5-small][model_url2] |[tquad2-train][data_url1] |QA,QG,AnsExt  |1.2GB |
|[mt5-small-3task-prepend-tquad2][model_url6] |[mt5-small][model_url2] |[tquad2-train][data_url1] |QA,QG,AnsExt  |1.2GB |
|[mt5-small-3task-both-combined3][model_url7] |[mt5-small][model_url2] |[tquad2-train][data_url1]+[tquad2-valid][data_url2]+[xquad.tr][data_url3]|QA,QG,AnsExt  |1.2GB |
|[mt5-base-3task-both-tquad2][model_url5] |[mt5-base][model_url3] |[tquad2-train][data_url1] |QA,QG,AnsExt  |2.3GB |
|[mt5-base-3task-both-combined3][model_url8] |[mt5-base][model_url3] |[tquad2-train][data_url1]+[tquad2-valid][data_url2]+[xquad.tr][data_url3]|QA,QG,AnsExt |2.3GB |

</details>

<details closed>
<summary>
<big><b>format</b></big>
</summary>

- answer extraction:

input:
```
"<hl> Osman Bey 1258 yılında Söğüt’te doğdu. <hl> Osman Bey 1 Ağustos 1326’da Bursa’da hayatını kaybetmiştir.1281 yılında Osman Bey 23 yaşında iken Ahi teşkilatından olan Şeyh Edebali’nin kızı Malhun Hatun ile evlendi."
```

target:
```
<sep> 1258 <sep> Söğüt’te <sep>
```

- question answering:

input:
```
"question: Osman Bey nerede doğmuştur? context: Osman Bey 1258 yılında Söğüt’te doğdu. Osman Bey 1 Ağustos 1326’da Bursa’da hayatını kaybetmiştir.1281 yılında Osman Bey 23 yaşında iken Ahi teşkilatından olan Şeyh Edebali’nin kızı Malhun Hatun ile evlendi."
```

target:
```
"Söğüt’te"
```

- question generation (prepend):

input:
```
"answer: Söğüt’te context: Osman Bey 1258 yılında Söğüt’te doğdu. Osman Bey 1 Ağustos 1326’da Bursa’da hayatını kaybetmiştir.1281 yılında Osman Bey 23 yaşında iken Ahi teşkilatından olan Şeyh Edebali’nin kızı Malhun Hatun ile evlendi."
```

target:
```
"Osman Bey nerede doğmuştur?"
```

- question generation (highlight):

input:
```
"generate question: Osman Bey 1258 yılında <hl> Söğüt’te <hl> doğdu. Osman Bey 1 Ağustos 1326’da Bursa’da hayatını kaybetmiştir.1281 yılında Osman Bey 23 yaşında iken Ahi teşkilatından olan Şeyh Edebali’nin kızı Malhun Hatun ile evlendi."
```

target:
```
"Osman Bey nerede doğmuştur?"
```

- question generation (both):

input:
```
"answer: Söğüt’te context: Osman Bey 1258 yılında <hl> Söğüt’te <hl> doğdu. Osman Bey 1 Ağustos 1326’da Bursa’da hayatını kaybetmiştir.1281 yılında Osman Bey 23 yaşında iken Ahi teşkilatından olan Şeyh Edebali’nin kızı Malhun Hatun ile evlendi."
```

target:
```
"Osman Bey nerede doğmuştur?"
```
</details>

<details closed>
<summary>
<big><b>paper results</b></big>
</summary>

<div align="center">
 BERTurk-base and mT5-base QA evaluation results for TQuADv2 fine-tuning.
</p>
<img src="https://user-images.githubusercontent.com/34196005/141182075-5d3d1c43-fdf3-4892-921b-6e2d5c7550f9.png" width="500">
</div>

<div align="center">
mT5-base QG evaluation results for single-task (ST) and multi-task (MT) for TQuADv2 fine-tuning.
</p>
<img src="https://user-images.githubusercontent.com/34196005/141182079-40cd436c-2a93-4ad6-a59a-1db76e96ccfe.png" width="500">
</div>

<div align="center">
TQuADv1 and TQuADv2 fine-tuning QG evaluation results for multi-task mT5 variants. MT-Both means, mT5 model is fine-tuned with ’Both’ input format and in a multi-task setting.
</p>
<img src="https://user-images.githubusercontent.com/34196005/141182081-97dc188b-475a-4dfd-91f6-f08c14a472fc.png" width="700">
</div>

</details>

<details closed>
<summary>
<big><b>paper configs</b></big>
</summary>

You can find the config files used in the paper under [configs/paper](configs/paper).

</details>

<details closed>
<summary>
<big><b>contributing</b></big>
</summary>

Before opening a PR:

- Install required development packages:

```bash
pip install "black==21.7b0" "flake8==3.9.2" "isort==5.9.2"
```

- Reformat with black and isort:

```bash
black . --config pyproject.toml
isort .
```

</details>
