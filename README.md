**catalogue**

```
├── .cache
│   ├── dataset_cache_dev-clean.pkl
│   ├── dataset_cache_test-clean.pkl
│   ├── dataset_cache_train-clean-100-train-clean-360.pkl
│   ├── huggingface
│   │   └── hub
│   │       └── version.txt
│   ├── matplotlib
│   │   └── fontlist-v330.json
│   ├── Microsoft
│   │   └── DeveloperTools
│   │       └── deviceid
│   ├── models
│   │   └── wav2vec2-large-960h
│   │       ├── config.json
│   │       ├── preprocessor_config.json
│   │       ├── pytorch_model.bin
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.json
│   ├── version.txt
│   ├── wav2vec2_anti_explosion_20250724_041341
│   │   ├── best_finetuned_model
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── preprocessor_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_10.pth
│   │   ├── checkpoint_epoch_15.pth
│   │   ├── checkpoint_epoch_5.pth
│   │   ├── final_test_results.json
│   │   ├── initial_config.json
│   │   ├── predictions_test-clean.json
│   │   ├── training_config.json
│   │   ├── training_curves.png
│   │   └── training_history.json
│   ├── wav2vec2_finetuned_20250721_015618
│   │   ├── best_finetuned_model
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── preprocessor_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── checkpoint-epoch-5
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── training_state.pt
│   │   ├── finetune_curves.png
│   │   ├── finetune_history.json
│   │   ├── finetune_hyperparameters.json
│   │   ├── test_predictions.txt
│   │   └── test_results.json
│   ├── wav2vec2_librispeech_ctc_fixed_20250717_071203
│   │   ├── best_model
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── preprocessor_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── hyperparameters.json
│   │   ├── test_predictions.txt
│   │   ├── test_results.json
│   │   ├── training_curves.png
│   │   └── training_history.json
│   ├── wav2vec2_librispeech_fixed_20250717_124529
│   │   ├── best_model
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── preprocessor_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── hyperparameters.json
│   │   ├── test_predictions.txt
│   │   ├── test_results.json
│   │   ├── training_curves.png
│   │   └── training_history.json
│   ├── wav2vec2_training_20250723_032500
│   │   ├── best_finetuned_model
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── preprocessor_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_15.pth
│   │   ├── checkpoint_epoch_30.pth
│   │   ├── initial_config.json
│   │   ├── training_config.json
│   │   ├── training_curves.png
│   │   └── training_history.json
│   └── xet
│       └── https___cas_serv-tGqkUaZf_CBPHQ6h
│           ├── chunk-cache
│           └── staging
├── finetuning_ASR
│   ├── download_model.py
│   ├── fixed_model_file.py
│   ├── fixed_Wav2vec2_finetuning.py
│   └── Wav2vec_finetuning.py
├── huggingface
│   └── datasets
│       ├── dev-clean.tar.gz
│       ├── LibriSpeech
│       │   ├── BOOKS.TXT
│       │   ├── CHAPTERS.TXT
│       │   ├── dev-clean
│       │   ├── LICENSE.TXT
│       │   ├── README.TXT
│       │   ├── SPEAKERS.TXT
│       │   ├── test-clean
│       │   ├── train-clean-100
│       │   └── train-clean-360
│       ├── test-clean.tar.gz
│       ├── train-clean-100.tar.gz
│       └── train-clean-360.tar.gz
├── project_structure.txt
├── requirements.txt
└── Ultimate_version_for_finetuning
    ├── download.py
    ├── download_t_model.py
    ├──  enhanced_wav2vec2_training.py
    ├── LibriSpeech_validator.py
    ├── RTX5090_based_on_best_model.py
    ├── RTX5090_finetuning_on_train_clean_360.py
    ├── RTX5090_finetuning.py
    └── test.py

33 directories, 101 files
```




Datasets please check [OpenSLR](https://www.openslr.org/12/), including train-clean-360,train-clean-100, test-clean, dev-clean


**path**          mv datasets to ```/root/fssd/ASR_task/huggingface/datasets/LibriSpeech```


**models path**     my codes will generate paths automatically


**python file**     ```/root/fssd/ASR_task/Ultimate_version_for_finetuning/RTX5090_finetuning_on_train_clean_360.py```   


**Implementation**

```
you can finetuned baseline  on  train-clean-100, and best_finetuned_model(it generates automatically) is the best model, which performed wer 0.0346 on VAL SET
```

```
you can finetuned best_finetuned_model on train-clean-100 + train-clean-360, and new best_finetuned_model(it generates automatically) is the best model, which performed wer 0.0317 on VAL SET and 0.0306 on TEST SET
```

**models** I have not uploaded my finetuned models and basic pretrained model wav2vec2-large-960h to this repo, so you can just run my code that can download and keep the models to the default path automatically


**performance** after a series of technical finetuning, it got WER 0.0306 ON TEST SET


```
1. 前向传播 → 计算损失
2. 检查损失是否异常 → 异常则跳过
3. 反向传播 → 计算梯度
4. 清理异常梯度值 (NaN, Inf)
5. 计算梯度范数
6. 检测是否梯度爆炸
   ├─ 是 → 跳过更新 + 降低学习率
   └─ 否 → 继续处理
7. 梯度裁剪 (固定或自适应)
8. 优化器更新
9. 清零梯度
```

**大了就裁剪，爆了就跳过，动态调整学习率**
```
检测 → 裁剪 → 跳过 → 衰减
 ↓      ↓      ↓      ↓
监控   限制   避免   调整
异常   幅度   更新   速率
```
