Datasets please check [OpenSLR](https://www.openslr.org/12/), including train-clean-360,train-clean-100, test-clean, dev-clean

**path**          mv datasets to ```/root/fssd/ASR_task/huggingface/datasets/LibriSpeech```

**models path**  
 
            paradigm 1: ```/root/fssd/ASR_task/.cache/wav2vec2_librispeech_ctc_fixed_20250717_071203/best_model```

            paradigm 2: ```/root/fssd/ASR_task/.cache/wav2vec2_finetuned_20250721_015618/best_finetuned_model```

            paradigm 3: ```/root/fssd/ASR_task/.cache/wav2vec2_training_20250723_032500/best_finetuned_model```


**python file**     ```/root/fssd/ASR_task/Ultimate_version_for_finetuning/RTX5090_finetuning_on_train_clean_360.py```   

**Implementation**

```
you can finetuned baseline  on  train-clean-100, and best_finetuned_model(it generates automatically) is the best model, which performed wer 0.0346 on VAL SET
```

```
you can finetuned best_finetuned_model on train-clean-100 + train-clean-360, and new best_finetuned_model(it generates automatically) is the best model, which performed wer 0.0317 on VAL SET and 0.0317 on TEST SET
```

**models** I have not uploaded my finetuned models and basic pretrained model wav2vec2-large-960h to this repo, so you can just run my code that can download and keep the models to the default path automatically

**performance** after a series of technical finetuning, it got WER 0.0346 ON DEV SET

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
