Datasets please check [OpenSLR](https://www.openslr.org/12/), including train-clean-360,train-clean-100, test-clean, dev-clean

**path** mv datasets to ```/root/fssd/ASR_task/huggingface/datasets/LibriSpeech```

**models path**  paradigm 1: ```/root/fssd/ASR_task/.cache/wav2vec2_librispeech_ctc_fixed_20250717_071203/best_model```

            paradigm 2: ```/root/fssd/ASR_task/.cache/wav2vec2_finetuned_20250721_015618/best_finetuned_model```

            paradigm 3: ```/root/fssd/ASR_task/.cache/wav2vec2_training_20250723_032500/best_finetuned_model```


**python file** ```/root/fssd/ASR_task/Ultimate_version_for_finetuning/RTX5090_finetuning_on_train_clean_360.py```    you can finetuned ```/root/fssd/ASR_task/.cache/wav2vec2_finetuned_20250721_015618/best_finetuned_model``` on train-clean-100, and ```/root/fssd/ASR_task/.cache/wav2vec2_training_20250723_032500/best_finetuned_model``` is the best model, which performed wer 0.0346 on VAL SET

**models** I have not uploaded my finetuned models and basic pretrained model wav2vec2-large-960h to this repo, so you can go to my huggingface repo ```wesfggfd/wav2vec2-large-960h-finetuning-best-model```

**performance** after a series of technical finetuning, it got WER 0.0346 ON DEV SET
