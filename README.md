# Third-Eye

There are two main files named `main_full_params.py` and `main_lora.py`. They are currently targeting the model `Qwen2.5-VL-3B-Instruct` and dataset `remyxai/SpaceThinker`. It could be easily customized to other Qwen2.5 variants and datasets.

`main_full_params.py` is to finetune/infer the whole model, while `main_lora.py` is to train a LoRA.and then attach to the base model. 

## Training
To keep concise, please directly modify hyper-params in the code. The whole model params will be saved for each epoch. The processor will be saved once and can be shared with all models in inference.
```
# finetune full params
python main_full_params.py --train

# LoRA
python main_lora.py --train
```

## Inference
```
# full-params finetuned model
python main_full_params.py --model=./finetuned_models/full_params-SpaceThinker-Qwen2.5-VL-3B-Instruct/epoch-1 --processor=./finetuned_models/full_params-SpaceThinker-Qwen2.5-VL-3B-Instruct/processor --image=./test_img1.jpg --question="I'm blind and holding the camera in my hand. How to reach the cup on the table? Please give a consise and quantitative answer."

# LoRA
python main_lora.py --model=./finetuned_models/LoRA-SpaceThinker-Qwen2.5-VL-3B-Instruct/epoch-1 --processor=./finetuned_models/LoRA-SpaceThinker-Qwen2.5-VL-3B-Instruct/processor --image=./test_img1.jpg --question="I'm blind and holding the camera in my hand. How to reach the cup on the table? Please give a consise and quantitative answer."
```
