# Third-Eye

## Dataset
In this repo, we currently support two datasets `remyxai/OpenSpaces_MC_R1` and `remyxai/SpaceThinker`.

`remyxai/OpenSpaces_MC_R1` is a multimodal dataset focused on visual question answering (VQA) and spatial reasoning using images, while `remyxai/SpaceThinker` is a larger, reasoning-optimized dataset synthesizing spatial relationships and distances in scenes. Both leverage VQASynth for data generation but differ in sample count, format, and reasoning depth.

## Training
There are two types of files named `main_full_params_XXX.py` and `main_lora_XXX.py`. They are currently targeting the model `Qwen2.5-VL-3B-Instruct` and `Qwen2.5-VL-7B-Instruct` and could be easily customized to other Qwen variants as well.

`main_full_params_XXX.py` is to finetune/infer the whole model (full parameters), while `main_lora_XXX.py` is only to finetune the LoRA, and then insert it to the base model. 

To keep concise, please directly modify hyper-params in the code. The whole model params will be saved for each epoch. The processor will be saved once and can be shared with all models in inference. e.g.:
```
# finetune full params on SpaceThinker dataset
python main_full_params_spacethinker.py --train

# finetune LoRA on OpenSpaces_MC_R1 dataset
python main_lora_openspaces.py --train
```

## Inference
```
# Inference on full-params finetuned model
python main_full_params_spacethinker.py --model=./finetuned_models/full_params-SpaceThinker-Qwen2.5-VL-3B-Instruct/epoch-1 --processor=./finetuned_models/full_params-SpaceThinker-Qwen2.5-VL-3B-Instruct/processor --image=./test_img1.jpg --question="I'm blind and holding the camera in my hand. How to reach the cup on the table? Please give a consise and quantitative answer."

# LoRA
python main_lora_openspaces.py --model=./finetuned_models/LoRA-SpaceThinker-Qwen2.5-VL-3B-Instruct/epoch-1 --processor=./finetuned_models/LoRA-SpaceThinker-Qwen2.5-VL-3B-Instruct/processor --image=./test_img1.jpg --question="I'm blind and holding the camera in my hand. How to reach the cup on the table? Please give a consise and quantitative answer."
```

## Evaluation
In `eval.py`, feel free to add any models and processors to `model_processor_name_dict` for evaluation. All model results will be printed to the console and saved as csv files under `./results/`.
