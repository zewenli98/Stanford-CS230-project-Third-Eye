from datasets import load_dataset
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import lightning as L
from torch.optim import AdamW
from peft import PeftModel, LoraConfig, get_peft_model
import argparse

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

class SpaceThinkerDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("remyxai/SpaceThinker")[split]

    def __getitem__(self, idx):
        sample = self.ds[idx]
        imgs = sample.get("images", None)
        img = imgs[0] if isinstance(imgs, list) else imgs
        if isinstance(img, str):
            image = Image.open(img).convert("RGB")
        else:
            image = img.convert("RGB") if hasattr(img, "convert") else img

        questions = sample.get("input")
        answers = sample.get("output")
        return image, {"questions": questions, "answers": answers}

    def __len__(self):
        return len(self.ds)
class OpenSpaces_MC_R1Dataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("remyxai/OpenSpaces_MC_R1")[split]

    def __getitem__(self, idx):
        sample = self.ds[idx]
        imgs = sample.get("images", None)
        img = imgs[0] if isinstance(imgs, list) else imgs
        if isinstance(img, str):
            image = Image.open(img).convert("RGB")
        else:
            image = img.convert("RGB") if hasattr(img, "convert") else img

        questions = sample.get("messages")
        answers = sample.get("reasoning")
        return image, {"questions": questions, "answers": answers}

    def __len__(self):
        return len(self.ds)

class QwenTrainer(L.LightningModule):
    def __init__(self, dataset, model, processor, save_dir):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.processor = processor
        self.save_dir = save_dir

    def training_step(self, batch, batch_idx):
        images, info = batch
        questions = info["questions"]
        answers = info.get("answers")

        messages_list = []
        # dataset flag uses 'spacethinker' or 'openspaces'
        if self.dataset == "spacethinker":
            for q, a in zip(questions, answers):
                messages_list.append([
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]},
                ])
        elif self.dataset == "openspaces":
            # OpenSpaces provides messages in the correct chat format
            messages_list = questions
        
        texts = [self.processor.apply_chat_template(m, add_generation_prompt=False) for m in messages_list]
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        outputs = self.model(**inputs)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.1)

    def on_train_epoch_end(self):
        # Save HF-format weights and processor after each epoch on rank 0 only
        try:
            if getattr(self.trainer, "is_global_zero", True):
                epoch_idx = int(self.current_epoch) + 1
                epoch_dir = f"{self.save_dir}/epoch-{epoch_idx}"
                # Save model in Hugging Face format
                self.model.save_pretrained(epoch_dir, safe_serialization=True)
        except Exception as e:
            # Log but don't crash training on save errors
            self.print(f"[warn] Failed to save epoch checkpoint: {e}")

def run_train(full_params, dataset, processor_path, save_dir, epochs, bs):
    # Load the base model for training
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    if full_params:
        # Set full-parameter finetuning
        base_model.train()
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
    else:
        # Set LoRA finetuning
        lora_config = LoraConfig(r=128, lora_alpha=256, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none")
        lora_model = get_peft_model(base_model, lora_config).cuda()

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if dataset == "spacethinker":
        train_dataset = SpaceThinkerDataset("train")
    else:
        train_dataset = OpenSpaces_MC_R1Dataset("train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    if full_params:
        trainer = L.Trainer(max_epochs=epochs, devices=1, accelerator="auto", precision="bf16-mixed", accumulate_grad_batches=2)
        trainer.fit(QwenTrainer(dataset, base_model, processor, save_dir), train_loader)
    else:
        trainer = L.Trainer(max_epochs=epochs, devices=1, accelerator="auto")
        trainer.fit(QwenTrainer(dataset, lora_model, processor, save_dir), train_loader)
    processor.save_pretrained(processor_path)

def collate_fn(batch):
    images = [b[0] for b in batch]
    questions = [b[1]["questions"] for b in batch]
    answers = [b[1]["answers"] for b in batch]
    return images, {"questions": questions, "answers": answers}

def run_inference(dataset, model, processor, image, question):
    # image = Image.open(img_path).convert("RGB")
    if dataset == "spacethinker":
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
    elif dataset == "openspaces":
        messages = question
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

def run_inference_on_test_set(dataset, model, processor, num_of_batches=1):
    # Run inference for all images in the load_dataset(DATASET)["test"]
    if (dataset == "spacethinker"):
        test_dataset = SpaceThinkerDataset("test")
    elif (dataset == "openspaces"):
        test_dataset = OpenSpaces_MC_R1Dataset("test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    for i, batch in enumerate(test_loader):
        if i > num_of_batches:
            break
        images, info = batch
        questions = info["questions"]
        answers = info["answers"]
        for idx, (q, a) in enumerate(zip(questions, answers)):
            output = run_inference(dataset, model, processor, images[idx], q)
            print("#" * 100)
            print(f"Question: \n{q}")
            print(f"Predicted Answer: \n{output}")
            print(f"Ground Truth: \n{a}")
            print("#" * 100)

def run_test(full_params, dataset, model_path, processor_path, image, question):
    if full_params:  # Load the fine-tuned model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    else:  # Load the LoRA model
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        model = PeftModel.from_pretrained(base_model, model_path).cuda().eval()

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    image = Image.open(image).convert("RGB")
    output = run_inference(dataset, model, processor, image, question)
    print("#" * 100)
    print(f"Question: \n{question}")
    print("-" * 100)
    print(f"Answer: \n{output}")
    print("#" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    # args for training
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dataset", type=str, default="spacethinker", choices=["spacethinker", "openspaces"])
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--batch", type=int, default=-1)
    # args for inference
    parser.add_argument("--model", "-m", type=str, help="Path to model epoch directory (overrides computed path)")
    parser.add_argument("--processor", "-p", type=str, help="Path to processor directory (overrides computed path)")
    parser.add_argument("--image", "-i", type=str)
    parser.add_argument("--question", "-q", type=str)

    args = parser.parse_args()
    # examples:
    # python main.py --train --dataset=spacethinker --lora --epoch=5 --batch=32
    # python main.py --image=./test_img1.jpg --question="I'm blind and holding the camera in my hand. How to reach the cup on the table? Please give a consise and quantitative answer."

    train = args.train
    dataset = args.dataset
    full_params = args.lora == False
    epochs = args.epoch
    bs = args.batch
    image = args.image
    question = args.question

    if full_params:
        if dataset == "spacethinker":
            save_dir = f"./finetuned_models/full_params-SpaceThinker-{BASE_MODEL.split('/')[-1]}"
        else:
            save_dir = f"./finetuned_models/full_params-OpenSpaces_MC_R1-{BASE_MODEL.split('/')[-1]}"
        if epochs == -1:
            epochs = 3
        if bs == -1:
            bs = 32
    else:
        if dataset == "spacethinker":
            save_dir = f"./finetuned_models/LoRA-SpaceThinker-{BASE_MODEL.split('/')[-1]}"
        else:
            save_dir = f"./finetuned_models/LoRA-OpenSpaces_MC_R1-{BASE_MODEL.split('/')[-1]}"
        if epochs == -1:
            epochs = 5
        if bs == -1:
            bs = 32

    model_path = f"{save_dir}/epoch-{epochs}"
    processor_path = f"{save_dir}/processor"
    # Allows override of model/processor paths
    if getattr(args, "model", None):
        model_path = args.model
    if getattr(args, "processor", None):
        processor_path = args.processor

    if train:
        run_train(full_params, dataset, processor_path, save_dir, epochs, bs)

    else:
        run_test(full_params, dataset, model_path, processor_path, image, question)
