from datasets import load_dataset
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import lightning as L
from torch.optim import AdamW
import argparse


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET = "remyxai/SpaceThinker"
EPOCHS = 3
BS = 32
SAVE_DIR = f"./finetuned_models/full_params-{DATASET.split('/')[-1]}-{BASE_MODEL.split('/')[-1]}"


class SpaceThinkerDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset(DATASET)[split]

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


class QwenTrainer(L.LightningModule):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def training_step(self, batch, batch_idx):
        images, info = batch
        questions = info["questions"]
        answers = info["answers"]
        messages_list = []
        for q, a in zip(questions, answers):
            messages_list.append([
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ])
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
                epoch_dir = f"{SAVE_DIR}/epoch-{epoch_idx}"
                # Save model and processor in Hugging Face format
                self.model.save_pretrained(epoch_dir, safe_serialization=True)
        except Exception as e:
            # Log but don't crash training on save errors
            self.print(f"[warn] Failed to save epoch checkpoint: {e}")


def collate_fn(batch):
    images = [b[0] for b in batch]
    questions = [b[1]["questions"] for b in batch]
    answers = [b[1]["answers"] for b in batch]
    return images, {"questions": questions, "answers": answers}


def run_inference(image, question, model, processor):
    # image = Image.open(img_path).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

def run_inference_on_test_set(model, processor, num_of_batches=1):
    # Run inference for all images in the load_dataset(DATASET)["test"]
    test_dataset = SpaceThinkerDataset("test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=False, collate_fn=collate_fn)
    for i, batch in enumerate(test_loader):
        if i > num_of_batches:
            break
        images, info = batch
        questions = info["questions"]
        answers = info["answers"]
        for q, a in zip(questions, answers):
            output = run_inference(images[0], q, model, processor)
            print("#" * 100)
            print(f"Question: \n{q}")
            print(f"Predicted Answer: \n{output}")
            print(f"Ground Truth: \n{a}")
            print("#" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    # args for training
    parser.add_argument("--train", action="store_true")
    # args for inference
    parser.add_argument("--model", "-m", type=str, default=f"{SAVE_DIR}/epoch-{EPOCHS}")
    parser.add_argument("--processor", "-p", type=str, default=f"{SAVE_DIR}/processor")
    parser.add_argument("--image", "-i", type=str)
    parser.add_argument("--question", "-q", type=str)
    args = parser.parse_args()
    # example: python main_full_params.py --model=./finetuned_models/full_params-SpaceThinker-Qwen2.5-VL-3B-Instruct/epoch-1 --processor=./finetuned_models/full_params-SpaceThinker-Qwen2.5-VL-3B-Instruct/processor --image=./test_img1.jpg --question="I'm blind and holding the camera in my hand. How to reach the cup on the table? Please give a consise and quantitative answer."
    
    train = args.train
    model_path = args.model
    processor_path = args.processor
    image_path = args.image
    question = args.question

    if train:
        # Load the base model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        # Set full-parameter finetuning
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        train_dataset = SpaceThinkerDataset("train")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True, collate_fn=collate_fn)
        trainer = L.Trainer(max_epochs=EPOCHS, devices=1, accelerator="auto", precision="bf16-mixed", accumulate_grad_batches=2)
        trainer.fit(QwenTrainer(model, processor), train_loader)
        processor.save_pretrained(f"{SAVE_DIR}/processor")

    else:
        # Load the fine-tuned model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        image = Image.open(image_path).convert("RGB")
        output = run_inference(image, question, model, processor)
        print("#" * 100)
        print(f"Question: \n{question}")
        print("-" * 100)
        print(f"Answer: \n{output}")
        print("#" * 100)
