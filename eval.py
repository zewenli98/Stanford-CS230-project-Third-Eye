import os
import csv
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import io
import json
from tqdm import tqdm
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data classes for better type management
@dataclass
class Query:
    """Single query object containing image and prompt"""
    prompt_id: int
    prompt_text: str
    image_name: str
    image_path: str = None
    image: Image.Image = None
    
@dataclass
class Result:
    """Result object containing query and response"""
    prompt_id: int
    prompt_text: str
    image_name: str
    model_name: str
    response: str

class LocalLLMProcessor:
    """Handle batch processing for local Qwen models"""
    
    def __init__(self, device: str = None):
        """
        Initialize the local LLM processor
        
        Args:
            device: Device to run the model on ('cuda', 'cpu', or auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        
    def load_model(self, model_name: str):
        """
        Load a local Qwen model
        
        Args:
            model_name: Path or name of the model (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        """
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return
            
        logger.info(f"Loading model: {model_name}")
        
        # Clear previous model from memory
        if self.model is not None:
            del self.model
            del self.processor
            if self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # For Qwen-VL models (vision-language models)
            if "VL" in model_name or "vl" in model_name:
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.processor = AutoProcessor.from_pretrained(model_name)
                
        
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            self.current_model_name = model_name
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def call_llm_batch(self, model_name: str, query_list: List[Query]) -> List[str]:
        """
        Call local LLM model in batch with images and prompts
        
        Args:
            model_name: Name/path of the model
            query_list: List of Query objects containing images and prompts
            
        Returns:
            List of response texts from the LLM
        """
        # Load model if not already loaded
        self.load_model(model_name)
        
        responses = []
        logger.info(f"Processing {len(query_list)} queries with model: {model_name}")
        
        # Process with progress bar
        for query in tqdm(query_list, desc="Processing queries"):
            try:
                response = self._process_single_query(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query {query.prompt_id}: {e}")
                responses.append(f"Error: {str(e)}")
        
        return responses
    
    def _process_single_query(self, query: Query) -> str:
        """
        Process a single query with image and text
        
        Args:
            query: Query object with image and prompt
            
        Returns:
            Response text from the model
        """
        return self._process_vision_query(query)

    
    def _process_vision_query(self, query: Query) -> str:
        """Process query with Qwen-VL model"""
        
        # Prepare the messages in Qwen-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": query.image,
                    },
                    {"type": "text", "text": query.prompt_text},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode only the generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    
    def _process_vision_info(self, messages):
        """Extract images and videos from messages"""
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image":
                        image_inputs.append(item["image"])
                    elif item.get("type") == "video":
                        video_inputs.append(item["video"])
        
        return image_inputs, video_inputs

def load_queries_from_local(
    csv_path: str = "queries/prompts.csv",
    images_folder: str = "queries/images"
) -> List[Query]:
    """
    Load queries from local files
    
    Args:
        csv_path: Path to CSV file containing prompts and image names
        images_folder: Path to folder containing images
        
    Returns:
        List of Query objects
    
    CSV format expected:
        prompt_id, prompt_text, image_name
        1, "What is in this image?", "image1.jpg"
        2, "Describe the scene", "image2.png"
    """
    queries = []
    
    # Check if paths exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    logger.info(f"Loading queries from {csv_path}")
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            prompt_id = int(row.get('prompt_id', 0))
            prompt_text = row.get('prompt_text', '').strip()
            # update prompt_text
            prompt_text = update_prompt_context(prompt_text)
            image_name = row.get('image_name', '').strip()
            
            # Load image
            image_path = os.path.join(images_folder, image_name)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}, skipping query {prompt_id}")
                continue
            
            try:
                # Load image with PIL
                image = Image.open(image_path).convert('RGB')
                
                query = Query(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    image_name=image_name,
                    image_path=image_path,
                    image=image
                )
                queries.append(query)
                
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                continue
    
    logger.info(f"Loaded {len(queries)} queries successfully")
    return queries

def update_prompt_context(prompt: str) -> str:
    context_prompt = """
    **Role:** You are a visual assistant AI for a blind user.

    **Task:** The user will provide an image of their environment and ask to find an object. You must analyze the image from their perspective and give clear, verbal instructions to help them locate it.

    **Instruction Guidelines:**
        1.  **Use Relative Directions:** Always use terms like 'in front of you,' 'to your left,' 'to your right,' 'reach down,' or 'at your waist level.'
        2.  **Use Clear Distances:** Use 'inches' or 'feet.' '
        3.  **Use Landmarks:** Reference other objects. For example: "It's on the table, to the right 5 inches of your keyboard."
        4.  **Prioritize Safety:** If you see an obstacle, mention it. For example: "Move forward one step, but be aware there is a bag on the floor to your left."

    **User's Request:**

"""
    return context_prompt + prompt

def save_results_to_csv(
    results: List[Result],
    output_path: str = None
) -> str:
    """
    Save results to CSV file
    
    Args:
        results: List of Result objects
        output_path: Path for output CSV (optional, auto-generated if not provided)
        
    Returns:
        Path to the saved CSV file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_{timestamp}.csv"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving {len(results)} results to {output_path}")
    
    # Write results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['prompt_id', 'prompt_text', 'image_name', 'model_name', 'llm_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'prompt_id': result.prompt_id,
                'prompt_text': result.prompt_text,
                'image_name': result.image_name,
                'model_name': result.model_name,
                'llm_response': result.response
            })
    
    logger.info(f"Results saved successfully to {output_path}")
    return output_path

def eval(
    model_names: List[str],
    csv_path: str = "queries/prompts.csv",
    images_folder: str = "queries/images",
    output_folder: str = "results",
    device: str = None
) -> Dict[str, str]:
    """
    Evaluate multiple models and generate result CSVs
    
    Args:
        model_names: List of model names/paths to evaluate
        csv_path: Path to CSV file with prompts
        images_folder: Path to images folder
        output_folder: Folder to save result CSVs
        device: Device to run models on ('cuda', 'cpu', or None for auto)
        
    Returns:
        Dictionary mapping model names to output CSV paths
    
    Example:
        # For Qwen models
        model_names = [
            "Qwen/Qwen2-VL-2B-Instruct",  # Vision-language model
            "./models/qwen-3b-local"        # Local path to model
        ]
        results = eval(model_names)
    """
    logger.info(f"Starting evaluation for models: {model_names}")
    
    # Load queries once for all models
    queries = load_queries_from_local(csv_path, images_folder)
    
    if not queries:
        logger.error("No queries loaded, exiting evaluation")
        return {}
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize processor
    processor = LocalLLMProcessor(device=device)
    
    # Process each model
    output_paths = {}
    
    for model_name in model_names:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Call LLM in batch
            logger.info(queries)
            responses = processor.call_llm_batch(model_name, queries)
            
            # Create Result objects
            results = []
            for query, response in zip(queries, responses):
                result = Result(
                    prompt_id=query.prompt_id,
                    prompt_text=query.prompt_text,
                    image_name=query.image_name,
                    model_name=model_name,
                    response=response
                )
                results.append(result)
            
            # Save results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = model_name.replace("/", "_").replace(":", "_").replace("\\", "_")
            output_path = os.path.join(
                output_folder,
                f"results_{safe_model_name}_{timestamp}.csv"
            )
            
            saved_path = save_results_to_csv(results, output_path)
            output_paths[model_name] = saved_path
            
            logger.info(f"Completed evaluation for {model_name}")
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            output_paths[model_name] = None
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Evaluation Summary:")
    logger.info(f"{'='*50}")
    for model_name, path in output_paths.items():
        if path:
            logger.info(f"✓ {model_name}: {path}")
        else:
            logger.info(f"✗ {model_name}: Failed")
    
    return output_paths

def main():  
    model_names = [
        "Qwen/Qwen2-VL-2B-Instruct",
        # "/path/to//model"
    ]
    
    # Run evaluation
    results = eval(
        model_names=model_names,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\nEvaluation complete!")
    print("Results saved to:", results)

if __name__ == "__main__":
    main()