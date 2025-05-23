import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Union
from kaggle_evaluation.svg_constraints import SVGConstraints
from tqdm import tqdm
import os
import hashlib

def validate_svg(svg: str, max_svg_size: int = 10000) -> bool:
    svg_constraints = SVGConstraints(max_svg_size=max_svg_size)
    try:
        svg_constraints.validate(svg)
        return True
    except Exception as e:
        return False


class SVGRetriever:
    def __init__(
        self, 
        caption_field: str = "caption_cogvlm", 
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        n_samples: int = 1000,
        task_description: str = "Given a short description of an image, retrieve the most similar long description that better matches it",
        max_svg_size: int = 100000,
        batch_size: int = 64,
        cache_dir: str = "cache"  # Add cache directory parameter
    ):
        """
        Initialize the SVG retriever.
        
        Args:
            caption_field: The field to use for caption matching ('caption_cogvlm', 'caption_llava', etc.)
            model_name: The embedding model to use
            n_samples: Maximum number of samples to use from the dataset
            task_description: The task description for the embedding model
            max_svg_size: Maximum size in bytes for SVG files (default: 100000)
            batch_size: Number of samples to process at once (default: 32)
            cache_dir: Directory to store cached embeddings (default: "cache")
        """
        self.caption_field = caption_field
        self.model_name = model_name
        self.n_samples = n_samples
        self.task_description = task_description
        self.max_svg_size = max_svg_size
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        
        # Initialize cache
        self._embeddings_cache = None
        self._dataset_cache = None
        self._filtered_dataset_cache = None
    
    @property
    def dataset(self):
        """Lazy-load the dataset"""
        if self._dataset_cache is None:
            self._dataset_cache = load_dataset("starvector/text2svg-stack")
            print(f"Loaded dataset with {len(self._dataset_cache['train'])} samples")
        return self._dataset_cache
    
    def get_detailed_instruct(self, query: str) -> str:
        """Format query with instruction for the embedding model"""
        return f'Instruct: {self.task_description}\nQuery: {query}'
    
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Average pooling to create embeddings"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embed_texts(self, texts: List[str]) -> Tensor:
        """Create embeddings for a list of texts"""
        batch_dict = self.tokenizer(
            texts, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _filter_valid_svgs(self, samples):
        """Filter out invalid SVGs and those exceeding the size limit"""
        filtered_samples = []
        for sample in samples:
            svg = sample["Svg"]
            # Check SVG size
            if len(svg.encode('utf-8')) > self.max_svg_size:
                continue
            # Validate SVG structure
            if not validate_svg(svg, self.max_svg_size):
                continue
            filtered_samples.append(sample)
        
        print(f"Filtered dataset: {len(filtered_samples)}/{len(samples)} valid SVGs")
        return filtered_samples
    
    @property
    def filtered_dataset(self):
        """Get dataset with only valid SVGs within size limit"""
        if self._filtered_dataset_cache is None:
            samples = self.dataset["train"]
            # Limit to n_samples
            if self.n_samples is not None:
                samples = samples[:min(self.n_samples, len(samples))]
            
            # Filter valid SVGs
            self._filtered_dataset_cache = self._filter_valid_svgs(samples)
        
        return self._filtered_dataset_cache
    
    def _get_cache_path(self):
        """Generate a unique cache file path based on the configuration"""
        # Create a hash based on important parameters
        config_str = f"{self.model_name}_{self.caption_field}_{self.n_samples}_{self.task_description}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:10]
        
        return os.path.join(self.cache_dir, f"svg_embeddings_{config_hash}.pt")
    
    def save_embeddings(self, embeddings, samples):
        """Save embeddings and sample metadata to disk"""
        cache_path = self._get_cache_path()
        
        # Extract minimal sample data to save from the dict structure
        sample_metadata = {
            "Filename": samples["Filename"],
            "Svg": samples["Svg"],
            self.caption_field: samples[self.caption_field]
        }
        
        # Save as PyTorch file
        torch.save({
            "embeddings": embeddings,
            "samples": sample_metadata,
            "config": {
                "model_name": self.model_name,
                "caption_field": self.caption_field,
                "n_samples": self.n_samples,
                "task_description": self.task_description
            }
        }, cache_path)
        
        print(f"Embeddings saved to {cache_path}")
    
    def load_embeddings(self):
        """Load embeddings and sample metadata from disk if available"""
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached embeddings from {cache_path}")
                cache_data = torch.load(cache_path)
                
                # Verify config matches current settings
                config = cache_data["config"]
                if (config["model_name"] == self.model_name and
                    config["caption_field"] == self.caption_field and
                    config["n_samples"] == self.n_samples and
                    config["task_description"] == self.task_description):
                    
                    return cache_data["embeddings"], cache_data["samples"]
                else:
                    print("Cache config mismatch, recomputing embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        return None, None
    
    @property
    def embeddings(self) -> Tuple[Tensor, List[Dict]]:
        """Get or compute embeddings for the filtered dataset (cached)"""
        if self._embeddings_cache is None:
            # Try to load from disk cache first
            embeddings, samples = self.load_embeddings()
            
            if embeddings is not None and samples is not None:
                self._embeddings_cache = (embeddings, samples)
            else:
                # Use filtered samples or fall back to limited dataset
                # samples = self.filtered_dataset
                samples = self.dataset["train"]

                # Get captions
                captions = samples[self.caption_field]

                # Compute embeddings in batches
                print(f"Computing embeddings for {len(captions)} captions...")
                
                all_embeddings = []
                # Process in batches with tqdm progress bar
                for i in tqdm(range(0, len(captions), self.batch_size), desc="Computing embeddings"):
                    batch_captions = captions[i:i+self.batch_size]
                    batch_embeddings = self.embed_texts(batch_captions)
                    all_embeddings.append(batch_embeddings)
                
                # Concatenate all batch embeddings
                embeddings = torch.cat(all_embeddings, dim=0)
                
                # Cache results
                self._embeddings_cache = (embeddings, samples)
                print(f"Embeddings computed and cached: {embeddings.shape}")
                
                # Save to disk for future use
                self.save_embeddings(embeddings, samples)
            
        return self._embeddings_cache
    
    def predict(self, description: str) -> str:
        """
        Find the SVG that best matches the given description.
        
        Args:
            description: A description of the desired SVG
            
        Returns:
            The SVG string of the best match
        """
        # Get database embeddings and samples
        db_embeddings, samples = self.embeddings
        
        # Create query embedding
        query_text = self.get_detailed_instruct(description)
        query_embedding = self.embed_texts([query_text])
        
        # Compute similarity scores
        scores = (query_embedding @ db_embeddings.T)
        best_idx = torch.argmax(scores, dim=1).item()
        best_score = scores[0, best_idx].item()
        
        # print(f"Best match score: {best_score:.2f}")
        svg = samples["Svg"][best_idx]
    
        return svg
    
    def predict_with_details(self, description: str) -> Dict:
        """
        Find the SVG that best matches the given description and return details.
        
        Args:
            description: A description of the desired SVG
            
        Returns:
            Dict with SVG and metadata about the match
        """
        # Get database embeddings and samples
        db_embeddings, samples = self.embeddings
        
        # Create query embedding
        query_text = self.get_detailed_instruct(description)
        query_embedding = self.embed_texts([query_text])
        
        # Compute similarity scores
        scores = (query_embedding @ db_embeddings.T)
        best_idx = torch.argmax(scores, dim=1).item()
        best_score = scores[0, best_idx].item()
        
        return {
            "svg": samples["Svg"][best_idx],
            "filename": samples["Filename"][best_idx],
            "caption": samples[self.caption_field][best_idx],
            "score": best_score
        }


# Example usage
if __name__ == "__main__":
    # Initialize the retriever
    retriever = SVGRetriever(
        caption_field="caption_cogvlm",
        batch_size=128,
        # n_samples=100  # Limit to 100 samples for testing
    )
    
    # Test a query
    description = "a purple forest at dusk"
    svg_string = retriever.predict(description)
    
    # Save to file for inspection
    os.makedirs("output", exist_ok=True)
    with open("output/matched_svg.svg", "w") as f:
        f.write(svg_string)
    
    print(f"SVG saved to output/matched_svg.svg")
    
    # Get more details about the match
    details = retriever.predict_with_details(description)
    print(f"Matched caption: {details['caption']}")
    print(f"Match score: {details['score']:.2f}")
