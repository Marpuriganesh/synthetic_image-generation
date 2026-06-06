import datasets
from datasets import Dataset, DatasetDict

def convert_and_upload_to_hf(repo_id: str = "marpuriganesh/synthetic-ocr-43k"):
    """
    Loads local synthetic OCR datasets via CSV, converts string paths to 
    decoded Hugging Face Image objects, and pushes the collection to the Hub.
    """
    # 1. Load your local train and test CSV files into Hugging Face Dataset objects
    print("[HF Engine] Loading local CSV splits...")
    train_dataset = Dataset.from_csv("output/train.csv")
    test_dataset  = Dataset.from_csv("output/test.csv")
    
    # 2. Tell Hugging Face that the "path" column points to real images on disk
    # This instructs the library to convert text paths into decoded PIL/NumPy images
    print("[HF Engine] Casting file paths to native image features...")
    train_dataset = train_dataset.cast_column("path", datasets.Image())
    test_dataset  = test_dataset.cast_column("path", datasets.Image())
    
    # 3. Combine both splits into a single unified DatasetDict bundle
    hf_dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # 4. Push everything (images, text labels, and metadata splits) to the cloud
    print(f"[HF Engine] Uploading 43,500 payloads to Hugging Face Hub: {repo_id}...")
    hf_dataset_dict.push_to_hub(
        repo_id = repo_id,
        private = False
    )
    print("[Success] Dataset is live on the Hugging Face Hub!")

# Example usage (uncomment to run):
if __name__ == "__main__":
    convert_and_upload_to_hf()