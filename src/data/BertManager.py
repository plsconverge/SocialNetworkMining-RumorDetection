import os
import torch
from transformers import AutoTokenizer, AutoModel

import warnings
warnings.filterwarnings("ignore")


class BertManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def download(self, model_name='bert-base-chinese'):
        os.makedirs(self.model_dir, exist_ok=True)

        # check existence
        if os.path.exists(os.path.join(self.model_dir, "config.json")):
            print("Bert Model Already Downloaded")
            return

        print("Downloading Bert Model...")
        try:
            print("Downloading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(self.model_dir)

            print("Downloading Model...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(self.model_dir)

            print(f"Download Complete, model saved to {self.model_dir}")
            self._verify_download()

        except Exception as e:
            print("Failed to download: ", e)
            raise

    def _verify_download(self):
        required_files = [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json"
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                missing_files.append(file)

        if missing_files:
            print("Download Failed. Missing: ", missing_files)
        else:
            print("All necessary files have been downloaded")

    def load(self):
        print("Loading Model from local files...")

        if not os.path.exists(os.path.join(self.model_dir, "config.json")):
            print("Bert Model Not Downloaded")
            raise FileNotFoundError(f"Necessary Filed not Downloaded")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModel.from_pretrained(self.model_dir)

        print("Bert Model Loaded")
        return self.tokenizer, self.model


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(root_dir, "data", "models", "bert-base-chinese")
    manager = BertManager(model_dir)
    manager.download()
    # try:
    #     _, _ = manager.load()
    # except Exception as e:
    #     print("Failed to load Bert Model: ", e)
    #     raise

if __name__ == "__main__":
    main()
