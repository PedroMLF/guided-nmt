import os
from dotenv import find_dotenv
from dotenv import load_dotenv

# Find and load dotenv
load_dotenv(find_dotenv())

class Config:

    def __init__(self):
 
        # Source and target languages
        self.SRC = os.environ.get("SRC")
        self.TGT = os.environ.get("TGT")

        # Dirs
        self.BASE_DIR = os.environ.get("BASE_DIR")
        self.TP_DIR = os.environ.get("TP_DIR")

        # Paths
        self.FASTTEXT_MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH")
        self.STOPWORDS_PATH = os.environ.get("STOPWORDS_PATH")
        self.EXTRA_ALIGN_PATH = os.environ.get("EXTRA_ALIGN_PATH")

