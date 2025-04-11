# # import json
# # import torch
# # from torch.utils.data import Dataset
# # from torch.nn.utils.rnn import pad_sequence

# # class TextGenerationDataset(Dataset):
# #     def __init__(self, file_path, tokenizer, max_length=128):
# #         """
# #         Reads a JSONL file where each line has "prompt" and "completion" fields,
# #         concatenates them, tokenizes the text, and forms input-target pairs.
# #         """
# #         self.data = []
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length
        
# #         with open(file_path, 'r', encoding='utf-8') as f:
# #             for line in f:
# #                 item = json.loads(line)
# #                 # Concatenate prompt and completion into one text
# #                 text = item['prompt'] + " " + item['completion']
# #                 token_ids = tokenizer.encode(text)
                
# #                 # Skip if the resulting token sequence is too short
# #                 if len(token_ids) < 2:
# #                     continue
                
# #                 # Truncate to maximum length
# #                 token_ids = token_ids[:max_length]
                
# #                 # Create input (all tokens except last) and target (all tokens except first)
# #                 input_ids = token_ids[:-1]
# #                 target_ids = token_ids[1:]
# #                 self.data.append((input_ids, target_ids))
    
# #     def __len__(self):
# #         return len(self.data)
    
# #     def __getitem__(self, idx):
# #         input_ids, target_ids = self.data[idx]
# #         return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

# # def collate_fn(batch):
# #     """
# #     Pads sequences in the batch. For the targets, pads with -100 so that the loss ignores them.
# #     """
# #     inputs, targets = zip(*batch)
# #     inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
# #     targets = pad_sequence(targets, batch_first=True, padding_value=-100)
# #     return inputs, targets


# import os
# import sentencepiece as spm
# import torch
# from sklearn.model_selection import train_test_split
# from torch.nn.utils.rnn import pad_sequence
 
 
# def combine_text_files(input_dir, output_file):
#     """
#     Combine all .txt files in the input directory into a single file.
#     :param input_dir: Directory containing raw .txt files
#     :param output_file: Path to the combined output file
#     """
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         for filename in os.listdir(input_dir):
#             if filename.endswith('.txt'):
#                 file_path = os.path.join(input_dir, filename)
#                 with open(file_path, 'r', encoding='utf-8') as infile:
#                     outfile.write(infile.read() + '\n')
#     print(f"Combined text files into {output_file}")
 
 
# def train_tokenizer(combined_file, model_prefix, vocab_size=10000):
#     """
#     Train a SentencePiece tokenizer using the combined text file.
#     :param combined_file: Path to the combined text file
#     :param model_prefix: Prefix for the tokenizer model files
#     :param vocab_size: Vocabulary size for the tokenizer
#     """
#     spm.SentencePieceTrainer.Train(
#         input=combined_file,
#         model_prefix=model_prefix,
#         vocab_size=vocab_size,
#         bos_id=0,  # Beginning of sequence token
#         eos_id=1,  # End of sequence token
#         unk_id=2,  # Unknown token (default, no need to redefine)
#         pad_id=3,  # Padding token
#         user_defined_symbols=["<bos>", "<eos>", "<pad>"]  # Custom symbols
#     )
#     print(f"Trained tokenizer saved as {model_prefix}.model")
 
 
# def tokenize_dataset(tokenizer_model, combined_file, output_dir, test_size=0.2):
#     """
#     Tokenize the dataset and split it into training and testing sets.
#     :param tokenizer_model: Path to the trained SentencePiece tokenizer model
#     :param combined_file: Path to the combined text file
#     :param output_dir: Directory to save tokenized datasets
#     :param test_size: Proportion of the dataset to include in the test split
#     """
#     # Load the tokenizer
#     sp = spm.SentencePieceProcessor()
#     sp.Load(tokenizer_model)
 
#     # Read the combined text file
#     with open(combined_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
 
#     # Tokenize each line
#     tokenized_lines = [sp.Encode(line.strip(), out_type=int) for line in lines if line.strip()]
 
#     # Split into training and testing sets
#     train_data, test_data = train_test_split(tokenized_lines, test_size=test_size, random_state=42)
 
#     # Save tokenized datasets
#     os.makedirs(output_dir, exist_ok=True)
#     torch.save(train_data, os.path.join(output_dir, 'train_data.pkl'))
#     torch.save(test_data, os.path.join(output_dir, 'test_data.pkl'))
#     print(f"Tokenized datasets saved in {output_dir}")
 
 
# class TextDataset(torch.utils.data.Dataset):
#     """
#     Custom Dataset for tokenized text data.
#     """
#     def __init__(self, data):
#         self.data = data
 
#     def __len__(self):
#         return len(self.data)
 
#     def __getitem__(self, idx):
#         sequence = self.data[idx]
#         # Ensure tensors are of type LongTensor
#         return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)
 
 
# def collate_fn(batch):
#     """
#     Custom collate function to pad and truncate sequences in a batch.
#     :param batch: List of (input_sequence, target_sequence) tuples
#     :return: Padded and truncated input and target tensors
#     """
#     inputs, targets = zip(*batch)
#     max_seq_length = 50  # Ensure sequences do not exceed max_seq_length
#     inputs_truncated = [seq[:max_seq_length] for seq in inputs]
#     targets_truncated = [seq[:max_seq_length] for seq in targets]
#     inputs_padded = pad_sequence(inputs_truncated, batch_first=True, padding_value=0)
#     targets_padded = pad_sequence(targets_truncated, batch_first=True, padding_value=0)
#     return inputs_padded, targets_padded
 
 
# if __name__ == "__main__":
#     # Paths and parameters
#     raw_data_dir = "/Users/jmoore/Documents/RNN Project 2/CSC7809_FoundationModels/Project2/data/raw"  # Directory containing raw .txt files
#     combined_file = "/Users/jmoore/Documents/RNN Project 2/CSC7809_FoundationModels/Project2/data/combined.txt"  # Combined text file
#     tokenizer_model_prefix = "/Users/jmoore/Documents/RNN Project 2/CSC7809_FoundationModels/Project2/data/tokenizer"  # Prefix for tokenizer model files
#     processed_data_dir = "/Users/jmoore/Documents/RNN Project 2/CSC7809_FoundationModels/Project2/data/processed"  # Directory to save processed datasets
#     vocab_size = 10000  # Vocabulary size for the tokenizer
 
#     # Step 1: Combine all .txt files into a single file
#     combine_text_files(raw_data_dir, combined_file)
 
#     # Step 2: Train the SentencePiece tokenizer
#     train_tokenizer(combined_file, tokenizer_model_prefix, vocab_size)
 
#     # Step 3: Tokenize the dataset and split into train/test sets
#     tokenizer_model = f"{tokenizer_model_prefix}.model"
#     tokenize_dataset(tokenizer_model, combined_file, processed_data_dir)
 
import os
import glob
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

def combine_text_files(input_folder, output_file):
    """Combine all .txt files from input_folder into one output_file."""
    file_list = glob.glob(os.path.join(input_folder, "*.txt"))
    with open(output_file, "w", encoding="utf-8") as outfile:
        for fname in file_list:
            with open(fname, "r", encoding="utf-8") as infile:
                content = infile.read().strip()
                outfile.write(content + "\n")
    print(f"Combined {len(file_list)} files into {output_file}")

def train_sentencepiece_tokenizer(input_file, model_prefix="spm_model", vocab_size=10000):
    """Train the SentencePiece tokenizer on the combined text."""
    spm.SentencePieceTrainer.train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
    )
    print(f"Trained SentencePiece model. Files generated: {model_prefix}.model and {model_prefix}.vocab")

def load_tokenizer(model_file="spm_model.model"):
    """Load a trained SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp

def tokenize_file(sp, input_file, output_file, as_ids=True):
    """Tokenize a file using SentencePiece and write tokenized output to file."""
    with open(input_file, "r", encoding="utf-8") as infile:
        text = infile.read()
    if as_ids:
        tokens = sp.encode_as_ids(text)
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(" ".join(map(str, tokens)))
    else:
        tokens = sp.encode_as_pieces(text)
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(" ".join(tokens))
    print(f"Tokenized data written to {output_file}")

def split_tokenized_data(tokenized_file, train_file="train_tokens.txt", test_file="test_tokens.txt", split_ratio=0.8):
    """Split the tokenized file (a space-separated list of token IDs) into train and test files."""
    with open(tokenized_file, "r", encoding="utf-8") as infile:
        tokens = infile.read().split()
    tokens = list(map(int, tokens))
    split_idx = int(len(tokens) * split_ratio)
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]
    with open(train_file, "w", encoding="utf-8") as outfile:
        outfile.write(" ".join(map(str, train_tokens)))
    with open(test_file, "w", encoding="utf-8") as outfile:
        outfile.write(" ".join(map(str, test_tokens)))
    print(f"Split tokenized data into {train_file} ({split_ratio*100}%) and {test_file} ({(1-split_ratio)*100}%).")

class LanguageModelDataset(Dataset):
    """PyTorch Dataset for language model training. Generates overlapping sequences."""
    def __init__(self, token_file, seq_length=32):
        with open(token_file, "r", encoding="utf-8") as infile:
            tokens = infile.read().split()
        self.tokens = list(map(int, tokens))
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y
