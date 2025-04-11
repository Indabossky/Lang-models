import os
import time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, random_split

from data_processing import (combine_text_files, train_sentencepiece_tokenizer, load_tokenizer,
                        tokenize_file, split_tokenized_data, LanguageModelDataset)
from language_models import RNNLanguageModel, LSTMLanguageModel, TransformerLanguageModel
from training import train_model, calculate_perplexity, compute_bleu_score, plot_loss_curves

def main():
    # Define file paths
    input_folder = "/Users/jmoore/Documents/RNN Project 2/CSC7809_FoundationModels/Project2/data/raw"
    combined_file = "combined.txt"
    tokenized_file = "combined_tokenized.txt"
    train_token_file = "train_tokens.txt"
    test_token_file = "test_tokens.txt"

    # Data Processing
    combine_text_files(input_folder, combined_file)
    train_sentencepiece_tokenizer(combined_file, model_prefix="spm_model", vocab_size=10000)
    sp = load_tokenizer("spm_model.model")
    tokenize_file(sp, combined_file, tokenized_file, as_ids=True)
    split_tokenized_data(tokenized_file, train_file=train_token_file, test_file=test_token_file, split_ratio=0.8)

    # Create datasets and loaders
    seq_length = 32
    train_dataset = LanguageModelDataset(train_token_file, seq_length=seq_length)
    test_dataset = LanguageModelDataset(test_token_file, seq_length=seq_length)
    
    valid_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model hyperparameters
    vocab_size = sp.get_piece_size()
    embedding_dim, hidden_dim = 256, 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models dictionary
    models = {
        "RNN": RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers=1),
        "LSTM": LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers=1),
        "Transformer": TransformerLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers=2, nhead=2, max_seq_length=seq_length)
    }
    
    results = defaultdict(dict)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        trained_model, train_losses, valid_losses = train_model(model, train_loader, valid_loader, device,
                                                                epochs=30, lr=1e-3, early_stopping_patience=3)
        test_perplexity = calculate_perplexity(trained_model, test_loader, device)
        print(f"{model_name} Test Perplexity: {test_perplexity:.2f}")
        plot_loss_curves(train_losses, valid_losses, model_name)
        results[model_name]['model'] = trained_model
        results[model_name]['perplexity'] = test_perplexity

    # Text Generation and BLEU evaluation
    prompt_text = "Which do you prefer? Dogs or cats?"
    print("\nText Generation:")
    for model_name, info in results.items():
        model = info['model']
        generated_text = model.prompt(prompt_text, sp, max_seq_length=50, temperature=1.0, device=device)
        print(f"\n[{model_name}] Generated Text:")
        print(generated_text)
        reference_text = prompt_text + " I prefer dogs because they are loyal and friendly."
        bleu = compute_bleu_score(reference_text, generated_text)
        results[model_name]['BLEU'] = bleu
        print(f"{model_name} BLEU Score: {bleu:.4f}")

    # Print summary of evaluation
    print("\nSummary of model evaluation:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Perplexity = {metrics['perplexity']:.2f}, BLEU Score = {metrics['BLEU']:.4f}")

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"\nTotal Execution Time: {(time.time() - start_time)/60:.2f} minutes.")
