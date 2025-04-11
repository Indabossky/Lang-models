# import torch
# import torch.nn as nn

# class VanillaRNNModel(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size):
#         super(VanillaRNNModel, self).__init__()
#         # Embedding layer: convert token indices into dense vectors
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.hidden_size = hidden_size
        
#         # Weight matrices: 
#         # - Wxh: maps input embeddings to hidden state.
#         # - Whh: maps previous hidden state to new hidden state (no bias).
#         # - Who: maps hidden state to output logits (vocabulary scores).
#         self.Wxh = nn.Linear(embed_size, hidden_size)
#         self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.Who = nn.Linear(hidden_size, vocab_size)
        
#         # Tanh activation for the recurrence
#         self.tanh = nn.Tanh()
    
#     def init_hidden(self, batch_size):
#         """
#         Initialize the hidden state with zeros.
#         :param batch_size: Number of sequences in the batch.
#         :return: Tensor of shape (batch_size, hidden_size).
#         """
#         return torch.zeros(batch_size, self.hidden_size)
    
#     def forward(self, x):
#         """
#         Process the entire input sequence.
#         :param x: Tensor of shape (batch_size, seq_len) with token IDs.
#         :return: Tensor of shape (batch_size, seq_len, vocab_size) with logits for each time step.
#         """
#         batch_size, seq_len = x.size()
#         hidden = self.init_hidden(batch_size).to(x.device)
        
#         # Get embeddings: shape (batch_size, seq_len, embed_size)
#         embeddings = self.embedding(x)
#         outputs = []
        
#         # Process the sequence one timestep at a time
#         for t in range(seq_len):
#             # Get the embedding for the current timestep
#             x_t = embeddings[:, t, :]  # shape: (batch_size, embed_size)
            
#             # Update the hidden state (apply tanh to the sum of input and previous hidden state projections)
#             hidden = self.tanh(self.Wxh(x_t) + self.Whh(hidden))
            
#             # Compute output logits from the updated hidden state
#             out_t = self.Who(hidden)
#             outputs.append(out_t)
        
#         # Stack outputs along the time dimension: (batch_size, seq_len, vocab_size)
#         outputs = torch.stack(outputs, dim=1)
#         return outputs


import torch
import torch.nn as nn
import torch.nn.functional as F

# class BaseLanguageModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super(BaseLanguageModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.vocab_size = vocab_size
#         self.fc_out = nn.Linear(hidden_dim, vocab_size)

#     def sample_next(self, logits, temperature=1.0):
#         logits = logits / temperature
#         probs = F.softmax(logits, dim=-1)
#         next_token = torch.argmax(probs, dim=-1)
#         return next_token

#     def prompt(self, prompt_text, sp, max_seq_length=50, temperature=1.0, device="cpu"):
#         token_ids = sp.encode_as_ids(prompt_text)
#         input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
#         self.eval()
#         generated = token_ids.copy()
#         with torch.no_grad():
#             for _ in range(max_seq_length):
#                 logits = self.forward(input_ids)  # shape: [1, seq_length, vocab_size]
#                 next_token_logits = logits[:, -1, :]
#                 next_token = self.sample_next(next_token_logits, temperature)
#                 next_token_id = next_token.item()
#                 generated.append(next_token_id)
#                 input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
#         return sp.decode_ids(generated)

class RNNLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=1):
        super(RNNLanguageModel, self).__init__(vocab_size, embedding_dim, hidden_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc_out(output)
        return logits

class LSTMLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=1):
        super(LSTMLanguageModel, self).__init__(vocab_size, embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc_out(output)
        return logits

class TransformerLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, nhead=2, max_seq_length=512):
        # For Transformer, note that we use embedding_dim for hidden sizes
        super(TransformerLanguageModel, self).__init__(vocab_size, embedding_dim, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.embedding(x) + self.positional_embedding(positions)
        x = x.transpose(0, 1)  # Shape to [seq_length, batch, embedding_dim]
        output = self.transformer_encoder(x)
        output = output.transpose(0, 1)
        logits = self.fc_out(output)
        return logits
