import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniLM(nn.Module):
    def __init__(self, rnn, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # create an instance of the RNN module
        self.rnn = rnn
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def init_hidden(self, batch_size):
        """Return initial hidden state for RNN."""
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.config.device)
        if self.config.rnn_type == "lstm":
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(self.config.device)
            return (h0, c0)
        else:
            return h0

    def forward(self, inputs, targets):
        """
        Args:
            inputs: integer tensor of shape (batch_size, T) containing token indices.
            targets: integer tensor of shape (batch_size, T) containing token indices.

        Returns:
            loss: scalar tensor containing the cross-entropy loss

        The relations hold for the inputs and targets tensors:
            targets[:, t] = inputs[:, t + 1] for 0 <= t < T - 1
            T = config.seq_len - 1
        """
        batch_size, _ = inputs.shape
        hidden = self.init_hidden(batch_size)
        inputs = F.one_hot(
            inputs, self.vocab_size
        ).float()  # (batch_size, T, vocab_size)
        out, _ = self.rnn(inputs, hidden)
        out = self.fc(out)
        out = out.view(-1, out.size(2))
        targets = targets.view(-1)
        loss = F.cross_entropy(out, targets)
        return loss

    @torch.no_grad()
    def generate(
        self,
        initial_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
    ):
        """
        Args:
            initial_tokens: Integer-encoded input tensor of shape (T, )
            max_length: Maximum number of tokens to generate
            temperature: Scaling factor for the logits

        Returns:
            generated_tokens: Integer tensor representing the generated tokens
        """
        generated_tokens = []

        x = initial_tokens.unsqueeze(0).to(self.config.device)
        hidden = self.init_hidden(1)

        for _ in range(max_length):
            x = F.one_hot(x, num_classes=self.vocab_size).float()
            out, hidden = self.rnn(x, hidden)
            out = out[:, -1, :]
            out = self.fc(out)
            out = out.squeeze(0)
            out = out / temperature
            probs = F.softmax(out, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            x = next_token.unsqueeze(0)

        return torch.tensor(generated_tokens)
