import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import LMDataset


def generate_text(
    model,
    tokenizer,
    max_length: int,
    prompt: str = None,
    temperature: float = 1.0,
) -> str:
    """Generate text using a language model.

    Args:
        model: Language model (RNN or Transformer)
        tokenizer: Tokenizer with encode() and decode() methods
        max_length: Maximum number of tokens to generate
        prompt: Initial text prompt to start generation
        temperature: Controls randomness (higher = more random)

    Returns:
        Generated text string including the initial prompt if provided
    """
    was_training = model.training
    model.eval()

    # Generate random token if no prompt provided
    if prompt is None:
        rand_id = torch.randint(0, tokenizer.vocab_size, (1,))
        prompt = tokenizer.decode(rand_id.tolist())

    # Convert initial text to tokens
    generated = list(prompt)
    input_ids = torch.tensor(tokenizer.encode(prompt))

    # Generate continuation
    with torch.no_grad():
        generated_tokens = model.generate(input_ids, max_length, temperature)
        generated += tokenizer.decode(generated_tokens.tolist())

    if was_training:
        model.train(was_training)

    return "".join(generated)


def format_generation_logging(generated_text: str, prompt: str = None) -> str:
    """Format generated text for logging in Markdown format.

    Args:
        generated_text: Full generated text
        prompt: Optional initial text prompt used for generation

    Returns:
        Formatted string with Markdown formatting
    """
    formatted_lines = []

    if prompt:
        formatted_lines.extend(
            [f"**Prompt:** {prompt}", "", "```", generated_text, "```"]
        )
    else:
        formatted_lines.extend(["**Random Start**", "", "```", generated_text, "```"])

    formatted_lines.extend(["---", ""])
    return "\n".join(formatted_lines)


def create_lm_dataloaders(config, context_length):
    with open(config.train_file.path, "r") as f:
        data = f.read()

    train_len = int(len(data) * config.train_file.train_ratio)
    train_data = data[:train_len]
    val_data = data[train_len:]

    tokenizer = config.tokenizer
    train_dataset = LMDataset(train_data, context_length, tokenizer)
    val_dataset = LMDataset(val_data, context_length, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader
