import torch
from typing import List
import sacrebleu
from .dataset import BatchCollator, Multi30kDataset
from torch.utils.data import DataLoader


class TranslationExamplesMixin:
    """Mixin class for generating translation examples during training."""

    def is_usable(self):
        """Check if the mixin can be used."""
        assert hasattr(self, "model")
        assert hasattr(self.model, "translate")
        assert hasattr(self, "val_loader")
        assert hasattr(self, "src_tokenizer")
        assert hasattr(self, "tgt_tokenizer")
        assert hasattr(self, "logger")

    def get_translation_examples(self, indices, example_type: str) -> str:
        """Get translation examples for given indices."""
        src_tokens = [self.val_loader.dataset[i][0] for i in indices]
        tgt_tokens = [self.val_loader.dataset[i][1] for i in indices]
        src_sents = self.src_tokenizer.decode_batch(src_tokens)
        tgt_sents = self.tgt_tokenizer.decode_batch(tgt_tokens)
        pred_sents = self.model.translate(src_sents)

        return format_translation_logging(
            src_sents, pred_sents, tgt_sents, f"{example_type} Examples"
        )

    def get_fixed_examples(self) -> str:
        """Get translations for fixed example indices."""
        fixed_indices = torch.tensor([0, 1, 2, 3, 4])
        return self.get_translation_examples(fixed_indices, "Fixed")

    def get_random_examples(self) -> str:
        """Get translations for random example indices."""
        random_indices = torch.randint(0, len(self.val_loader), (5,))
        return self.get_translation_examples(random_indices, "Random")

    def log_translation_examples(self, step: int) -> None:
        """Log both fixed and random translation examples."""
        formatted_result = self.get_fixed_examples()
        formatted_result += self.get_random_examples()
        self.logger.add_text("translations", formatted_result, global_step=step)


def format_translation_logging(
    src_sents: List[str],
    tgt_sents: List[str],
    ref_sents: List[str],
    section_header: str = None,
) -> str:
    """Format translation results in a readable Markdown format for Tensorboard logging.

    Args:
        src_sents: List of source sentences
        tgt_sents: List of model predictions/translations
        ref_sents: List of reference/ground truth sentences
        add_numbering: Whether to add sample numbers in output

    Returns:
        Formatted string combining all samples with Markdown formatting
    """
    if not (len(src_sents) == len(tgt_sents) == len(ref_sents)):
        raise ValueError("All input lists must have the same length")

    formatted_lines = []
    if section_header is not None:
        formatted_lines.append(f"\n### {section_header}\n")

    for idx, (src, tgt, ref) in enumerate(zip(src_sents, tgt_sents, ref_sents)):
        sample = []
        sample.extend(
            [
                f"**Source:** {src} ",
                f"**Prediction:** {tgt} ",
                f"**Reference:** {ref} ",
                "---",
            ]
        )
        formatted_lines.extend(sample)

    return "\n\n".join(formatted_lines)


def compute_bleu(
    model, val_loader, src_tokenizer, tgt_tokenizer, beam_width=None, num_examples=None
):
    """Compute BLEU score on the validation dataset."""
    hypotheses = []
    references = []
    for i, (src, tgt) in enumerate(val_loader):
        if num_examples is not None and i >= num_examples:
            break

        # Translate the source sentence
        src_sents = src_tokenizer.decode_batch(src.tolist())
        tgt_sents = tgt_tokenizer.decode_batch(tgt.tolist())
        pred_sents = model.translate(src_sents, beam_width=beam_width)

        # Log translations
        hypotheses.extend(pred_sents)
        references.extend(tgt_sents)

    # Compute BLEU score
    score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    return score


def create_nmt_dataloaders(config):
    src_tokenizer = config.src_tokenizer
    tgt_tokenizer = config.tgt_tokenizer

    colattor = BatchCollator(
        src_tokenizer.pad_id, tgt_tokenizer.pad_id, batch_first=True
    )
    train_dataset = Multi30kDataset(config.files.train, src_tokenizer, tgt_tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=colattor
    )

    val_dataset = Multi30kDataset(config.files.val, src_tokenizer, tgt_tokenizer)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, collate_fn=colattor
    )

    return train_loader, val_loader
