import torch
import torch.nn as nn
import torch.nn.functional as F

from exercise_utils.nlp.nmt.dataset import to_padded_tensor


class Seq2Seq(nn.Module):
    def __init__(self, EncoderCls, DecoderCls, config):
        super().__init__()
        self.encoder = EncoderCls(config)
        self.decoder = DecoderCls(config)
        self.src_tokenizer = config.src_tokenizer
        self.tgt_tokenizer = config.tgt_tokenizer

    def forward(self, src, tgt):
        """
        Args:
            src: input tokens ids of shape (batch_size, src_seq_len)
            tgt: target tokens ids of shape (batch_size, tgt_seq_len)

        Returns:
            logits: logits from the first token to the last token (excluding <end> token)
                of shape (batch_size, tgt_seq_len-1, vocab_size)
            score: log probability of the target sequence normalized by its length (batch_size,)

        """
        encoder_hidden = self.encoder(src, self.src_tokenizer.pad_id)
        logits, _ = self.decoder(
            tgt[:, :-1], encoder_hidden, self.tgt_tokenizer.pad_id
        )  # (batch_size, tgt_seq_len-1, vocab_size)
        tgt_true = tgt[:, 1:]  # remove <start> token

        log_probs = F.log_softmax(logits, dim=-1)

        # masking out padding tokens
        mask = (
            tgt_true != self.tgt_tokenizer.pad_id
        ).float()  # (batch_size, tgt_seq_len-1)

        # compute each target length
        tgt_lens = mask.sum(1)  # (batch_size,)

        # compute log probability of the target sequence
        scores = torch.gather(log_probs, index=tgt_true.unsqueeze(-1), dim=-1).squeeze(
            -1
        )  # (batch_size, tgt_seq_len-1)

        # compute scores
        score = (scores * mask).sum(1) / tgt_lens  # (batch_size,)
        return logits, score

    def compute_loss(self, src, tgt):
        logits, score = self.forward(src, tgt)
        tgt_true = tgt[:, 1:]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_true.flatten(),
            ignore_index=self.tgt_tokenizer.pad_id,
        )
        return loss

    @torch.no_grad()
    def translate(
        self, src_sents: list[str], max_len=100, beam_width=None
    ) -> list[str]:
        """
        Translate a batch of source sentences.

        Args:
            src_sents: a list of source sentences
            max_len: maximum length of the target sentence

        Returns:
            tgt_sents: a list of translated sentences
        """
        was_training = self.training
        self.eval()

        # tokenize the source sentences
        src = self.src_tokenizer.encode_batch(src_sents)
        src = to_padded_tensor(src, self.src_tokenizer.pad_id, batch_first=True)
        src = src.to(self.device)

        # encode the source sentences
        encoder_hidden = self.encoder(src, self.src_tokenizer.pad_id)

        # start token
        x = (
            torch.tensor([[self.tgt_tokenizer.start_id]])
            .expand(len(src_sents), 1)
            .to(self.device)
        )  # (batch_size, 1)
        hidden = encoder_hidden

        # Store output token ids for each sentence
        output_ids = [[] for _ in range(len(src_sents))]
        finished_sentences = torch.tensor([False] * len(src_sents)).to(self.device)

        for _ in range(max_len):
            x, hidden = self.decoder(x, hidden, self.tgt_tokenizer.pad_id)
            x = x.argmax(dim=-1)

            # Check for end tokens
            ended = (x == self.tgt_tokenizer.end_id).squeeze()
            finished_sentences = finished_sentences | ended

            # Stop if all sentences have hit the end token
            if finished_sentences.all():
                break

            # Store output token ids
            for i, token_id in enumerate(x.squeeze()):
                if not finished_sentences[i]:
                    output_ids[i].append(token_id.item())

        # Convert token ids to strings
        tgt_sents = self.tgt_tokenizer.decode_batch(output_ids)

        if was_training:
            self.train(was_training)

        return tgt_sents

    @property
    def device(self):
        return next(self.parameters()).device
