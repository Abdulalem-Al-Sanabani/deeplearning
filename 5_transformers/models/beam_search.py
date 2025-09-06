import torch


class Hypothesis:
    def __init__(self, score: float, tokens: list[int]):
        self.score = score
        self.tokens = tokens

    def __lt__(self, other):
        return self._normalized_score() < other._normalized_score()

    def _normalized_score(self):
        assert len(self.tokens) > 0
        return self.score / len(self.tokens)


class DoublePriorityQueue:
    def __init__(self, max_size):
        self.data = []
        self.max_size = max_size

    def put(self, hypothesis):
        # If the queue is not full, replace the worst value
        if self.qsize() < self.max_size:
            self.data.append(hypothesis)
        else:
            worst = self.get_min()
            if worst < hypothesis:
                self.data.append(hypothesis)
            else:
                self.data.append(worst)

    def get_min(self):
        if not self.data:
            return None
        min_value = min(self.data)
        self.data.remove(min_value)
        return min_value

    def get_max(self):
        if not self.data:
            return None
        max_value = max(self.data)
        self.data.remove(max_value)
        return max_value

    def qsize(self):
        return len(self.data)


class Scorer:
    def __init__(self, batch_size, beam_width, end_id):
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.end_id = end_id
        self.max_hypotheses = beam_width
        # Set max_hypotheses to beam_width for simplicity
        self.hypotheses = [
            DoublePriorityQueue(max_size=self.max_hypotheses) for _ in range(batch_size)
        ]

    def is_done(self):
        for batch_idx in range(self.batch_size):
            if not self.hypotheses[batch_idx].qsize() >= self.max_hypotheses:
                return False
        return True

    def process(
        self,
        input_ids,
        next_token_scores,
        next_beam_indices,
        next_token_indices,
    ):
        """
        For each batch, this function basically schrinks the 2*beam_width candidates to beam_width candidates by:
            1. If the token is an end token, add it to the hypotheses
            2. If the token is a non-end token, add it to the beam outputs

        Args:
            input_ids: (batch_size * beam_width, t)
            next_token_scores: (batch_size, 2 * beam_width)
            next_beam_indices: (batch_size, 2 * beam_width)
            next_token_indices: (batch_size, 2 * beam_width)

        Returns:
            beam_outputs: {
                "next_token_scores": (batch_size, beam_width),
                "next_beam_indices": (batch_size, beam_width),
                "next_token_indices": (batch_size, beam_width),
            }
        """
        device = input_ids.device

        save_to_hypo_mask = (
            next_token_indices[:, : self.beam_width] == self.end_id
        )  # (batch_size, beam_width)
        save_to_hypo_mask = torch.cat(
            [save_to_hypo_mask, torch.full_like(save_to_hypo_mask, False)], dim=-1
        )  # (batch_size, 2 * beam_width)

        # Save all finished sentences to the hypotheses
        batch_idx, beam_idx = torch.where(save_to_hypo_mask)
        tokens = input_ids[self.beam_width * batch_idx + beam_idx]
        end_token = torch.full((tokens.shape[0], 1), self.end_id, dtype=torch.long).to(
            device
        )
        tokens = torch.cat([tokens, end_token], dim=-1).tolist()
        scores = next_token_scores[save_to_hypo_mask].tolist()

        for i in range(len(batch_idx)):
            hypothesis = Hypothesis(scores[i], tokens[i])
            batch_id = batch_idx[i].item()
            self.hypotheses[batch_id].put(hypothesis)

        # Save the best non-finished sentences to the beam outputs
        # For each batch, select the top beam_width candidates that are not ended
        out_mask = ~save_to_hypo_mask
        out_mask = out_mask & (next_token_indices != self.end_id)
        out_mask = (out_mask.cumsum(dim=-1) <= self.beam_width) & out_mask
        batch_idx, beam_idx = torch.where(out_mask)

        updated_next_token_indices = next_token_indices[batch_idx, beam_idx].view(
            self.batch_size, self.beam_width
        )
        updated_next_token_scores = next_token_scores[batch_idx, beam_idx].view(
            self.batch_size, self.beam_width
        )
        updated_next_beam_indices = next_beam_indices[batch_idx, beam_idx].view(
            self.batch_size, self.beam_width
        )

        beam_outputs = {}
        beam_outputs["next_token_indices"] = updated_next_token_indices
        beam_outputs["next_token_scores"] = updated_next_token_scores
        beam_outputs["next_beam_indices"] = updated_next_beam_indices

        return beam_outputs

    def finalize(self, input_ids):
        """
        Select the best sequence from both completed Hypotheses and in-progress beam candidates.

        Args:
            input_ids: (batch_size * beam_width, max_len)
                Active beam sequences that reached max length without end token

            beam_scores: (batch_size * beam_width)
                Cumulative scores for the active beam sequences

        Returns:
            sequences: list[list[int]]
                The best beam sequences for each batch
        """

        sequences = []
        for batch_idx in range(self.batch_size):
            if self.hypotheses[batch_idx].qsize() > 0:
                # If there are completed hypotheses, get the best one
                best_seq = self.hypotheses[batch_idx].get_max().tokens
            else:
                # If there are no completed hypotheses, get the best one from the in-progress beam candidates
                best_seq = input_ids[
                    batch_idx * self.beam_width, :
                ].tolist()  # This is sorted so the first one is the best
            sequences.append(best_seq)

        return sequences


class BeamSearchMixin:
    @torch.no_grad()
    def beam_search_decoding(
        self, memory, src_len, beam_width, max_len, start_id, end_id
    ):
        """
        Beam search decoding

        Returns:
            sequences (list[list[int]]): The best beam sequences for each batch
        """

        # initialize
        device = memory.device
        batch_size = memory.shape[0]
        batch_beam_size = batch_size * beam_width

        beam_scores = torch.zeros(batch_size, beam_width).to(device)
        beam_scores[:, 1:] = -float("inf")
        beam_scores = beam_scores.view(-1)  # (batch_size * beam_width)
        input_ids = torch.full((batch_beam_size, 1), start_id).to(device)

        beam_scorer = Scorer(batch_size, beam_width, end_id)

        # expand for beam search
        memory = memory.repeat_interleave(beam_width, dim=0)
        src_len = src_len.repeat_interleave(beam_width, dim=0)

        for t in range(max_len):
            logits = self.decode(input_ids, memory, src_len)
            next_token_logits = logits[:, -1, :].clone().float()
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            next_token_scores += beam_scores.view(batch_beam_size, 1)

            # Find top k candidates
            # Intead of finding beam_width top candidates, we find 2*beam_width candidates
            # since some (at most beam_width) of them will be ended and needed to be replaced
            vocab_size = logits.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, beam_width * vocab_size
            )
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_width, dim=1, largest=True, sorted=True
            )
            next_beam_indices = next_tokens // vocab_size
            next_token_indices = next_tokens % vocab_size
            beam_outputs = beam_scorer.process(
                input_ids, next_token_scores, next_beam_indices, next_token_indices
            )

            # Update beam scores (2*beam_width -> beam_width)
            beam_scores = beam_outputs["next_token_scores"].view(
                batch_size * beam_width
            )
            next_beam_indices = beam_outputs["next_beam_indices"]
            next_token_indices = beam_outputs["next_token_indices"]

            del logits
            torch.cuda.empty_cache()

            # Update input_ids
            batch_beam_idx = (
                torch.arange(batch_size).repeat_interleave(beam_width).to(device)
                * beam_width
            )
            batch_beam_idx = (batch_beam_idx + next_beam_indices.view(-1)).long()
            input_ids = torch.cat(
                [input_ids[batch_beam_idx, :], next_token_indices.view(-1, 1)], dim=-1
            ).long()

            if beam_scorer.is_done():
                break

        sequences = beam_scorer.finalize(input_ids)
        return sequences
