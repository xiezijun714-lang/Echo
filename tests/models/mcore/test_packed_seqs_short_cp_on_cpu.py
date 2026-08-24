import torch

from verl.models.mcore import util


def test_preprocess_packed_seqs_pads_short_sequences_for_each_cp_rank(monkeypatch):
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 2)

    input_ids = torch.tensor([[42, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)

    rank_outputs = []
    for cp_rank in (0, 1):
        monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank)
        output, packed = util.preprocess_packed_seqs(input_ids, attention_mask)
        rank_outputs.append(output)
        assert output.shape == (1, 8)
        assert packed.cu_seqlens_q_padded.tolist() == [0, 16]

    assert rank_outputs[0][0, 0].item() == 42
    assert torch.count_nonzero(rank_outputs[0]).item() == 1
    assert torch.count_nonzero(rank_outputs[1]).item() == 0


def test_preprocess_packed_seqs_preserves_tokens_across_short_cp_chunks(monkeypatch):
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 2)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    rank_outputs = []
    for cp_rank in (0, 1):
        monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank)
        output, _ = util.preprocess_packed_seqs(input_ids, attention_mask)
        rank_outputs.append(output[0])

    assert rank_outputs[0].tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert rank_outputs[1].tolist() == [5, 0, 0, 0, 0, 0, 0, 0]
