import pytest
from sulp import sample

TEST_USERIDS = {
    # all from same group
    "single": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # mix of group sizes
    "mix": [0, 5, 4, 1, 1, 4, 0, 0, 3, 4, 2, 4, 3, 4, 3],
    # all groups have one record
    "alldiff": [18, 26, 22, 14, 32, 24, 20, 28, 16, 30,  6, 10, 34,  8, 12]
}

# with seed=4243
TEST_SAMPLE_LEN = [
    # edge case: all records sampled
    (1.0, { "single": 15,
    "mix": 15,
    "alldiff": 15}),
    # typical case
    (0.5, { "single": 15,
    "mix": 12,
    "alldiff": 8}),
    # edge case: very low sampling rate
    (0.00001, { "single": 0, 
    "mix": 0,
    "alldiff": 0})
]

def test_iter_sample():
    for q, expected_len in TEST_SAMPLE_LEN:
        for groupids in ["single", "mix", "alldiff"]:
            sampler = sample.GroupPoissonSampler(TEST_USERIDS[groupids],
                    q, seed=4243)
            # sample size
            out_idx = list(sampler)
            assert len(out_idx) == expected_len[groupids]
            # no repeated records
            uniques = set(out_idx)
            assert len(uniques) == len(out_idx)

def test_iter_groupids():
    for q, _ in TEST_SAMPLE_LEN:
        for groupids in ["single", "mix", "alldiff"]:
            sampler = sample.GroupPoissonSampler(TEST_USERIDS[groupids],
                    q, seed=4243)
            out_idx = list(sampler)
            seen_group_ids = set()
            last_group_id = None
            # check that group ids are consecutive
            for i in out_idx:
                group_id = TEST_USERIDS[groupids][i]
                if last_group_id is None:
                    last_group_id = group_id
                elif group_id != last_group_id:
                    assert last_group_id not in seen_group_ids
                    seen_group_ids.add(last_group_id)
                    last_group_id = group_id

