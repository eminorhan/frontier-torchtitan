# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import pickle
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

import datasets
from datasets import load_dataset, load_from_disk, interleave_datasets
from datasets.distributed import split_dataset_by_node

datasets.logging.set_verbosity_error()

# map from dataset name to a local directory, or a dataset repository on the HF hub
_supported_datasets = {
    "c4": "allenai/c4",
    "full": "local cache",
}

def extract_code(rec):
    text = ""
    for f in rec["files"]:
        text += f"\n\n{f["text"]}"
    return text

class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process

    We currently support the c4 dataset, and a subset of it for testing purposes:
    c4_test (2K training entries)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at ...',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    Example use (c4):
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        shuffle_seed: int = 1
    ) -> None:
        
        # global random seed for the current training session
        self._seed = shuffle_seed

        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(f"Dataset {dataset_name} is not tested or verfied. Recommended datasets are: {list(_supported_datasets.keys())}")
            else:
                raise ValueError(f"Dataset {dataset_name} is not supported. Supported datasets are: {list(_supported_datasets.keys())}")

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            # c4 is huge, and requires both streaming and subset selection
            ds = load_dataset(dataset_path, name="realnewslike", split="train")
        elif dataset_name == "full":
            # 6 component datasets
            ds_dclm = load_dataset("Zyphra/Zyda-2", name="dclm_crossdeduped", split="train", streaming=True).select_columns("text")
            ds_fwe = load_dataset("Zyphra/Zyda-2", name="fwe3", split="train", streaming=True).select_columns("text")
            ds_dolma = load_dataset("Zyphra/Zyda-2", name="dolma-cc_crossdeduped-filtered", split="train", streaming=True).select_columns("text")
            ds_zyda = load_dataset("Zyphra/Zyda-2", name="zyda_crossdeduped-filtered", split="train", streaming=True).select_columns("text")
            ds_stack = load_from_disk("/lustre/orion/stf218/scratch/emin/huggingface/stack_v2_smol").to_iterable_dataset(num_shards=3000).select_columns("files")
            ds_finemath = load_dataset("HuggingFaceTB/finemath", name="finemath-3plus", split="train", streaming=True).select_columns("text")

            # interleave component datasets with given mixing probabilities
            ds = interleave_datasets(
                [ds_dclm, ds_fwe, ds_dolma, ds_zyda, ds_stack, ds_finemath],
                probabilities=[0.40, 0.44, 0.03, 0.02, 0.10, 0.01],
                seed=self._seed,
                stopping_strategy="all_exhausted"
                )
        else:
            ds = load_dataset(dataset_path, split="train")

        # shuffle shards and buffer
        ds = ds.shuffle(buffer_size=100000, seed=self._seed)

        # split across dp ranks
        self._data = split_dataset_by_node(ds, rank, world_size)

        self.dataset_name = dataset_name
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self._all_tokens: List[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in iter(self._data):
                if sample["files"] is None:
                    sample_text = sample["text"]
                else:
                    sample_text = extract_code(sample)  # handle code
                # logger.info(f"[rank {int(os.environ["RANK"])}] {sample["nemo_id"]}, {sample["repo_name"]}, {sample["url"]}, {self._seed}")  # test dataset iterator
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._all_tokens.extend(sample_tokens)

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}")
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    shuffle_seed
):
    hf_ds = HuggingFaceDataset(dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, shuffle_seed)

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)