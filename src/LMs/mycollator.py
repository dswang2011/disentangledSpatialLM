from typing import Dict, List
import torch
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping


class BlockMaskingDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm: bool, mlm_probability: float = 0.15):
        super().__init__(tokenizer=tokenizer, mlm = mlm, mlm_probability=mlm_probability)
        self.block_size = 10    # cannot mask more than 30 tokens at once;

    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], batch["position_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, position_ids: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # print('pos ids:', position_ids[:100,:100]) # [batch_size, dim] -> [12, 512]
        # print('label:', labels) # [batch_size, dim] -> [12, 512]
        # print('mask_indices:', masked_indices)
        # print('m size:',masked_indices.size())
        # print('l size:', labels.size())
        # print('p size:', position_ids.size())
        # count = torch.count_nonzero(masked_indices)
        # print('mask num:',count)

        # expand the masked_indices here
        masked_indices = self.extend_indices(position_ids, masked_indices)
        # test after masking!
        # count = torch.count_nonzero(masked_indices)
        # print('new mask num',count)
        # print('new m size:',masked_indices.size())
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


    def extend_indices(self, position_ids, masked_indices):
        # Convert the boolean matrix to a set of indices
        indices = set((i, j) for i, row in enumerate(masked_indices)
                    for j, value in enumerate(row) if value)
 
        # Iterate over each masked index
        for i, j in indices:
            # If the value at this index is not a number or is less than 2, skip it
            curr_val = position_ids[i][j].item()
            if not curr_val or curr_val < 2:
                continue
            
            # Extend leftwards until we hit a non-number or a number less than 2
            for k in range(j-1, -1, -1):
                value = position_ids[i][k].item()
                # stop conditions
                if not value or value <2 or j-k>3:
                    break
                elif value==2:
                    if curr_val>2:
                        masked_indices[i][k] = True
                        break
                    else:
                        break
                # extend
                masked_indices[i][k] = True
            
            # Extend rightwards until we hit a non-number or a number less than or equal to 2
            for k in range(j+1, len(position_ids[i])):
                value = position_ids[i][k].item()
                if not value or value <= 2 or k-j>3:
                    break
                masked_indices[i][k] = True
        
        return torch.tensor(masked_indices)



    def generate_masked_positions(self, position_ids, seq_len):
        positions_to_mask = set()
        mask_start = False
        mask_end = False

        for i in range(seq_len):
            if position_ids[i] == 2:
                if mask_start:
                    break
                else:
                    mask_start = True
                    mask_end = False
                    mask_size = 0
            elif position_ids[i] > 2 and not mask_end:
                mask_size += 1
            elif position_ids[i] == 2 and mask_start:
                mask_end = True
                if mask_size <= self.block_size:
                    positions_to_mask.update(range(max(0, i - mask_size - 1), i - 1))
                mask_size = 0

        if mask_start and not mask_end and mask_size >= self.block_size:
            positions_to_mask.update(range(max(0, seq_len - mask_size - 1), seq_len - 1))

        return positions_to_mask
