import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from typing import List, Optional, Union
import pandas as pd

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    prompt: bool = False,
    template: Optional[str] = None,
    label_word_list=None,
    first_sent_limit: Optional[int] = None,
    other_sent_limit: Optional[int] = None,
    truncate_head=False,
    support_labels=None,

    # lxuechen: Not sure why these were included originally.
    task_name=None,
    gpt3=False,

    # lxuechen: For checking the dataset.
    early_return=False,
):
    """Tokenize (potentially multiple) sentences according to a potential pattern.

    Args:
        input_text_list: A list of strings.
        max_length: Maximum length of the overall output id list.
        tokenizer: HF tokenizer object.
        prompt (bool): Tokenize the sentences according to the pattern described in `template` if True.
        template (str): The pattern.
        label_word_list (list): A list of strings for words that are labels.
        first_sent_limit (int): Maximum length the first sentence should occupy.
        other_sent_limit (int): Maximum length the other sentence should occupy in the output.
        truncate_head (bool): If True, remove some head tokens when the list of tokenized ids is longer than the limit.
        support_labels: Only useful in gpt3 setting.

    Returns:
        A dictionary describing the current example with keys 'input_ids', 'attention_mask', 'mask_pos'.
    """

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = []  # Only for BERT
    mask_pos = None  # Position of the mask token

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's 
            in-context learning

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = {
            'cls': tokenizer.cls_token_id,
            'mask': tokenizer.mask_token_id,
            'sep': tokenizer.sep_token_id,
            'sep+': tokenizer.sep_token_id,
        }
        template_list = template.split('*')  # Get variable list in the template
        segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.

        for part_id, part in enumerate(template_list):
            if part == "":
                continue
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                if part == 'cls' and 'T5' in type(tokenizer).__name__:
                    # T5 does not have cls token
                    continue
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id])
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            else:
                # Just natural language prompt
                part = part.replace('_', ' ')
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer.convert_tokens_to_ids(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__:  # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]

    if early_return:
        return input_ids

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    len_input_ids = len(input_ids)
    if len_input_ids < max_length:
        input_ids.extend([tokenizer.pad_token_id] * (max_length - len_input_ids))
        attention_mask.extend([0] * (max_length - len_input_ids))
        token_type_ids.extend([0] * (max_length - len_input_ids))

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos

    return result


class GlueDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        raw_datasets = load_dataset("glue", data_args.dataset_name, cache_dir="~/.cache/huggingface/infill" if data_args.infill else None)
        self.tokenizer = tokenizer
        self.data_args = data_args
        # labels
        self.is_regression = data_args.dataset_name == "stsb"
        if not self.is_regression:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
            print(self.label_list)
        else:
            self.num_labels = 1
        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if self.data_args.infill:
            assert self.data_args.mapping is not None
            self.label_to_word = eval(self.data_args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(self.tokenizer.tokenize(' ' + self.label_to_word[key])) == 1

                    self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(' ' + self.label_to_word[key])[0]
                    )
                else:
                    self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(self.label_to_word[key])
                logger.info(
                    "Label {} to word {} ({})".format(
                        key, self.tokenizer.convert_ids_to_tokens(self.label_to_word[key]), self.label_to_word[key]
                    )
                )
                print("Label {} to word {} ({})".format(
                        key, self.tokenizer.convert_ids_to_tokens(self.label_to_word[key]), self.label_to_word[key]
                    ))

            if not self.is_regression:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]


        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = load_metric("glue", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def preprocess_function(self, examples):
        if not self.data_args.infill:
            # Tokenize the texts
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (
                examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        else:
            max_length = self.data_args.max_seq_length
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (
                    examples[self.sentence1_key], examples[self.sentence2_key])
            )

            result = {'input_ids':[], 'attention_mask':[], 'mask_pos':[]}
            if 'BERT' in type(self.tokenizer).__name__:
                # Only provide token type ids for BERT
                result['token_type_ids'] = []

            for idx in range(len(examples[self.sentence1_key])):
                inp = tokenize_multipart_input(
                    input_text_list=[sentences[idx] for sentences in args],
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    prompt=True,
                    template=self.data_args.template,
                    label_word_list=self.label_word_list,
                    first_sent_limit=self.data_args.first_sent_limit,
                    other_sent_limit=self.data_args.other_sent_limit,
                    # --- lxuechen: Enable toggling this.
                    truncate_head=self.data_args.truncate_head,
                    # ---
                )
                for key in inp.keys():
                    result[key].append(inp[key])
        print("infill tokenization")
        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}



# Text_labels = {0:"negative",1:"positive"}
# #Text_labels = {0:"bad",1:"good"}
# if text_label_dict:
#     self.data_label_idx = []
#     text_labels = text_label_dict[dataset]
#     for text_label_list in text_labels:
#         idx_list = [tokenizer.encode(text_label)[1] for text_label in text_label_list]
#         self.data_label_idx.append(idx_list)
# else:
#     self.data_label_idx = None
#     if dataset in ["sst2"]:
#         self.pos_idx = tokenizer.encode("positive")[1]
#         self.neg_idx = tokenizer.encode("negative")[1]
#     elif dataset in ["mnli", "snli"]:
#         self.pos_idx = tokenizer.encode("yes")[1]
#         self.neg_idx = tokenizer.encode("no")[1]
#         self.neutral_idx = tokenizer.encode("moderate")[1]
#     elif dataset in ["qnli"]:
#         self.pos_idx = tokenizer.encode("yes")[1]
#         self.neg_idx = tokenizer.encode("no")[1]
#
# class SST2_Dataset(Dataset):
#     def __init__(self,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH,seed=None):
#         if split=="train":
#             if local_dir:
#                 dataset = datasets.load_from_disk(local_dir)['train']
#             elif cache_dir:
#                 dataset = load_dataset('glue', 'sst2', split='train',cache_dir=cache_dir,download_mode="reuse_cache_if_exists")
#             else:
#                 dataset = load_dataset('glue', 'sst2', split='train')
#         elif split == "val":
#             if local_dir:
#                 dataset = datasets.load_from_disk(local_dir)['validation']
#             elif cache_dir:
#                 dataset = load_dataset('glue', 'sst2', split='validation',cache_dir=cache_dir,download_mode="reuse_cache_if_exists")
#             else:
#                 dataset = load_dataset('glue', 'sst2', split='validation')
#         elif split == "test":
#             if local_dir:
#                 dataset = datasets.load_from_disk(local_dir)['test']
#             elif cache_dir:
#                 dataset = load_dataset('glue', 'sst2', split='test',cache_dir=cache_dir,download_mode="reuse_cache_if_exists")
#             else:
#                 dataset = load_dataset('glue', 'sst2', split='test')
#         self.text_infilling = text_infilling
#         dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
#         if text_infilling:
#             def process_infilling(example):
#                 example["sentence"] = example["sentence"]+"It was <mask>."
#                 example['text_labels'] = Text_labels[example['labels']]
#                 return example
#             dataset = dataset.map(process_infilling)