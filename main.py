import transformers
import random
import json
import torch
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field, fields
from enum import Enum

from trainer import DPRTrainer

class DPRDataset(object):
    def __init__(self, passage_path, query_path, tokenizer, max_passage_len, max_query_len, random_passaage_candidates=5,pad_to_multiple_of=64):
        passage = []
        with open(passage_path,"r") as f:
            for line in f:
                passage.append(json.loads(line))
        query = []
        with open(query_path,"r") as f:
            for line in f:
                query.append(json.loads(line))

        self.passage = {p[0]:p[1] for p in passage}
        self.query = query
        self.tokenizer = tokenizer
        self.max_passage_len = max_passage_len
        self.max_query_len = max_query_len
        self.random_passaage_candidates = random_passaage_candidates
        self.pad_to_multiple_of=pad_to_multiple_of
        self.all_passage_keys = set(self.passage.keys())
    
    def __len__(self):
        return len(self.query)
    
    def __getitem__(self, idx):
        query,answer_document_ids = self.query[idx]
        passage = self.passage[answer_document_ids]

        query = self.tokenizer(query,max_length=self.max_query_len,truncation=True,padding="longest",return_tensors="pt")
        passage = self.tokenizer(passage,max_length=self.max_passage_len,truncation=True,padding="longest",return_tensors="pt")

        negative_passage_pool = self.all_passage_keys - set([answer_document_ids])
        negative_passage = random.sample(negative_passage_pool,self.random_passaage_candidates)
        negative_passage = [self.passage[p] for p in negative_passage]
        negative_passage = self.tokenizer(negative_passage,max_length=self.max_passage_len,truncation=True,padding="longest",return_tensors="pt")

        return {
            "query_input_ids": query["input_ids"].squeeze(),
            "query_attention_mask": query["attention_mask"].squeeze(),
            "passage_input_ids": passage["input_ids"].squeeze(),
            "passage_attention_mask": passage["attention_mask"].squeeze(),
            "negative_passage_input_ids": negative_passage["input_ids"],
            "negative_passage_attention_mask": negative_passage["attention_mask"]
        }

    def collate_fc(self,batch):
        
        batch_size = len(batch)

        query_inputs = self.tokenizer.pad({
            "input_ids": [b["query_input_ids"] for b in batch],
            "attention_mask": [b["query_attention_mask"] for b in batch]
        },padding="longest",return_tensors="pt",pad_to_multiple_of=self.pad_to_multiple_of)
        passage_inputs = self.tokenizer.pad({
            "input_ids": [b["passage_input_ids"] for b in batch],
            "attention_mask": [b["passage_attention_mask"] for b in batch]
        },padding="longest",return_tensors="pt",pad_to_multiple_of=self.pad_to_multiple_of)
        negative_passage_inputs = self.tokenizer.pad({
            "input_ids": [x for b in batch for x in b["negative_passage_input_ids"]],
            "attention_mask": [x for b in batch for x in b["negative_passage_attention_mask"]]
        },padding="longest",return_tensors="pt",pad_to_multiple_of=self.pad_to_multiple_of)

        return {
            "query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"],
            "passage_input_ids": passage_inputs["input_ids"],
            "passage_attention_mask": passage_inputs["attention_mask"],
            "negative_passage_input_ids": negative_passage_inputs["input_ids"].view(batch_size,self.random_passaage_candidates,-1),
            "negative_passage_attention_mask": negative_passage_inputs["attention_mask"].view(batch_size,self.random_passaage_candidates,-1)
        }


@dataclass
class RunArguments(object):
    base_model_name_or_path: str = field(default="roberta-base")
    train_passage: str = field(default="data/train_passages.jsonl")
    train_query: str = field(default="data/train_queries.jsonl")
    dev_passage: str = field(default="data/dev_passages.jsonl")
    dev_query: str = field(default="data/dev_queries.jsonl")
    max_passage_len: int = field(default=512)
    max_query_len: int = field(default=128)
    random_passaage_candidates: int = field(default=5)

    def to_dict(self):
        d = dict((field.name, getattr(self, field.name)) for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

if __name__ == "__main__":
    parser = HfArgumentParser((RunArguments, TrainingArguments))
    run_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(run_args.base_model_name_or_path)
    train_dataset = DPRDataset(run_args.train_passage,run_args.train_query,tokenizer,run_args.max_passage_len,run_args.max_query_len,run_args.random_passaage_candidates)
    dev_dataset = DPRDataset(run_args.dev_passage,run_args.dev_query,tokenizer,run_args.max_passage_len,run_args.max_query_len,run_args.random_passaage_candidates)

    model = transformers.AutoModel.from_pretrained(run_args.base_model_name_or_path, add_pooling_layer=False)

    trainer = DPRTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=train_dataset.collate_fc,
        random_passaage_candidates=run_args.random_passaage_candidates
    )
    trainer.train()



