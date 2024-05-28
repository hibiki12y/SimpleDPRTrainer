import transformers
import torch
import math
import time
from tqdm import tqdm
from transformers.trainer_utils import speed_metrics, EvalLoopOutput

from utils import mean_pooling


class DPRTrainer(transformers.Trainer):
    def __init__(self,
                 random_passaage_candidates,
                 **kwargs):
        super().__init__(**kwargs)
        self.random_passaage_candidates = random_passaage_candidates
    
    def compute_loss(self, model, inputs, return_outputs=False, return_score=False):
        query_input_ids = inputs["query_input_ids"]
        query_attention_mask = inputs["query_attention_mask"]

        query_output = model(input_ids=query_input_ids,attention_mask=query_attention_mask)
        query = mean_pooling(query_output.last_hidden_state,query_attention_mask)

        passage_input_ids = inputs["passage_input_ids"]
        passage_attention_mask = inputs["passage_attention_mask"]
        passage_output = model(input_ids=passage_input_ids,attention_mask=passage_attention_mask)
        passage = mean_pooling(passage_output.last_hidden_state,passage_attention_mask)

        batch_size = query_input_ids.shape[0]
        negative_passage_input_ids = inputs["negative_passage_input_ids"].view(batch_size*self.random_passaage_candidates,-1)
        negative_passage_attention_mask = inputs["negative_passage_attention_mask"].view(batch_size*self.random_passaage_candidates,-1)
        
        negative_passage_output = model(input_ids=negative_passage_input_ids,attention_mask=negative_passage_attention_mask)
        negative_passage = mean_pooling(negative_passage_output.last_hidden_state,negative_passage_attention_mask)
        negative_passage = negative_passage.view(batch_size,self.random_passaage_candidates,-1)

        scores = torch.einsum("bd,ad->ba",query,passage)
        negative_scores = torch.einsum("bd,bcd->bc",query,negative_passage)
        scores = torch.cat([scores,negative_scores],dim=-1)

        labels = torch.arange(0,batch_size,dtype=torch.long).to(scores.device)
        loss = torch.nn.functional.cross_entropy(scores,labels)

        if return_score:
            return (loss, scores)

        return (loss, None) if return_outputs else loss
    
    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix= "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self.model.eval()
        start_time = time.time()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_loss = 0.0
        correct = 0
        total = 0
        for step, inputs in enumerate(tqdm(eval_dataloader,desc="Evaluating",disable=self.args.local_rank > 0)):
            loss, score = self.compute_loss(self.model, inputs, return_score=True)
            total_loss += loss.item()

            labels = torch.arange(0,score.shape[0],dtype=torch.long).to(score.device)
            correct += (score.argmax(dim=-1) == labels).sum().item()
            total += score.shape[0]
        metrics = {'eval_accuracy': correct/total,"eval_loss": total_loss/len(eval_dataloader)}

        output = EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(eval_dataloader))


        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics