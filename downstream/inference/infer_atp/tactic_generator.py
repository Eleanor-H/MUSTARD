import torch
import argparse
import json

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    BeamSearchScorer,
)
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartTokenizer, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, RobertaModel, AutoModelForCausalLM
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
import torch.nn.functional as F

import logging

logger = logging.getLogger('eval_search')

def logprobs_from_logits(logits, labels, start_id, end_id, return_mean_logprob=False):
    """Compute log softmax values from logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    logprobs = logprobs_labels.squeeze(-1)

    # we do not include the `STEP`'s logprob into acount, thus we +1 in here,
    # If the start index or end index do not exist, we use the full string.
    start_index = torch.vstack(
        [torch.where(example == start_id)[0][0] + 1 \
            if start_id in example \
            else torch.LongTensor([0]).to(example.device) \
            for example in labels]
    )
    end_index = torch.vstack(
        [torch.where(example == end_id)[0][0] \
            if end_id in example \
            else torch.LongTensor([example.size(-1)-1]).to(example.device) \
            for example in labels])

    # Get the indices of the elements we want to extract
    indices = torch.arange(logprobs.shape[1]).repeat(logprobs.shape[0], 1).to(logprobs.device)
    mask = (indices >= start_index) & (indices <= end_index)
    mask = mask.long()

    cum_logprobs = (logprobs * mask).sum(dim=1)
    if return_mean_logprob:
        # print("ori:", cum_logprobs)
        cum_logprobs /= mask.sum(dim=1)
        # print("avg:", cum_logprobs)
    return cum_logprobs

def preprocess_prompt(prompt):
    prompt = prompt.replace('\n', ' ').replace('\t',' ').strip()
    return prompt

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        input_ids: torch.Tensor
        padding_idx: int
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx)
    incremental_indices = torch.cumsum(mask, axis=1) * mask
    return incremental_indices


class TacticGenerator(object):
    def __init__(self,
                 devices,
                 model_name_or_path,
                 model_type,
                 model_class,
                 decoding_method,
                 length=1024,               # supposed to be part the request
                 temperature=1,             # supposed to be part the request
                 value_function_temperature=1,
                 value_function_type='binary',
                 value_function_model_type='gpt',
                 value_function_model_name_or_path=None,
                 topk=0,                    # supposed to be part the request
                 topp=0.0,                  # supposed to be part the request
                 repetition_penalty=1.0,
                 use_avg_logp=False,
                 return_mean_logprob=False,
                 dummy=False):   # supposed to be part the request
        self.length = length
        self.decoding_method = decoding_method
        self.use_avg_logp = use_avg_logp
        self.return_mean_logprob = return_mean_logprob

        self.temperature = temperature
        self.value_function_temperature = value_function_temperature
        self.value_function_model_type = value_function_model_type
        self.value_function_type = value_function_type

        # top-k and top-p sampling are not currently used
        self.topk = topk
        self.topp = topp
        self.repetition_penalty = repetition_penalty

        self.device = devices
        self.dummy = dummy
        self.trucation_stat = [0,0]

        if model_class.lower() == 'GPT2LMHeadModel'.lower():
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        elif model_class.lower() == 'Llama'.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path)
        else:
            raise NotImplementedError


        if value_function_model_type in \
                ['current_state_only', "root_state_and_current_state", "entire_trajectory", "previous_state_and_current_state"]:
            self.value_function_tokenizer = AutoTokenizer.from_pretrained(value_function_model_name_or_path)
            # if value_function_model_type == 'roberta_classification':
            self.value_function_model = RobertaForSequenceClassification.from_pretrained(
                    value_function_model_name_or_path)
            # elif value_function_model_type == 'roberta_similarity':
            #     self.value_function_model = RobertaModel.from_pretrained(
            #         value_function_model_name_or_path)
            self.value_function_model.eval()
        elif value_function_model_type == "external_gpt":
            self.value_function_tokenizer = self.tokenizer
            self.value_function_model = GPT2LMHeadModel.from_pretrained(
                value_function_model_name_or_path)
        elif value_function_model_type == 'gpt':
            pass

        # add special token
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_value_and_logprob(self, input_text, start_token, end_tken):
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.length, truncation=True, padding=True)
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits, value = outputs[0], outputs[2]

        start_token = self.tokenizer.encode(start_token)[-1]
        end_token = self.tokenizer.eos_token_id
        # output_start_token = self.tokenizer.encode(output_start_token)[0]
        logprobs = logprobs_from_logits(
            logits[:, :-1, :],
            inputs["input_ids"][:, 1:],
            start_token,
            end_token,
            return_mean_logprob=self.return_mean_logprob
        )

        logprobs = [v.item() for v in logprobs]
        value = [v.item() for v in value]
        return value, logprobs


    @torch.no_grad()
    def value_function_gpt(self, prefix_texts, root_state=None, value_function_model=None):
        assert value_function_model is not None
        logger.info(f"[VALUE_FUNCTION] Generating value function using gpt {self.value_function_type}")
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids = self.tokenizer(
            prefix_texts,
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=self.length - 2,
        ).to(value_function_model.device)

        # position_ids = create_position_ids_from_input_ids(input_ids['input_ids'], self.tokenizer.pad_token_id)

        # if input_ids['input_ids'].size(1) > self.length - 2:
        #     logger.warning("Input query length exceed limit, we skip the generation for this tactic state")
        #     # return non-zero value to prevent log_prob calculation error
        #     return [0.0001] * len(prefix_texts)

        outputs = value_function_model(**input_ids)
        pos = input_ids.attention_mask.sum(1) - 1
        next_token_logits = outputs.logits[torch.arange(outputs.logits.size(0)), pos, :]
        if outputs.logits.size(0) > 1:
            next_token_logits = next_token_logits.squeeze()
        INDEX = None
        if self.value_function_type == 'binary':
            # True False
            INDEX = self.tokenizer.encode(' False True')
        elif self.value_function_type == 'bucket':
            # A B C D E F G H I J K
            INDEX = self.tokenizer.encode(' A B C D E F G H I J K')
        else:
            raise NotImplementedError("value function type should be in [`binary` or `bucket`]")
        # li_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # if len(INDEX) == len(li_num) + 2: # seq2seq setting such as with BART, remove <s> and </s>.
        #     INDEX = INDEX[1:-1]
        assert INDEX is not None
        logits = torch.softmax(next_token_logits[:, INDEX] / self.value_function_temperature, dim=1)
        # scores = torch.FloatTensor(li_num).to(self.device)
        # scores = (logits * scores).sum(dim=1) / 10
        # # scores = scores + 0.0001 * torch.randn_like(scores)
        # # print(scores.tolist())
        # return scores.tolist()

        # INDEX = INDEX[0]
        # adding 0.00001 in order to prevent zero output
        scores = 1 - logits[:, 0] + 0.00001
        return scores.tolist()

    @torch.no_grad()
    def value_function_roberta_similarity(self, prefix_texts, original_states):
        logger.info("[VALUE_FUNCTION] Generating value function using roberta cl cosine similarity")
        value_function_device = self.value_function_model.device
        self.value_function_tokenizer.padding_side = 'right'
        input_ids = self.value_function_tokenizer(
            prefix_texts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors='pt',
        ).to(value_function_device)
        ori_input_ids = self.value_function_tokenizer(
            original_states,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors='pt',
        ).to(value_function_device)
        # if input_ids['input_ids'].size(1) > self.length - 2:
        #     logger.warning("Input query length exceed limit, we skip the generation for this tactic state")
        #     # return non-zero value to prevent log_prob calculation error
        #     return [0.0001] * len(prefix_texts)

        outputs = self.value_function_model(**input_ids)[1]
        ori_outputs = self.value_function_model(**ori_input_ids)[1]
        sim = torch.cosine_similarity(outputs, ori_outputs)

        return sim.tolist()

    def value_function_roberta_classification(self, prefix_texts):
        # print(prefix_texts)
        logger.info("[VALUE_FUNCTION] Generating value function using roberta classification")
        value_function_device = self.value_function_model.device
        self.value_function_tokenizer.padding_side = 'right'
        self.value_function_tokenizer.truncation_side = "left"
        inputs = self.value_function_tokenizer(
            text=prefix_texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        inputs_ver = self.value_function_tokenizer(
            text=prefix_texts,
            max_length=4096,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # import pdb
        # pdb.set_trace()
        # print(inputs.input_ids[:, :5])
        truncation_flag = (inputs.input_ids[:,1] != inputs_ver.input_ids[:,1]).sum(0).item() > 0 
        self.trucation_stat[0] += (inputs.input_ids[:,1] != 14740).sum(0).item()
        self.trucation_stat[1] += inputs.input_ids.size(0)
        inputs.to(value_function_device)
        logger.info(f"[VALUE_FUNCTION] truncatetion rate {self.trucation_stat[0]} /  {self.trucation_stat[1]} = {self.trucation_stat[0] / self.trucation_stat[1]}")
        outputs = self.value_function_model(**inputs)
        logits = outputs.logits
        logits = logits.softmax(dim=-1)
        ret = logits[:, 1].squeeze().tolist()
        if isinstance(ret, list):
            return ret, truncation_flag
        else:
            return [ret], truncation_flag

    def value_function(self, prefix_texts):

        if self.value_function_model_type == 'gpt':
            return self.value_function_gpt(prefix_texts, value_function_model=self.model)
        elif self.value_function_model_type == 'roberta_similarity':
            # return self.value_function_roberta_similarity(prefix_texts, root_states)
            raise NotImplementedError
        elif self.value_function_model_type in \
                ['current_state_only', "root_state_and_current_state", "entire_trajectory", "previous_state_and_current_state"]:
            return self.value_function_roberta_classification(prefix_texts)
        elif self.value_function_model_type == "external_gpt":
            return self.value_function_gpt(prefix_texts, value_function_model=self.value_function_model)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def batch_generate(self, prefix_texts, num_sample, calculate_logp=False):
        assert not isinstance(prefix_texts, list)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        input_ids = self.tokenizer(
            prefix_texts,
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=self.length - 40,
        ).to(self.device)

        # # assert if input length is to long for generation, spare 10 token for generated tactic
        # if input_ids['input_ids'].size(1) > self.length - 10:
        #     logger.warning("Input query length exceed limit, we skip the generation for this tactic state")
        #     return None, [], []

        if self.decoding_method == "sampling":
            # sampling
            # import pdb
            # pdb.set_trace()
            sample_outputs = self.model.generate(
                **input_ids,
                do_sample=True,
                # max_length=self.length,
                max_new_tokens=40,
                num_return_sequences=num_sample,
                temperature=self.temperature,
                top_k=self.topk,
                top_p=self.topp,
                repetition_penalty=self.repetition_penalty,
                output_scores=calculate_logp,
                return_dict_in_generate=calculate_logp,
                pad_token_id=self.tokenizer.eos_token_id
            )
        elif self.decoding_method == "beam_search":
            # beam search
            sample_outputs = self.model.generate(
                **input_ids,
                max_length=self.length,
                num_beams=num_sample * 2,
                early_stopping=True,
                output_scores=calculate_logp,
                return_dict_in_generate=calculate_logp,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=num_sample,
            )
        elif self.decoding_method == 'beam_sample':
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                max_length=self.length,
                num_beams=num_sample * 2,
                device=self.device,
            )
            # instantiate logits processors
            logits_processor = LogitsProcessorList(
                    [MinLengthLogitsProcessor(5, eos_token_id=self.tokenizer.eos_token_id)]
            )
            # instantiate logits processors
            logits_warper = LogitsProcessorList(
                [
                    TopKLogitsWarper(50),
                    TemperatureLogitsWarper(0.7),
                ]
            )
            sample_outputs = self.model.generate(
                **input_ids,
                max_length=self.length,
                do_sample=True,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                num_beams=num_sample * 2,
                early_stopping=True,
                output_scores=calculate_logp,
                return_dict_in_generate=calculate_logp,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=num_sample,
            )

        else:
            raise NotImplementedError(f"""
                The "{self.decoding_method}" decoding method has not yet been implemented.
            """)
        
        if not calculate_logp:
            log_probs = None
            gen_sequences = sample_outputs[:, input_ids['input_ids'].size(1):]
            sequences = self.tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)
            ori_sequence = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        else:
            if self.decoding_method == 'sampling':
                # only use id's that were generated
                gen_sequences = sample_outputs.sequences[:, input_ids['input_ids'].size(1):]

                # let's stack the logits generated at each step to a tensor and transform
                # logits to probs
                probs = torch.stack(sample_outputs.scores, dim=1).softmax(-1)

                # now we need to collect the probability of the generated token
                # we need to add a dummy dim in the end to make gather work
                gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

                # calculate non eos token mask, we exclude the token probabilities for eos tokens.
                # TODO: is removing eos prob correct?
                non_eos_token_mask = gen_sequences != self.tokenizer.eos_token_id
                log_probs = (gen_probs.log().nan_to_num(0) * non_eos_token_mask).sum(-1)
                # log_probs = gen_probs.prod(-1).log()

                # if we use length averaged log probability
                if self.use_avg_logp:
                    log_probs = log_probs / non_eos_token_mask.sum(-1)

                # Add small random Gaussian noise to prevent priority queue errors
                log_probs += 0.00001 * torch.randn_like(log_probs)
                log_probs = log_probs.tolist()
            
            elif self.decoding_method == 'beam_search':
                # unfortunately, this log_probs calculate the log probablity of the
                # entire sentence. (need confirmation)
                log_probs = sample_outputs.sequences_scores
                if self.use_avg_logp:
                    log_probs *= sample_outputs.sequences.size(-1)
                log_probs += 0.00001 * torch.randn_like(log_probs)
                log_probs = log_probs.tolist()

            gen_sequences = sample_outputs.sequences[:, input_ids['input_ids'].size(1):]
            sequences = self.tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)
            ori_sequence = self.tokenizer.batch_decode(sample_outputs.sequences, skip_special_tokens=True)
        return log_probs, sequences, ori_sequence

    @torch.no_grad()
    def skip_generate(self, prefix_text, num_sample, logp=False, deduplicate=True):
        if isinstance(prefix_text, list):
            batch_size = 4
            scores, sequences, ori_sequences = [], [], []
            for i in range(len(prefix_text) // batch_size + 1):
                batch = prefix_text[i * batch_size : (i + 1) * batch_size]
                if len(batch) == 0:
                    continue
                score, sequence, ori_sequence = self.batch_generate(batch, num_sample, logp)
                scores.extend(score)
                sequences.extend(sequence)
                ori_sequences.extend(ori_sequence)
        else:
            scores, sequences, ori_sequences = self.batch_generate(prefix_text, num_sample, logp)

        result_set = set()
        result = []
        for idx, text in enumerate(sequences):
            text = 'SKIP' + text.strip()

            if deduplicate:
                if text not in result_set:
                    result_set.add(text)
                    if logp:
                        result.append((scores[idx], text))
                    else:
                        result.append(text)
            else:
                if logp:
                    result.append((scores[idx], text))
                else:
                    result.append(text)
        return result

    @torch.no_grad()
    def generate(self, prefix_text, num_sample, logp=True, generation_batch=2):
        # if False:
        #     # inputs = self.tokenizer(prefix_text, return_tensors="pt")
        #     # for k, v in inputs.items():
        #     #     inputs[k] = v.to(self.model.device)
        #     # result = self.model.generate(**inputs, max_length=1024, pad_token_id=self.tokenizer.eos_token_id,
        #     #                              output_scores=True, return_dict_in_generate=True)
        #     # import pdb
        #     # pdb.set_trace()
        #     if prefix_text.strip() in self.dummy_data:
        #         return [(0.0, self.dummy_data[prefix_text])]
        #     else:
        #         print("-----------------------------------")
        #         print("error prefix text:")
        #         print(prefix_text)
        #         print("-----------------------------------")
        #         return []

        if isinstance(prefix_text, list):
            raise NotImplementedError("For support of long generation sequence length, we remove batch generation")
            total_generate_sample = num_sample * len(prefix_text)
            scores, sequences, ori_sequences = [], [], []
            for i in range(total_generate_sample // generation_batch + 1):
                batch = prefix_text[i * generation_batch : (i + 1) * generation_batch]
                if len(batch) == 0:
                    continue
                score, sequence, ori_sequence = self.batch_generate(batch, num_sample, logp)
                scores.extend(score)
                sequences.extend(sequence)
                ori_sequences.extend(ori_sequence)
        else:
            total_generate_sample = num_sample
            assert total_generate_sample % generation_batch == 0
            scores, sequences, ori_sequences = [], [], []
            for i in range(total_generate_sample // generation_batch):
                score, sequence, ori_sequence = self.batch_generate(prefix_text, generation_batch, logp)
                scores.extend(score)
                sequences.extend(sequence)
                ori_sequences.extend(ori_sequence)

        result_set = set()
        result = []
        for idx, text in enumerate(sequences):
            logger.info(f'[MODEL GENERATE] {ori_sequences[idx]}')
            text = text.strip()

            if text not in result_set:
                result_set.add(text)
                if logp:
                    result.append((scores[idx], text))
                else:
                    result.append(text)
        return result

    def share_memory(self):
        self.model.share_memory()

    def req_generate(self, req):
        """Generation with respect to the request `req`, according the 
        specifications here:
            https://beta.openai.com/docs/api-reference/completions
        """
        prefix_text = req['prompt']
        max_length = req['max_tokens']
        num_sample = req['n']
        temperature = req['temperature']
        top_p = req['top_p']

        prefix_text = preprocess_prompt(prefix_text)

        input_ids = self.tokenizer.encode(
            prefix_text, return_tensors='pt').to(self.device)

        # Doc of `generate`: 
        #   https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
        # Doc of SampleOutput:
        #   https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation_utils.SampleDecoderOnlyOutput
        sample_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            num_return_sequences=num_sample,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=top_p
        )

        sequences = sample_outputs.sequences.cpu()
        scores = sample_outputs.scores
        logp_sequences = sequences[:, input_ids.size(1):]  # 去掉input部分
        # 处理一下score  32 100    ([32，1,  50270], [32, 50270],
        scores = [s.unsqueeze(1).cpu() for s in scores]   
        scores = torch.cat(scores, dim=1)        # [32, 100, 50270]
        scores = scores.softmax(dim=2).log() # log(p)

        logp_index = logp_sequences.reshape(num_sample, -1, 1)
        scores = scores.gather(2, logp_index).reshape(num_sample, -1)
        scores = scores * \
            (logp_sequences != self.tokenizer.convert_tokens_to_ids('<|endoftext|>'))

        scores = scores.nan_to_num(0)

        choices = []
        for i in range(len(logp_sequences)):
            s = logp_sequences[i]
            j = 0
            while j < len(s):
                if s[j] == self.tokenizer.eos_token_id:
                    break
                j += 1
            s_valid = s[:j]
            text = self.tokenizer.decode(s_valid)
            token_logprobs = scores[i][:j]
            tokens = [self.tokenizer.decode(idx) for idx in s_valid]
            choices.append(
                {
                    'logprobs': {
                        'token_logprobs': token_logprobs.tolist(),
                        'tokens': tokens,
                    },
                    'text': text
                }
            )
        
        response = {'choices': choices}
        return response


class RoteTacticGenerator(object):

    def __init__(self, mem_dict):
        if isinstance(mem_dict, str) and mem_dict.endswith('.json'):
            with open(mem_dict, 'r') as f:
                mem_dict = json.load(f)
            self.mem_dict = {}
            for k in mem_dict:
                new_k = k.replace('\n', ' ').replace('\t' ,' ').strip()
                if not new_k.endswith("PROOFSTEP"):
                    new_k = "GOAL " + new_k + " PROOFSTEP"
                self.mem_dict[new_k] = mem_dict[k]
        else:
            self.mem_dict = mem_dict

    def generate(self, prefix_text, num_sample=1, logp=True):
        if prefix_text in self.mem_dict:
            return [(0, self.mem_dict[prefix_text])]
        else:
            return []


class NLGRoteTacticGenerator(object):

    def __init__(self, mem_dict_path):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str,
                        required=False, help='设置使用哪些显卡')
    parser.add_argument('--dec_names_path', default='data/test.names',
                        type=str, required=False, help='原始语料')
    parser.add_argument('--batch_size', default=1, type=int,
                        required=False, help='batch size')
    parser.add_argument('--pretrained_model', default='/cache/proofGPT-finetune-256batch-1.3B-seqloss-39000step/',
                        type=str, required=False, help='模型起点路径')
    parser.add_argument('--model_type', default='/cache/proofGPT-finetune-256batch-1.3B-seqloss-39000step/',
                        type=str, required=False, help='模型起点路径')
    parser.add_argument('--temperature', default=1, type=float,
                        required=False, help='generation temperature')
    parser.add_argument('--max_length', default=2048, type=int,
                        required=False, help='generation maximun length')
    parser.add_argument('--topk', default=0.0, type=float,
                        required=False, help='generation top-k for top-k sampling')
    parser.add_argument('--topp', default=1.0, type=float,
                        required=False, help='generation top-p for top-p sampling')
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        required=False, help='generation repetition penalty')
    parser.add_argument('--nsamples', default=4, type=int, required=False, help='生成几个样本')
    parser.add_argument('--prefix', default="GOAL ⊢ ∀ {α : Type u} (b : buffer α) (i : ℕ) (h : i < b.size) (v : α), b.write ⟨i, h⟩ v = b.write' i v PROOFSTEP",
            type=str, required=False, help='生成文章的开头')

    args = parser.parse_args()
    # print('args:\n' + args.__repr__())

    tg = TacticGenerator(
        devices=args.device,
        model_name_or_path=args.pretrained_model,
        model_type=args.model_type,
        decoding_method='sampling',
        length=args.max_length,
        temperature=args.temperature,
        topk=args.topk,
        topp=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
    while True:
        prefix = ["DEC finset.min'_eq_of_dual_max' GOAL α : Type u_1, _inst_1 : linear_order α, s : finset α, hs_w : α, hs_h : hs_w ∈ s ⊢ (image id s).max' _ = s.min' _ PROOFSIZE",
                "DEC finset.sup_eq_Sup_image GOAL α : Type u_1, β : Type u_2, _inst_1 : complete_lattice β, s : finset α, f : α → β ⊢ s.sup f = Sup (f '' ↑s) PROOFSIZE",
                "DEC polynomial.chebyshev.T_of_two_le GOAL R : Type u_1, _inst_1 : comm_ring R, k : ℕ, h : 2 ≤ 2 + k ⊢ chebyshev.T R (2 + k) = 2 * X * chebyshev.T R (2 + k - 1) - chebyshev.T R (2 + k - 2) PROOFSIZE"]
        # prefix = ["DEC finset.sup_eq_Sup_image GOAL α : Type u_1, β : Type u_2, _inst_1 : complete_lattice β, s : finset α, f : α → β ⊢ s.sup f = Sup (f '' ↑s) PROOFSIZE"]
        prefix = """CONTEXT note IH = \<open>\<And>n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>1 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; Exit \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s))\<close> GOAL proof (state) this: \<lbrakk>c\<^sub>1 \<turnstile> ?n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p ?n\<^sub>2'; n = ?n\<^sub>2; Exit \<noteq> ?n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) goal (13 subgoals): 1. \<And>c\<^sub>1 n et c\<^sub>2 n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>1 \<turnstile> n -et\<rightarrow>\<^sub>p Exit; \<And>n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>1 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; Exit \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)); n \<noteq> Entry; c\<^sub>1;; c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; Label #:c\<^sub>1 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 2. \<And>c\<^sub>2 n et n' c\<^sub>1 n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>2 \<turnstile> n -et\<rightarrow>\<^sub>p n'; \<And>n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; n' \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)); n \<noteq> Entry; c\<^sub>1;; c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n \<oplus> #:c\<^sub>1 = n\<^sub>2; n' \<oplus> #:c\<^sub>1 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 3. \<And>b c\<^sub>1 c\<^sub>2 n\<^sub>2 n\<^sub>2'. \<lbrakk>if (b) c\<^sub>1 else c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; Label 0 = n\<^sub>2; Label 1 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. IEdge (\<lambda>cf. state_check cf b (Some true))\<^sub>\<surd> = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 4. \<And>b c\<^sub>1 c\<^sub>2 n\<^sub>2 n\<^sub>2'. \<lbrakk>if (b) c\<^sub>1 else c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; Label 0 = n\<^sub>2; Label (#:c\<^sub>1 + 1) \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. IEdge (\<lambda>cf. state_check cf b (Some false))\<^sub>\<surd> = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 5. \<And>c\<^sub>1 n et n' b c\<^sub>2 n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>1 \<turnstile> n -et\<rightarrow>\<^sub>p n'; \<And>n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>1 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; n' \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)); n \<noteq> Entry; if (b) c\<^sub>1 else c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n \<oplus> 1 = n\<^sub>2; n' \<oplus> 1 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 6. \<And>c\<^sub>2 n et n' b c\<^sub>1 n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>2 \<turnstile> n -et\<rightarrow>\<^sub>p n'; \<And>n\<^sub>2 n\<^sub>2'. \<lbrakk>c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; n' \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)); n \<noteq> Entry; if (b) c\<^sub>1 else c\<^sub>2 \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n \<oplus> #:c\<^sub>1 + 1 = n\<^sub>2; n' \<oplus> #:c\<^sub>1 + 1 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 7. \<And>b c' n\<^sub>2 n\<^sub>2'. \<lbrakk>while (b) c' \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; Label 0 = n\<^sub>2; Label 2 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. IEdge (\<lambda>cf. state_check cf b (Some true))\<^sub>\<surd> = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 8. \<And>b c' n\<^sub>2 n\<^sub>2'. \<lbrakk>while (b) c' \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; Label 0 = n\<^sub>2; Label 1 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. IEdge (\<lambda>cf. state_check cf b (Some false))\<^sub>\<surd> = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 9. \<And>b c' n\<^sub>2 n\<^sub>2'. \<lbrakk>while (b) c' \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; Label 1 = n\<^sub>2; Exit \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. IEdge \<Up>id = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) 10. \<And>c' n et n' b n\<^sub>2 n\<^sub>2'. \<lbrakk>c' \<turnstile> n -et\<rightarrow>\<^sub>p n'; \<And>n\<^sub>2 n\<^sub>2'. \<lbrakk>c' \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n = n\<^sub>2; n' \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)); n \<noteq> Entry; n' \<noteq> Exit; while (b) c' \<turnstile> n\<^sub>2 -et\<^sub>2\<rightarrow>\<^sub>p n\<^sub>2'; n \<oplus> 2 = n\<^sub>2; n' \<oplus> 2 \<noteq> n\<^sub>2'\<rbrakk> \<Longrightarrow> \<exists>Q Q'. et = IEdge (Q)\<^sub>\<surd> \<and> et\<^sub>2 = IEdge (Q')\<^sub>\<surd> \<and> (\<forall>s. (Q s \<longrightarrow> \<not> Q' s) \<and> (Q' s \<longrightarrow> \<not> Q s)) A total of 13 subgoals... \<and> (Q' s \<longrightarrow> \<not> Q s)) A total of 13 subgoals... \<and> (Q' s \<longrightarrow> \<not> Q s)) A total of 13 subgoals...\<and> (Q' s \<longrightarrow> \<not> Q s)) A total of 13 subgoals... PROOFSTEP"""
        result = tg.generate(prefix, 8)
        print(result)