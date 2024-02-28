import logging
import math
import random
from copy import copy
import time

import numpy as np

from formal_system_client import LeanClient
from tactic_generator import TacticGenerator, RoteTacticGenerator
from utils import print
from environment_base import FormalSystemEnvironment

logger = logging.getLogger('eval_search')


class LeanEnvironment(FormalSystemEnvironment):
    """
    Lean specific operations
    """
    def __init__(self,
                 try_intros=False,
                 prompt_mode=2,
                 use_value_function=False,
                 server_port=-1,
                 random_value=False,
                 pass_at=1,
                 ) -> None:
        super().__init__()
        self.search_id = None
        self.try_intros = try_intros
        self.prompt_mode = prompt_mode
        self.use_value_function = use_value_function
        self.server_port = server_port
        self.random_value=random_value
        self.pass_at = pass_at

    def init_search(self, init_args=None):
        """
        Initialize lean-gym proof search with specific declartion name and namespaces
        return the environment_state for the proof search tree
        """
        decl_name, namespaces = init_args
        logger.info(f'[INIT SEARCH] {decl_name} {namespaces}')
        search = self.lean_server.init_search(decl_name, namespaces)
        if not search['error'] is None:
            logger.info(f'[INIT SEARCH] Failed with {search["error"]}')
            return None
        logger.info(f'[INIT SEARCH] Succeed!')
        self.search_id = search['search_id']
        if self.try_intros:    # to macht the format of the training dataset, use intros
            search = self.lean_server.run_tac(search_id=self.search_id,
                                              tactic_id=search['tactic_state_id'],
                                              tactic='try {intros}')
        return search

    def initialized_environment(self, params, device=0):
        if not params["use_dummy_tactic_generator"]:
            print(f"Creating tactic generator with decoding method `{params['decoding_method']}`")
            self.tactic_generator = TacticGenerator(
                devices=device,
                model_name_or_path=params['model_name_or_path'],
                model_type=params['model_type'],
                decoding_method=params['decoding_method'],
                length=params['max_seq_length'],
                temperature=params['temperature'],
                value_function_type=params['value_function_type'],
                value_function_temperature=params['value_function_temperature'],
                value_function_model_type=params['value_function_model_type'],
                value_function_model_name_or_path=params['value_function_model_name_or_path'],
                topk=params['topk'],
                topp=params['topp'],
                repetition_penalty=params['repetition_penalty'],
                dummy=params['dummy'],
                model_class=params["model_class"],
                return_mean_logprob=params["return_mean_logprob"]
            )
            self.value_function_model_type = params['value_function_model_type']
        else:
            raise NotImplementedError
        
        # self.lean_gym_dir = params['lean_gym_dir']
        self.lean_server = LeanClient(self.server_port)

    def reset_formal_system(self):
        logger.info('Resetting Lean-gym, this will lost all searching history')
        self.lean_server.reset_server()

    def setup_data(self, params, rank=0, world_size=1):
        with open(params['dec_names_path'], 'r', encoding='utf8') as file:
            decl_nms = file.readlines()

        # Duplicate theorems
        decl_nms = decl_nms * self.pass_at

        # seed_every_thing
        random.Random(42).shuffle(decl_nms)
        splited_decl_nms = [[] for _ in range(world_size)]
        for idx, decl_nm in enumerate(decl_nms):
            splited_decl_nms[idx % world_size].append(decl_nm)

        decl_nms = splited_decl_nms[rank]
        datasets = []
        for line in decl_nms:
            names = line.split()
            dec_names = names[0]
            namespaces = ' '.join(names[1:])
            datasets.append((dec_names, namespaces))    # ('elimhyps2', '')
        return datasets

    def preprocess_prompt(self, goal_text, prompt_mode, dec_name=None, is_value_function=False, gap=-1, skip_goal=None):
        """
        Create query prompt for language model.
        Should be used according to the training of the model and be one of:
                0 for <goal>   (and without replacing `\n`)
                1 for GOAL <goal> PROOFSTEP
                2 for DEC <decl_name> GOAL <goal> PROOFSTEP
                3 for DEC <decl_name> SKIP{gap} <skip goal> GOAL <goal> PROOFSTEP
                4 for DEC <decl_name> SKIP{gap}
                5 for DEC <dec_name> ERROR None GOAL <cleaned_goal_text> <keyword>
                6 for DEC <dec_name> RESULT SUCCESS ERROR None GOAL <cleaned_goal_text> <keyword>
                7 for RESULT SUCCESS ERROR None GOAL <cleaned_goal_text> <keyword>
        """
        keyword = 'PROOFSTEP'
        if is_value_function is True:
            keyword = 'PROOFSIZE'

        cleaned_goal_text = goal_text.replace('\n', ' ').replace('\t' ,' ').strip()
        if skip_goal is not None:
            skip_goal = skip_goal.replace('\n', ' ').replace('\t' ,' ').strip()

        if prompt_mode == 0:
            query_text = goal_text.replace('\n', '\t').strip()
        elif prompt_mode == 1:
            query_text = f"GOAL {cleaned_goal_text} {keyword}"
        elif prompt_mode == 2:
            query_text = f"DEC {dec_name} GOAL {cleaned_goal_text} {keyword}"
        elif prompt_mode == 3:
            query_text = f"DEC {dec_name} {skip_goal} GOAL {cleaned_goal_text} {keyword}"
        elif prompt_mode == 4:
            query_text = f"DEC {dec_name} GOAL {cleaned_goal_text} SKIP"
        elif prompt_mode == 5:
            query_text = f"DEC {dec_name} ERROR None GOAL {cleaned_goal_text} {keyword}"
        elif prompt_mode == 6:
            query_text = f"DEC {dec_name} RESULT SUCCESS ERROR None GOAL {cleaned_goal_text} {keyword}"
        elif prompt_mode == 7:
            query_text = f"RESULT SUCCESS ERROR None GOAL {cleaned_goal_text} {keyword}"
        else:
            raise NotImplementedError(
                """
                `prompt_mode` should be used according to the training of the model and be one of:
                0 for <goal>   (and without replacing `\n`)
                1 for GOAL <goal> PROOFSTEP
                2 for DEC <decl_name> GOAL <goal> PROOFSTEP
                """)
        return query_text

    def eval_states(self, environment_states, data, trajectory=None):
        """
        Evaluate generated tactic using value function
        """
        if self.random_value is True:
            logger.info(f"[VALUE_FUNCTION] Generating value randomly")
            return np.random.uniform(low=0, high=1, size=(len(environment_states),)).tolist()
        tactic_states = [environment_state['tactic_state'] for environment_state in environment_states]
        # root_states = [root_state['tactic_state'] for _ in environment_states]
        decl_name, _= data
        # gpt get tactics text
        if self.value_function_model_type in ['gpt','external_gpt']:
            query_texts = [self.preprocess_prompt(tactic_state, self.prompt_mode, decl_name, True) for tactic_state in
                       tactic_states]
        elif self.value_function_model_type == 'current_state_only':
            query_texts = [ts.replace('\n', ' ').replace('\t' ,' ').strip() for ts in tactic_states]
        elif self.value_function_model_type == "root_state_and_current_state":
            root_state = trajectory[0]
            root_states = [root_state['tactic_state'] for _ in environment_states]
            query_texts = [f"{root}</s></s>{cur}" for root, cur in zip(root_states, tactic_states)]
        elif self.value_function_model_type == "entire_trajectory":
            paths = [trajectory[0]["tactic_state"]]
            for state in trajectory[1:]:
                proof_state, proof_step = state["tactic_state"], state["tactic"]
                paths.extend([proof_step, proof_state])
            query_texts = []
            for environment_state in environment_states:
                path_strs = []
                my_path = copy(paths)
                last_proof_state, proof_step = environment_state["tactic_state"], environment_state["tactic"]
                my_path.append(proof_step)
                path_chunked = [my_path[i:i + 2] for i in range(0, len(my_path), 2)]
                for ps, pstep in path_chunked:
                    ps = ps.replace('\n', ' ').replace('\t' ,' ').strip()
                    pstep = pstep.replace('\n', ' ').replace('\t' ,' ').strip()
                    path_str = f"GOAL {ps} PROOFSTEP {pstep}"
                    path_strs.append(path_str)
                last_proof_state = last_proof_state.replace('\n', ' ').replace('\t' ,' ').strip()
                path_strs.append(f"GOAL {last_proof_state}")
                query_texts.append("</s></s>".join(path_strs))
        elif self.value_function_model_type == "previous_state_and_current_state":
            paths = [trajectory[0]["tactic_state"]]
            for state in trajectory[1:]:
                proof_state, proof_step = state["tactic_state"], state["tactic"]
                paths.extend([proof_step, proof_state])
            query_texts = []
            for environment_state in environment_states:
                path_strs = []
                my_path = copy(paths)
                last_proof_state, proof_step = environment_state["tactic_state"], environment_state["tactic"]
                my_path.append(proof_step)
                path_chunked = [my_path[i:i + 2] for i in range(0, len(my_path), 2)]
                for ps, pstep in path_chunked:
                    ps = ps.replace('\n', ' ').replace('\t' ,' ').strip()
                    pstep = pstep.replace('\n', ' ').replace('\t' ,' ').strip()
                    path_str = f"GOAL {ps} PROOFSTEP {pstep}"
                    path_strs.append(path_str)
                last_proof_state = last_proof_state.replace('\n', ' ').replace('\t' ,' ').strip()
                path_strs.append(f"GOAL {last_proof_state}")
                query_texts.append("</s></s>".join(path_strs[-2:]))
        [logger.info(f'[VALUE_FUNCTION] Query text: {query_text}') for query_text in query_texts]
        return self.tactic_generator.value_function(query_texts)

    def model_generate_skip(self, environment_state, n_samples, data):
        decl_name, _ = data
        tactic_state = environment_state['tactic_state']
        # split goal with '\n', if one tactic state contains multiple goals,
        tactic_state = tactic_state.split('\n')
        # if 'goals' in tactic_state[0]:
        #     tactic_state = tactic_state[1:]
        #     if tactic_state[0].startswith('case'):
        #         tactic_state = tactic_state[1:]

        # TODO: is getting rid of this case is nessary, and will it cause other issue?
        # if tactic_state[0].startswith('case'):
        #     tactic_state = tactic_state[1:]
        tactic_state = ' '.join(tactic_state)

        skip_ts = environment_state['tactic_state'].replace('\n', ' ').replace('\t', ' ')
        skip_prompt = self.preprocess_prompt(skip_ts, 4, decl_name)
        skip_goals = self.tactic_generator.skip_generate(skip_prompt, n_samples, logp=False, deduplicate=False)

        for skip_goal in skip_goals:
            logger.info(f"[SKIP GENERATE] {skip_goal}")

        proofstep_prompts = []
        for skip_goal in skip_goals:
            proofstep_prompts.append(
                self.preprocess_prompt(tactic_state, 3, decl_name, skip_goal=skip_goal))

        if len(proofstep_prompts) > 0:
            result = self.tactic_generator.generate(proofstep_prompts, 1, logp=True)
            result.sort(key=lambda x: x[0], reverse=True)
            return result[:n_samples]
        return []

    def model_generate(self, environment_state, n_samples, data):
        """
        Generate tactic using tactic state
        return list of deduplicated tactics: `list[(logprob:float, tactic:str)]`,
        length = deduplicated list of lenght self.e <= self.e
        """
        decl_name = data[1]
        tactic_state = environment_state['tactic_state']
        # split goal with '\n', if one tactic state contains multiple goals,
        tactic_state = tactic_state.split('\n')
        # if 'goals' in tactic_state[0]:
        #     tactic_state = tactic_state[1:]
        #     if tactic_state[0].startswith('case'):
        #         tactic_state = tactic_state[1:]
        #     # goal = goal[: goal.index('')]   # extract the first goal

        # # TODO: is getting rid of this case is nessary, and will it cause other issue?
        # if tactic_state[0].startswith('case'):
        #     tactic_state = tactic_state[1:]
        tactic_state = ' '.join(tactic_state)
        # gpt get tactics text
        query_text = self.preprocess_prompt(tactic_state, self.prompt_mode, decl_name)
        logger.info(f'[MODEL GENERATE] Query text: {query_text}')

        # list[(logprob:float, tactic:str)], length = deduplicated list of lenght self.e <= self.e
        generation_start_time = time.time()
        result = self.tactic_generator.generate(query_text, n_samples)
        generation_end_time = time.time()
        logger.info(f"[MODEL GENERATE] Generated {n_samples} samples using {generation_end_time - generation_start_time:.2f}s")

        return result

    def formal_system_filter(self, generated_list, environment_state, data, trajectory):
        """
        filter the generated tactics using lean-gym.
        return a list of applicable tactics tree node.
        """
        return_list = []
        failed_actions = []
        tactic_states = set()
        generated_list = sorted(generated_list, key=lambda x: x[0], reverse=True)

        for (logp, tactic) in generated_list:
            logger.info(f"[FORMAL SYSTEM FILTER] Generated tactic - {logp:.4f} - {tactic}")
            result = self.lean_server.run_tac(search_id=self.search_id,
                                              tactic_id=environment_state['tactic_state_id'],
                                              tactic=tactic)  # dict returned by lean-gym

            logger.info(f"[FORMAL SYSTEM FILTER] Lean_result - {result}")
            if result['error'] is not None:
                failed_actions.append((tactic, result['error']))
                continue
            if result['tactic_state'] in tactic_states:
                continue
            tactic_states.add(result['tactic_state'])

            result['tactic'] = tactic
            result['logp'] = logp
            return_list.append((logp, result))
        
        trucation_flag = False
        # use value function to evaluate current states
        if self.use_value_function and len(return_list) > 0:
            logps = [item[0] for item in return_list]
            environment_states = [item[1] for item in return_list]
            scores, trucation_flag = self.eval_states(environment_states, data, trajectory)
            return_list = list(zip(scores, logps, environment_states))
            for score, logp, environment_state in return_list:
                tactic = environment_state['tactic_state'].replace('\n', ' ').replace('\t', ' ')
                logger.info(f"[VALUE FUNCTION RESULT] Generated tactic - {score:.4f} - {logp:.4f} - {tactic}")
        else:
            logps = [item[0] for item in return_list]
            environment_states = [item[1] for item in return_list]
            scores = copy(logps)
            scores = [math.exp(score) for score in scores]
            return_list = list(zip(scores, logps, environment_states))

        return_list = sorted(return_list, key=lambda x: x[0], reverse=True)

        return return_list, failed_actions, trucation_flag

    def __get_goals(self, state):
        '''
        get goal list of the tactic state
        if tactic state has multiple goals, it should be like:
            X goals\n
            goal1\n\n
            goal2\n\n
            ...  \n\n
            goalX
        return [goal1, goal2, ..., goalX] (the head 'X goals' is also removed)
        '''
        goals = state.split("\n\n")
        if len(goals) > 1:
            goals_title = f"{len(goals)} goals\n"
            if goals[0].startswith(goals_title):
                goals[0] = goals[0][len(goals_title):]
        return goals

    def split_environment_states(self, environment_state, data, pre_state, use_value_function=True):
        """
        In Lean, we can not actually split out specific goal,
        but we can change the order of the goal we currently want to proof
        """
        tactic_state = environment_state['tactic_state']
        pre_tactic_state = pre_state['tactic_state']
        # split goal with '\n', if one tactic state contains multiple goals,
        goals = self.__get_goals(tactic_state)
        pre_goals = self.__get_goals(pre_tactic_state)

        # The problem remains, if model predict tactics that applies to all goals,
        # every goal will be changed.
        # assert that all goals does not change except for the first one
        assert all([g in goals for g in pre_goals[1:]])

        # check goal solved
        if len(set(goals) - set(pre_goals)) == 0:
            environment_state['tactic_state'] = 'no goals'
            print('partial solved!!')
            return [1.0], [environment_state]

        if len(goals) > 1:
            result_states = []
            _tactic_id = environment_state['tactic_state_id']
            for goal_id, goal in enumerate(goals):
                if goal in pre_goals:
                    continue
                sub_result = self.lean_server.run_tac(search_id=self.search_id,
                                                      tactic_id=_tactic_id,
                                                      tactic=f"tactic.rotate_left {goal_id}")

                # make sure the result state is the same as the first goal
                assert self.__get_goals(sub_result['tactic_state'])[0] == goal
                sub_result['tactic'] = environment_state['tactic']
                sub_result['logp'] = environment_state['logp']
                result_states.append(sub_result)
        else:
            result_states = [environment_state]

        if use_value_function and len(result_states) > 0:
            # use value function to evaluate states again
            scores = self.eval_states(result_states, data)
            return scores, result_states

        return None, result_states

    def check_success(self, environment_state):
        if environment_state['tactic_state'] == 'no goals':
            return True
        else:
            return False

    def state_str(self, environment_state):
        return environment_state['tactic_state']

    def get_state_id(self, environment_state):
        return environment_state['tactic_state_id']

    def get_record(self, environment_state, action, data):
        record = {
            'decl_name': data,
            'goal': environment_state['tactic_state'].replace('\n', ' '),
            'proofstep': action
        }
        return record

    def get_rlhf_record(self, path, record_state, data):
        if record_state in ["success", "failed"]:
            final_state = path[-1]['tactic_state'].replace('\n', ' ')
            path = path[:-1]
            assert len(path) % 2 == 0
            path_chunked = [path[i:i + 2] for i in range(0, len(path), 2)]
            lines = []
            for state, action in path_chunked:
                state = state['tactic_state'].replace('\n', ' ')
                line = f"GOAL {state} PROOFSTEP {action}"
                lines.append(line)
            lines.append(f"GOAL {final_state}")
            if record_state == "success":
                labels = len(lines)
            elif record_state == "failed":
                labels = -1
            else:
                raise NotImplementedError
            path_str = "\t\t".join(lines)
            record = {
                'decl_name': data,
                'path': path_str,
                'label': labels,
                'error': "",
            }
            return record, path_str
        if record_state == 'error':
            assert len(path) % 2 == 0
            final_action, final_error = path[-1]
            path = path[:-1] + [final_action]
            path_chunked = [path[i:i + 2] for i in range(0, len(path), 2)]
            lines = []
            for state, action in path_chunked:
                state = state['tactic_state'].replace('\n', ' ').strip()
                action = action.replace("\n", " ").strip()
                line = f"GOAL {state} PROOFSTEP {action}"
                lines.append(line)
            labels = -2
            path_str = "\t\t".join(lines)
            record = {
                'decl_name': data,
                'path': path_str,
                'label': labels,
                'error': final_error.replace("\n", ' '),
            }
            return record, path_str

    def process_train_data(self, data):
        return data

    def verbalize_environment_states(self, environment_states):
        print(environment_states)
        if not isinstance(environment_states, list):
            return environment_states['tactic_state'].replace('\t', ' ').replace('\n', ' ')
        else:
            return [state['tactic_state'].replace('\t', ' ').replace('\n', ' ') for state in environment_states]

    def verbalize_environment_steps(self, environment_states):
        if not isinstance(environment_states, list):
            if 'tactic' in environment_states:
                return environment_states['tactic'].replace('\t', ' ').replace('\n', ' ')
            else:
                return ""
        else:
            return [state.get('tactic', "").replace('\t', ' ').replace('\n', ' ') for state in environment_states]

    def clear_environment(self, state):
        self.lean_server.clear_search(search_id=state['search_id'])
