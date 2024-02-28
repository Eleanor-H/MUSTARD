import random
import shutil
import time
from copy import copy
from enum import Enum
import math
from queue import PriorityQueue, Queue

import logging
from itertools import count
import treelib

from environment_base import FormalSystemEnvironment
# from src.environment.formal_system_client import TimeoutException, time_limit
from utils import append_to_csv, get_bucket, append_to_jsonl
import os
import multiprocessing
from utils import print
# from profilehooks import profile

logger = logging.getLogger("eval_search")
unique = count()


class SearchReturnState(Enum):
    SUCCESS = 0
    EMPTY_QUEUE = 1
    GLOBAL_TIMEOUT = 2
    MAX_STEP_REACHED = 3
    INIT_FAILED = 4
    SYSTEM_ERROR = 5


class TreeNode:
    """
    Graph node for proof graph

    Tree node for Best first search tree
    the `state` stores proof state
    """

    def __init__(self, state, children, parents, actions=None, priority=0.0):
        # dict: tactic state returned by lean-gym
        self.state = state
        self.depth = 0
        self.individual_priority = 0
        self.priority = priority  # logprob computed by the LM
        self.children = children if children is not None else []
        self.parent = parents if parents is not None else []
        self.actions = actions if actions is not None else []  # actions that create current node
        self.expanded = False
        self.failed_actions = []

    def __repr__(self) -> str:
        return str(self.state)


class BestFirstSearch():
    """
    Best first search algorithm from gpt-f
    """

    def __init__(self,
                 rank,
                 environment: FormalSystemEnvironment,
                 data=None,
                 n_expansion=8,
                 max_search_steps=512,
                 n_retry=0,
                 n_success_proof=1,
                 max_proof_steps=30,
                 global_timeout=100,
                 step_timeout=100,
                 expert_iter_path=None,
                 use_skip=False,
                 accumulate_log_prob=True,
                 create_gt_rlhf_data=False,
                 ) -> None:
        self.rank = rank
        self.environment = environment
        self.data = data
        self.n_expansion = n_expansion
        self.max_search_steps = max_search_steps
        self.n_retry = n_retry
        self.max_proof_steps = max_proof_steps
        self.global_timeout = global_timeout
        self.step_timeout = step_timeout
        self.use_skip = use_skip
        self.n_success_proof = n_success_proof
        self.accumulate_log_prob = accumulate_log_prob
        self.truncation_flag = False

        self.search_queue = PriorityQueue()
        self.environment_state_history = {}
        self.proof_tree = None
        self.success_nodes = []
        self.create_gt_rlhf_data = create_gt_rlhf_data

        environment_state = self.environment.init_search(self.data)
        if environment_state is not None:
            init_node = TreeNode(environment_state, None, None, None, 0.0)
            self.proof_tree = init_node
            self.search_queue.put((0, next(unique), init_node))
            self.environment_state_history[self.environment.verbalize_environment_states(environment_state)] = init_node

        self.expert_iter_path = expert_iter_path

    def generate_rlhf_data(self, iteration_id=None, eval_batch_id=None):
        """
        generate RL data
        The data is formatted as list of trajectories
        the data will be written in jsonlines format
        {
            "init_data" : [decl_name, namespaces]
            "trajectories" : [
                {
                    "tactic_state": tactic_state
                    "after_tactic_state" tactic_state
                    "tactic": tactic
                    "log_prob": log_prob
                    "value": value
                    "reward": reward
                }, ...
            ]
        }

        """
        create_data_start_time = time.time()



        # init search failed
        if self.proof_tree is None:
            return None

        # add paths that success
        success_proof_path = []
        for success_node in self.success_nodes:
            success_proof_path.extend(self.get_proof_paths_to_node(success_node))

        # add paths that failed
        failed_proof_path = []
        q = Queue()
        q.put(self.proof_tree)
        while not q.empty():
            node = q.get()
            if self.environment.check_success(node.state) is False and not node.expanded:
                continue
            has_expanded_or_success_child = False
            for child in node.children:
                if child.expanded or self.environment.check_success(child.state):
                    has_expanded_or_success_child = True
                    break
            if has_expanded_or_success_child:
                for child in node.children:
                    q.put(child)
                continue

            # leaf expanded node
            if self.environment.check_success(node.state) is True:
                continue
            failed_proof_path.extend(self.get_proof_paths_to_node(node))

        # add path that cause formal system error
        error_proof_path = []
        q = Queue()
        q.put(self.proof_tree)
        while not q.empty():
            node = q.get()
            if not node.expanded:
                continue
            for child in node.children:
                q.put(child)
            proof_paths = self.get_proof_paths_to_node(node)
            new_proof_paths = []
            for path in proof_paths:
                for action in node.failed_actions:
                    new_path = copy(path)
                    # print("failed action: ",action)
                    new_path.append(action)
                    new_proof_paths.append(new_path)
            error_proof_path.extend(new_proof_paths)

        output_data = []
        path_set = set()
        all_proof_paths = {
            "success": success_proof_path,
            "failed": random.sample(failed_proof_path, min(20, len(failed_proof_path))),
            "error": random.sample(error_proof_path, min(20, len(error_proof_path))),
        }
        for finish_status, proof_paths in all_proof_paths.items():
            for path in proof_paths:
                if len(path) <= 1:
                    continue
                _, path_str = self.environment.get_rlhf_record(path, finish_status, self.data)
                if path_str in path_set:
                    continue
                path_set.add(path_str)
                trajectories = []
                endpoint = len(path)-1 if finish_status == "error" else len(path)-2
                for ix in range(0, endpoint, 2):
                    cur_state = self.environment.verbalize_environment_states(path[ix])
                    tactic = path[ix+1]
                    if finish_status == "error" and isinstance(tactic, tuple):
                        tactic, next_state = tactic[0], tactic[1]
                        next_state = "<ERROR> " + next_state.replace("\n", " ").strip()
                    else:
                        next_state = self.environment.verbalize_environment_states(path[ix+2])
                    if ix - 1 > 0:
                        context = path[ix-1]
                    else:
                        context = self.data[1].replace('\n', " ").strip()
                    assert "<ERROR>" not in cur_state
                    input_text = self.environment.preprocess_prompt(
                        goal_text=cur_state,
                        proof_context=context,
                        prompt_mode=-1,
                        is_value_function=False,
                        action_text=tactic,
                        result_text=next_state
                    )
                    # value, logprob = self.environment.get_value_and_logprob(input_text, "PROOFSTEP", " RESULT")
                    # value, logprob = value[0], logprob[0]
                    if next_state == "no goals":
                        reward = 1
                    elif "<ERROR>" in next_state:
                        reward = -1
                    else:
                        reward = 0
                    element = {
                        "input_text": input_text,
                        "tactic_state": cur_state,
                        "after_tactic_state": next_state,
                        "tactic": tactic,
                        "log_prob": None,
                        "value": None,
                        "reward": reward
                    }
                    assert "<ERROR>" not in element["tactic_state"]
                    trajectories.append(element)
                output_data.append({
                    "init_data": self.data,
                    "trajectories": trajectories,
                    "source": "bfs"
                })
        
        all_text = []
        for element in output_data:
            for step in element["trajectories"]:
                all_text.append(step["input_text"])
            
        logger.info(f"Calculating {len(all_text)} logprob and values...")
        batch_size = 4
        all_values, all_logprobs = [], []
        for ix in range(len(all_text) // batch_size + 1):
            batch_text = all_text[ix * batch_size: (ix+1) * batch_size]
            if len(batch_text) == 0: 
                continue
            value, logprob = self.environment.get_value_and_logprob(batch_text, "PROOFSTEP", "<|endoftext|>")
            all_values.extend(value)
            all_logprobs.extend(logprob)

        ix = 0
        for element in output_data:
            for step in element["trajectories"]:
                step["log_prob"] = all_logprobs[ix]
                step["value"] = all_values[ix]
                ix += 1

        # add ground truth data to rlhf data
        if self.create_gt_rlhf_data:
            gt_rlhf_data = self.environment.generate_rlhf_data_from_ground_truth(self.data)
            if gt_rlhf_data is not None:
                output_data.append(gt_rlhf_data)

        # filter outline data
        step_filtered = 0
        filtered_output_data = []
        for element in output_data:
            filtered_element = copy(element)
            filtered_element["trajectories"] = []
            for step in element["trajectories"]:
                if step["log_prob"] > -100:
                    filtered_element["trajectories"].append(step)
                else:
                    step_filtered += 1
                    logger.info(f'[RLHF DATA] Filtered step: logprob={step["log_prob"]}, step_context={step["input_text"]}')
            if len(filtered_element["trajectories"]) > 0:
                filtered_output_data.append(filtered_element)

        if eval_batch_id is not None and iteration_id is not None:
            fname = f"rlhf_data_rank{self.rank}_iteration{iteration_id}_batch{eval_batch_id}.jsonl"
            fname = os.path.join(self.expert_iter_path, fname)
        else:
            fname = f'{self.expert_iter_path}{self.rank}_rlhf.jsonl'
        append_to_jsonl(filtered_output_data, fname)

        create_data_time = time.time() - create_data_start_time
        logger.info(f"[GET RL DATA] gather data with {create_data_time}s, number of data filtered: {step_filtered}")

        # path_set = set()
        # results = []
        # for path in success_proof_path:
        #     rlhf_record, path_str = self.environment.get_rlhf_record(path, "success", self.data)
        #     if path_str not in path_set:
        #         results.append(rlhf_record)
        #         path_set.add(path_str)
        # for path in failed_proof_path:
        #     if len(path) <= 1:
        #         continue
        #     rlhf_record, path_str = self.environment.get_rlhf_record(path, "failed", self.data)
        #     if path_str not in path_set:
        #         results.append(rlhf_record)
        #         path_set.add(path_str)
        # for path in error_proof_path:
        #     if len(path) <= 1:
        #         continue
        #     rlhf_record, path_str = self.environment.get_rlhf_record(path, "error", self.data)
        #     if path_str not in path_set:
        #         results.append(rlhf_record)
        #         path_set.add(path_str)

        # fname = f'{self.expert_iter_path}{self.rank}_rlhf.csv'
        # append_to_csv(results, fname)

    def generate_expert_iteration_data(self):
        """
        generate PROOFSIZE and PROOFSTEP objective data. use for expert iteration.
        """

        all_proof_path = []
        for success_node in self.success_nodes:
            all_proof_path.extend(self.get_proof_paths_to_node(success_node))

        proofsize_records = []
        proofstep_records = []

        # -- step 1: record every successful proofs
        success_states = set()
        for path in all_proof_path:
            path = path[:-1]
            assert len(path) % 2 == 0
            path_chunked = [path[i:i + 2] for i in range(0, len(path), 2)]
            path_chunked.reverse()
            for idx, (state, action) in enumerate(path_chunked):
                record = self.environment.get_record(state, action, self.data)
                record['proofsize'] = idx + 1
                proofsize_records.append(record)
                proofstep_records.append(record)
                success_states.add(self.environment.verbalize_environment_states(state))

        # -- step 2: level order traversal, add failed node with `inf` proofsize
        q = Queue()
        q.put(self.proof_tree)
        while not q.empty():
            node = q.get()
            if self.environment.verbalize_environment_states(node.state) not in success_states and \
                    node.expanded is True:
                record = self.environment.get_record(node.state, '', self.data)
                record['proofsize'] = -1
                proofsize_records.append(record)
            for child in node.children:
                q.put(child)

        # -- step 3: write results to the file
        fname = f'{self.expert_iter_path}{self.rank}_proofsize.csv'
        append_to_csv(proofsize_records, fname)

        fname = f'{self.expert_iter_path}{self.rank}_proofstep.csv'
        append_to_csv(proofstep_records, fname)

    def get_proof_steps(self, current_node: TreeNode):
        """
        Traverse tree node backward and get all history proof steps
        """
        paths = self.get_proof_paths_to_node(current_node)
        result = []
        for path in paths:
            path = path[1::2]
            result.append(path)
        return result

    def get_past_proof_state(self, current_node: TreeNode, return_node=False):
        """
        In new settings, a node allow multiple parents, like DAG
        This method retrieve all the parents for current node
        """
        paths = self.get_proof_paths_to_node(current_node, return_node)
        past_proof_states = []
        for path in paths:
            path = path[0::2]
            past_proof_states.extend(path)
        return past_proof_states

    def get_proof_paths_to_node(self, target_node, return_node=False):
        """
        return all possible proof path from root to node.
        """
        # print("In get proof paths to node:")
        result_list = []
        node_queue = Queue()
        if return_node:
            node_queue.put((target_node, [target_node]))
        else:
            node_queue.put((target_node, [target_node.state]))
        while not node_queue.empty():
            # print("queue length:", node_queue.qsize())
            node, path = node_queue.get()
            # print(node.state)
            # print(node.parent)
            # print(node.actions)
            if node == self.proof_tree:
                result_path = copy(path)
                result_path.reverse()
                result_list.append(result_path)
            assert len(node.parent) == len(node.actions)
            # print("add length:", len(node.parent))
            for parent, action in zip(node.parent, node.actions):
                new_path = copy(path)
                if return_node:
                    new_path.extend([action, parent])
                else:
                    new_path.extend([action, parent.state])
                node_queue.put((parent, new_path))
        return result_list

    def get_on_track_rate(self):
        if len(self.success_nodes) == 0:
            return -1.0

        on_track_states = []
        for node in self.success_nodes:
            on_track_states.extend(self.get_past_proof_state(node))
        on_track_states_ids = [self.environment.get_state_id(state) for state in on_track_states]

        n_total_expanded_nodes = 0
        n_on_track_nodes = 0
        node_queue = Queue()
        node_queue.put(self.proof_tree)
        while not node_queue.empty():
            node = node_queue.get()
            if node.expanded:
                n_total_expanded_nodes += 1
            if self.environment.get_state_id(node.state) in on_track_states_ids and \
                    not self.environment.check_success(node.state):
                try:
                    assert node.expanded
                except AssertionError:
                    continue
                n_on_track_nodes += 1
            for child in node.children:
                node_queue.put(child)
        return n_on_track_nodes / n_total_expanded_nodes

    def get_return_state(self, state):
        tactics = []
        on_track_rate = self.get_on_track_rate()
        if len(self.success_nodes) >= 1:
            state = SearchReturnState.SUCCESS
            for node in self.success_nodes:
                tactics.append(self.get_proof_steps(node))

        return_state = {"success": True if state == SearchReturnState.SUCCESS else False,
                        "empty_queue": True if state == SearchReturnState.EMPTY_QUEUE else False,
                        "max_step_reached": True if state == SearchReturnState.MAX_STEP_REACHED else False,
                        "global_timeout": True if state == SearchReturnState.GLOBAL_TIMEOUT else False,
                        'init_failed': True if state == SearchReturnState.INIT_FAILED else False,
                        "tactics": tactics,
                        "on_track_rate": on_track_rate,
                        'decl_nms': self.data}

        # this should return a dictionary containing all the proof search log.
        proof_log = self.proof_log_analysis()
        if state == SearchReturnState.SUCCESS:
            self.generate_expert_iteration_data()
        # self.generate_rlhf_data()

        return state, return_state, proof_log

    def __search(self, node: TreeNode, n_nodes=0):
        """
        Main method for BFS search
        """
        # print proof search tree
        self.print_current_tree()

        # log search states
        logger.info(f"[SEARCHING GOAL] {self.environment.verbalize_environment_states(node.state)}")
        logger.info(f"[PREVIOUS STEPS] {self.get_proof_steps(node)}")

        # return search
        if n_nodes > self.max_search_steps:
            return self.get_return_state(SearchReturnState.MAX_STEP_REACHED)
        if node.depth > self.max_proof_steps:
            logger.info("[SEARCH] Maximum proof steps achieved. Skipping this node")
            if not self.search_queue.empty():
                _, _, next_node = self.search_queue.get()
                return self.__search(next_node, n_nodes=n_nodes + 1)
            else:
                return self.get_return_state(SearchReturnState.EMPTY_QUEUE)

        assert self.environment.check_success(node.state) is not True
        node.expanded = True

        # generate tactics
        for _ in range(self.n_retry + 1):
            if self.use_skip is True:
                generated_list = self.environment.model_generate_skip(node.state, self.n_expansion, self.data)
            else:
                generated_list = self.environment.model_generate(node.state, self.n_expansion, self.data)
            verified_list, failed_actions, truncation_flag = self.environment.formal_system_filter(generated_list, node.state, self.data,
                                                                                  self.proof_tree.state)
            node.failed_actions = failed_actions
            if len(verified_list) > 0:
                break
        
        if truncation_flag:
            self.truncation_flag = True
        
        # generate nodes
        child_nodes = []
        past_proof_states_and_action = self.get_past_proof_state(node)
        past_proof_states_and_action = [
            f"{self.environment.verbalize_environment_steps(envstate)} -> {self.environment.verbalize_environment_states(envstate)}" 
            for envstate in past_proof_states_and_action
        ]
        # past_proof_states_and_action = set(self.environment.verbalize_environment_states(self.get_past_proof_state(node)))
        assert f"{self.environment.verbalize_environment_steps(node.state)} -> {self.environment.verbalize_environment_states(node.state)}" in past_proof_states_and_action
        for score, _, environment_state in verified_list:
            action = self.environment.verbalize_environment_steps(environment_state)
            state = self.environment.verbalize_environment_states(environment_state)
            current_state_and_action = f"{action} -> {state}"
            # if the tactic state occurred in parent nodes
            if current_state_and_action in past_proof_states_and_action:
                continue

            # if the state exist in other proof path
            if self.environment.check_success(environment_state) is not True and \
                    current_state_and_action in self.environment_state_history:
                duplicated_node = \
                    self.environment_state_history[current_state_and_action]
                duplicated_node.parent.append(node)
                duplicated_node.actions.append(action)

                # TODO: do we need to add child to parent node
                continue

            # we convert score (probability that this goal is solvable) into log probability,
            # so they can be added
            if self.accumulate_log_prob:
                cum_p_score = node.priority + math.log(score)
            else:
                cum_p_score = math.log(score)
            new_node = TreeNode(environment_state, None, [node], [action], cum_p_score)
            new_node.depth = node.depth + 1
            new_node.individual_priority = math.log(score)

            # add new node to state history
            self.environment_state_history[current_state_and_action] = new_node
            child_nodes.append(new_node)
        node.children = child_nodes

        for child_node in child_nodes:
            if self.environment.check_success(child_node.state) is True:
                self.success_nodes.append(child_node)
                if len(self.success_nodes) >= self.n_success_proof:
                    return self.get_return_state(SearchReturnState.SUCCESS)
            else:
                self.search_queue.put((-child_node.priority, next(unique), child_node))

        if not self.search_queue.empty():
            _, _, next_node = self.search_queue.get()
            return self.__search(next_node, n_nodes=n_nodes + 1)
        else:
            return self.get_return_state(SearchReturnState.EMPTY_QUEUE)

    def search(self):
        """
        Begin best first proof search
        """
        if self.proof_tree is not None:
            _, _, first_node = self.search_queue.get()
            return self.__search(first_node)
            # try:
            #     with time_limit(self.global_timeout):
            #         return self.__search(first_node)
            # except TimeoutException:
            #     return (self.get_return_state(None, SearchReturnState.GLOBAL_TIMEOUT), SearchReturnState.GLOBAL_TIMEOUT)
        else:
            return self.get_return_state(SearchReturnState.INIT_FAILED)

    def on_search_end(self):
        if self.proof_tree is not None:
            self.environment.clear_environment(self.proof_tree.state)

    def print_current_tree(self):
        # level order traversal
        self.tree = treelib.Tree()
        self.cnt = 0

        def create_tree(node, parent):
            if node is None:
                return
            info = {'logp': node.priority,
                    'action': self.environment.verbalize_environment_steps(node.state) if parent is not None else None,
                    "state": self.environment.verbalize_environment_states(node.state),
                    'cnt': self.cnt}
            self.cnt += 1
            self.tree.create_node(str(info), info['cnt'], parent=parent)
            for child in node.children:
                create_tree(child, info['cnt'])

        create_tree(self.proof_tree, None)
        logger.info("===========** PRETTY PROOF TREE **===========")
        self.tree.save2file(f'temp_{self.rank}.txt')
        with open(f'temp_{self.rank}.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
        open(f'temp_{self.rank}.txt', 'w').close()
        for line in lines:
            logger.info(line[:-1])
        logger.info("===========** PRETTY PROOF TREE **===========")

    def proof_log_analysis(self):
        # if init search failed:
        if self.proof_tree is None:
            return None
        # level order traversal
        self.tree = treelib.Tree()
        self.cnt = 0

        def create_tree(node, parent):
            if node is None:
                return
            info = {'logp': node.priority,
                    "state": node.state,
                    'cnt': self.cnt}
            self.cnt += 1
            self.tree.create_node(str(info), info['cnt'], parent=parent)
            for child in node.children:
                create_tree(child, info['cnt'])

        create_tree(self.proof_tree, None)
        proof_tree = self.tree.to_dict()
        log_output = {
            'data': self.data,
            'proof_tree': proof_tree,
        }
        return log_output
