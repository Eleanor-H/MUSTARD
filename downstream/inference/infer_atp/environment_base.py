from abc import ABC, abstractmethod
import logging

logger = logging.getLogger('eval_search')

class FormalSystemEnvironment(ABC):
    """
    Abstract class for formal mathmatic system (Lean, Metamath, ...) operations.
    """
    @abstractmethod
    def init_search(self, init_args):
        pass

    @abstractmethod
    def eval_states(self, environment_state, data, root_state):
        pass

    @abstractmethod
    def model_generate(self, environment_state, n_samples, data):
        pass

    @abstractmethod
    def model_generate_skip(self, environment_state, n_samples, data):
        pass

    @abstractmethod
    def formal_system_filter(self, generated_list, environment_state, data, trajectory):
        pass

    @abstractmethod
    def split_environment_states(self, environment_state, data, pre_state):
        pass

    @abstractmethod
    def check_success(self, environment_state):
        pass

    @abstractmethod
    def state_str(self, environment_state):
        pass

    @abstractmethod
    def get_record(self, environment_state, child_environment_state, data):
        pass

    def get_rlhf_record(self, path, state, data):
        pass

    @abstractmethod
    def process_train_data(self, data):
        return data

    @abstractmethod
    def initialized_environment(self):
        pass

    @abstractmethod
    def setup_data(self):
        pass

    @abstractmethod
    def verbalize_environment_states(self, environment_states):
        pass

    @abstractmethod
    def get_state_id(self, environment_state):
        pass

    @abstractmethod
    def verbalize_environment_steps(self, environment_states):
        pass

    @abstractmethod
    def reset_formal_system(self):
        pass

    @abstractmethod
    def clear_environment(self, state):
        pass

    def get_value_and_logprob(self, input_text, input_end_token, output_start_token):  
        pass

    def preprocess_prompt(self):
        pass

    def generate_rlhf_data_from_ground_truth(self, init_data):
        pass