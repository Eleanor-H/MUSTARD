from leaven.src.lean_server import LeanEnv
import shelve
import re
from pathlib import Path
import networkx as nx
from typing import Optional, Dict, Any, Tuple, List

PROVING_ENV_PATH = Path(__file__).parent.parent.resolve() / 'proving_env_lib'

class ProvingSearchAgent:
    """
    This class facilitates interaction between Python and Lean 3 for theorem proving tasks.
    It supports three modes of operation: Raw Search, Plain Search, and Sequential Search, 
    each tailored for different scenarios and levels of integration with mathlib.
    """

    def __init__(self) -> None:
        """
        Initializes the ProvingSearchAgent by setting up data structures to keep track of the 
        proving process, but without establishing any connections yet.
        """
        self.search_history: nx.DiGraph = nx.DiGraph()
        self.server: Optional[LeanEnv] = None
        self.proving_envs: Optional[shelve.DbfilenameShelf] = None
        self.init_context = None

    def __enter__(self) -> 'ProvingSearchAgent':
        """
        Facilitates the usage of the ProvingSearchAgent instance in a 'with' context. This
        method returns the instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Ensures that all resources (server connection and shelve file) are properly released when 
        exiting the 'with' context.
        """
        self.close()

    def close(self) -> None:
        """
        Releases all resources held by the ProvingSearchAgent instance, including server 
        connections and shelve file connections.
        """
        if self.server is not None:
            self.server.close()
            self.server = None
        if self.proving_envs is not None:
            self.proving_envs.close()
            self.proving_envs = None

    def open_proving_envs(self) -> None:
        """
        Opens a shelve file named 'proving_env_lib' which contains pre-defined theorem 
        environments. This is necessary for accessing the theorem environments stored 
        in this file.
        """
        self.proving_envs = shelve.open(str(PROVING_ENV_PATH))

    def close_proving_envs(self) -> None:
        """
        Closes the proving environments shelve file, if it's open. This ensures that no 
        resources are wasted.
        """
        if self.proving_envs is not None:
            self.proving_envs.close()
            self.proving_envs = None

    def add_theorem_env(self, name: str, init_context: str, pre_lines: str = '', post_lines: str = '') -> None:
        """
        Stores a theorem environment in the shelve file. This allows the user to add new 
        theorem environments which can be used later in the proving process.
        
        Raises:
            ValueError: If a theorem with the same name already exists in the proving environments.
        """
        if self.proving_envs is None:
            with shelve.open(str(PROVING_ENV_PATH)) as proving_envs:
                if name in proving_envs:
                    raise ValueError(f'{name} already exists in proving environments')
                proving_envs[name] = (pre_lines, init_context, post_lines)
        else:
            if name in self.proving_envs:
                raise ValueError(f'{name} already exists in proving environments')
            self.proving_envs[name] = (pre_lines, init_context, post_lines)

    def get_theorem_env(self, name: str) -> Tuple[str, str, str]:
        """
        Retrieves the theorem environment for a given theorem name from the shelve file. This 
        theorem environment includes any necessary lines of code before and after the theorem 
        itself, as well as the theorem's context.
        """
        if self.proving_envs is None:
            with shelve.open(str(PROVING_ENV_PATH)) as proving_envs:
                return proving_envs[name]
        return self.proving_envs[name]

    def _verify_lean_file(self, context: str, pre_lines: str = '', post_lines: str = '', **kwargs) -> Dict[str, Any]:
        """
        Sends the provided Lean code context to the Lean server for verification. This internal 
        method also keeps track of the verification results in the search history graph.
        
        Returns:
            A dictionary containing the verification results.
        """
        if self.server is None:
            self.server = LeanEnv()
        results = self.server.verify_lean_file(pre_lines + context + post_lines, check_span=(len(pre_lines), len(pre_lines) + len(context)))
        assert pre_lines in results['context'] and post_lines in results['context']
        core_context = results['context'][len(pre_lines):-len(post_lines) if len(post_lines) > 0 else len(results['context'])]
        results.update({'core_context': core_context,
                        'pre_lines': pre_lines,
                        'post_lines': post_lines,
                        **kwargs})
        self.search_history.add_node(results['context'], **{k : v for k, v in results.items() if k != 'context'})
        return results
    
    def get_context_properties(self, pre_lines: str, post_lines: str, last_context: str) -> Tuple[str, str]:
        """
        Retrieves the pre-lines and post-lines associated with a particular tactic state in the 
        search history. This is useful when applying tactics to ensure that the context is 
        correctly structured.
        """
        if pre_lines is None or post_lines is None:
            node_props = self.search_history.nodes[last_context]
            if pre_lines is None:
                pre_lines = node_props['pre_lines']
            if post_lines is None:
                post_lines = node_props['post_lines']
        return pre_lines, post_lines

    def init_search_raw(self, context: Optional[str] = None, filename: Optional[str] = None, pre_lines: str = '', post_lines: str = '', **kwargs: Any) -> Dict[str, Any]:
        """
        Initializes a Raw Search by using a custom Lean code context or reading from a specified 
        file. This search is not dependent on the theorem environments in the shelve file.
        
        Raises:
            ValueError: If both context and filename are not provided or the specified file does not exist.
        """
        if context is None:
            if filename is not None:
                with open(filename) as f:
                    context = f.read()
                    pre_lines = post_lines = ''
            else:
                raise ValueError('unexpected_none_context')
        if self.server is not None:
            self.server.close()
        self.search_history = nx.DiGraph()
        self.server = LeanEnv()
        results = self._verify_lean_file(context=context, pre_lines=pre_lines or '', post_lines=post_lines or '', **kwargs)
        self.init_context = results['context']
        return results
    
    def init_search_plain(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Initializes a Plain Search by using a theorem name from the shelve file. This search 
        directly uses the theorem's environment without any modifications.
        """
        pre_lines, init_context, post_lines = self.get_theorem_env(name)
        pre_lines = '\n'.join([i for i in init_context.split('\n') if i.startswith('import') and i not in pre_lines] + [pre_lines])
        init_context = '\n'.join(i for i in init_context.split('\n') if not i.startswith('import'))
        return self.init_search_raw(context=init_context, pre_lines=pre_lines, post_lines=post_lines, **kwargs)
    
    def init_search_sequential(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Initializes a Sequential Search. This mode is tailored for the lean-gym tactic mode 
        environment. It starts with a theorem from the shelve file and sequentially modifies 
        the proof context.
        """
        pre_lines, init_context, post_lines = self.get_theorem_env(name)
        context = re.sub(r'\bsorry\b', 'begin\n repeat { sorry } \nend', init_context)
        return self.init_search_raw(context=context, pre_lines=pre_lines, post_lines=post_lines, **kwargs)
    
    def run_tac_raw(self, context: str, last_context: str, pre_lines: str = '', post_lines: str = '', **kwargs: Any) -> Dict[str, Any]:
        """
        In Raw Search mode, this method modifies the current Lean context by applying a user-provided 
        tactic. The result of this modification is then sent for verification.
        
        Returns:
            The verification result of the modified context.
        """
        pre_lines, post_lines = self.get_context_properties(pre_lines, post_lines, last_context)
        results = self._verify_lean_file(context=context, pre_lines=pre_lines, post_lines=post_lines, **kwargs)
        results.update(kwargs)
        self.search_history.add_edge(last_context, results['context'], tactic=None)
        return results
    
    def run_tac_plain(self, context: str, last_context: str, **kwargs: Any) -> Dict[str, Any]:
        """
        In Plain Search mode, this method modifies the current Lean context by applying a user-provided 
        tactic. Unlike Raw Search, this method handles the importing of necessary modules.
        
        Returns:
            The verification result of the modified context.
        """
        node_props = self.search_history.nodes[last_context]
        pre_lines, post_lines = node_props['pre_lines'], node_props['post_lines']
        pre_lines = '\n'.join([i for i in context.split('\n') if i.startswith('import') and i not in pre_lines] + [pre_lines])
        context = '\n'.join(i for i in context.split('\n') if not i.startswith('import'))
        return self.run_tac_raw(context=context, last_context=last_context, pre_lines=pre_lines, post_lines=post_lines, **kwargs)

    def run_tac_sequential(self, tactic: str, last_context: str, **kwargs: Any) -> Dict[str, Any]:
        """
        In Sequential Search mode, applies a specified tactic to the current context. The modified 
        context is then sent for verification, allowing users to build proofs step-by-step.
        
        Returns:
            The verification result of the modified context.
        """
        node_props = self.search_history.nodes[last_context]
        pre_lines, context_to_expand, post_lines = node_props['pre_lines'], node_props['core_context'], node_props['post_lines']
        sorry_pos = [i.span() for i in re.finditer(r'\brepeat { sorry }', context_to_expand)]
        assert len(sorry_pos) == 1
        context = context_to_expand[:sorry_pos[0][0]] + tactic.rstrip(', ') + ',  repeat { sorry } ' + context_to_expand[sorry_pos[0][1]:]
        return self.run_tac_raw(context=context, last_context=last_context, pre_lines=pre_lines, post_lines=post_lines, **kwargs)
    
    def get_path_to_context(self, context: str) -> List[int]:
        """
        Returns a sequence of node identifiers, representing the path from the root of the search 
        history graph to a given context.
        """
        path = nx.shortest_path(self.search_history, source=self.init_context, target=context)
        return [self.search_history.nodes[p] for p in path]
    
    def get_context_to_expand(self):
        pass

def add_mathlib_data():
    from tqdm import tqdm
    agent = ProvingSearchAgent()
    agent.open_proving_envs()
    from pathlib import Path
    import json
    for p in tqdm(list((Path(__file__).resolve().parent.parent.parent / 'dataset_temp').glob('**/*.json'))):
        with open(p) as f:
            data = json.load(f)
        for k, v in data.items():
            agent.add_theorem_env(name=k, init_context=v['init_context'], pre_lines=v['prelines'], post_lines=v['postlines'])
    agent.close()

def add_minif2f_data():
    agent = ProvingSearchAgent()
    agent.open_proving_envs()
    from pathlib import Path
    from tqdm import tqdm
    import json
    with open(Path(__file__).resolve().parent.parent.parent / 'minif2f_import.lean') as f:
        pre_lines = f.read()
    with open(Path(__file__).resolve().parent.parent.parent / 'minif2f.json') as f:
        data = json.load(f)
    for k, v in tqdm(list(data.items())):
        agent.add_theorem_env(name=k, init_context=v["formal_statement"], pre_lines=pre_lines)
    agent.close()
