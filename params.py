from dataclasses import dataclass
import random

@dataclass
class MustardArgs:
    baseline_type:str=None
    n_iter:int=None
    qtype:str=None
    qlevel:str=None
    kw_mode:str="kwg"
    num_keyword:int=None    
    preset_keywords:list=None  # list of (concept, domain)
    num_correct:int=None
    

params_random = MustardArgs(
    baseline_type="step",
    n_iter=3,
    qtype=random.choice(["word_problem", "theorem_proving"]),
    qlevel=random.choice([
            "elementary_school",
            "middle_school",
            "high_school",
            "higher_edu"
        ]),
    num_keyword=random.choice([1, 2]),
    num_correct=2,
    preset_keywords=[("Alternate coordinate systems (base)", "Linear algebra"), ]
)

params_custom = MustardArgs(
    baseline_type=None, 
    n_iter=None, 
    qtype=None, 
    qlevel=None, 
    num_keyword=None, 
    num_correct=None,    
)

