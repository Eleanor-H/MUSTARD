# Author: Yinya Huang
# Date: 18/7/2023
#


import os
import json
from tqdm import tqdm, trange
import random
from dataclasses import dataclass
import datetime
from func_timeout import func_set_timeout


from openai_key import *
import openai
from tenacity import (
    retry,
    wait_fixed,
    stop_after_attempt,
    wait_random_exponential,
)  

from util_parsing import * 
from leaven.src.lean_server import LeanEnv


@dataclass(frozen=True)
class ConceptLibrary:
    """
    https://www.khanacademy.org/math
    """

    element_concepts = {
        "1st_grade": {
            "Place value",
            "Addition and subtraction",
            "Measurement, data, and geometry",
        },
        
        "2nd_grade": {
            "Add and subtract within 20",
            "Place value",
            "Add and subtract within 100",
            "Add and subtract within 1,000",
            "Money and time",
            "Measurement",
            "Data",
            "Geometry",
        },
        
        "3rd_grade": {
            "Intro to multiplication",
            "1-digit multiplication",
            "Addition, subtraction, and estimation",
            "Intro to division",
            "Understand fractions",
            "Equivalent fractions and comparing fractions",
            "More with multiplication and division",
            "Arithmetic patterns and problem solving",
            "Quadrilaterals",
            "Area",
            "Perimeter",
            "Time",
            "Measurement",
            "Represent and interpret data",
        },
        
        "4th_grade": {
            "Place value",
            "Addition, subtraction, and estimation",
            "Multiply by 1-digit numbers",
            "Multiply by 2-digit numbers",
            "Division",
            "Factors, multiples and patterns",
            "Equivalent fractions and comparing fractions",
            "Add and subtract fractions",
            "Multiply fractions",
            "Understand decimals",
            "Plane figures",
            "Measuring angles",
            "Area and perimeter",
            "Units of measurement",
        },

        "5th_grade": {
            "Decimal place value",
            "Add decimals",
            "Subtract decimals",
            "Add and subtract fractions",
            "Multi-digit multiplication and division",
            "Multiply fractions",
            "Divide fractions",
            "Multiply decimals",
            "Divide decimals",
            "Powers of ten",
            "Volume",
            "Coordinate plane", 
            "Algebraic thinking",
            "Converting units of measure",
            "Line plots",
            "Properties of shapes",
        },
        
        "6th_grade": {
            "Ratios",
            "Arithmetic with rational numbers",
            "Rates and percentages",
            "Exponents and order of operations",
            "Negative numbers",
            "Variables & expressions",
            "Equations & inequalities",
            "Plane figures",
        }
    }

    middle_concepts = {
        "7th_grade": {
            "Negative numbers: addition and subtraction",
            "Negative numbers: multiplication and division",
            "Fractions, decimals, & percentages",
            "Rates & proportional relationships",
            "Expressions, equations, & inequalities",
            "Geometry",
            "Statistics and probability",
        },
        
        "8th_grade": {
            "Numbers and operations",
            "Solving equations with one unknown",
            "Linear equations and functions",
            "Systems of equations",
            "Geometry",
            "Geometric transformations",
            "Data and modeling",
        },
        
        "Algebra_basics": {
            "Foundations",
            "Algebraic expressions",
            "Linear equations and inequalities",
            "Graphing lines and slope",
            "Systems of equations",
            "Expressions with exponents",
            "Quadratics and polynomials",
            "Equations and geometry",
        },
        
        "Pre-algebra": {
            "Factors and multiples",
            "Patterns",
            "Ratios and rates",
            "Percentages",
            "Exponents intro and order of operations",
            "Variables & expressions",
            "Equations & inequalities introduction",
            "Percent & rational number word problems",
            "Proportional relationships",
            "One-step and two-step equations & inequalities",
            "Roots, exponents, & scientific notation",
            "Multi-step equations",
            "Two-variable equations",
            "Functions and linear models",
            "Systems of equations",
        },
        
        "Basic geometry and measurement": {
            "Intro to area and perimeter",
            "Intro to mass and volume",
            "Measuring angles",
            "Plane figures",
            "Units of measurement",
            "Volume",
            "Coordinate plane",
            "Decomposing to find area",
            "3D figures",
            "Circles, cylinders, cones, and spheres",
            "Angle relationships",
            "Scale",
            "Triangle side lengths",
            "Geometric transformations",
        },
    }

    high_concepts = {
        "Algebra_1": {
            "Algebra foundations",
            "Solving equations & inequalities",
            "Working with units",
            "Linear equations & graphs",
            "Forms of linear equations",
            "Systems of equations",
            "Inequalities (systems & graphs)",
            "Functions",
            "Sequences",
            "Absolute value & piecewise functions",
            "Exponents & radicals",
            "Exponential growth & decay",
            "Quadratics: Multiplying & factoring",
            "Quadratic functions & equations",
            "Irrational numbers",
            "Creativity in algebra",
        },

        "Algebra_2": {
            "Polynomial arithmetic",
            "Complex numbers",
            "Polynomial factorization",
            "Polynomial division",
            "Polynomial graphs",
            "Rational exponents and radicals",
            "Exponential models",
            "Logarithms",
            "Transformations of functions",
            "Equations",
            "Trigonometry",
            "Modeling",
        },

        "High_school_geometry": {
            "Performing transformations",
            "Transformation properties and proofs",
            "Congruence",
            "Similarity",
            "Right triangles & trigonometry",
            "Analytic geometry",
            "Conic sections",
            "Circles",
            "Solid geometry",
        },

        "Trigonometry": {
            "Right triangles & trigonometry",
            "Trigonometric functions",
            "Non-right triangles & trigonometry",
            "Trigonometric equations and identities",
        },

        "Statistics_and_probability": {
            "Analyzing categorical data",
            "Displaying and comparing quantitative data",
            "Summarizing quantitative data",
            "Modeling data distributions",
            "Exploring bivariate numerical data",
            "Study design",
            "Probability",
            "Counting, permutations, and combinations",
            "Random variables",
            "Sampling distributions",
            "Confidence intervals",
            "Significance tests (hypothesis testing)",
            "Two-sample inference for the difference between groups",
            "Inference for categorical data (chi-square tests)",
            "Advanced regression (inference and transforming)",
            "Analysis of variance (ANOVA)",
        },

        "High_school_statistics": {
            "Displaying a single quantitative variable",
            "Analyzing a single quantitative variable",
            "Two-way tables",
            "Scatterplots",
            "Study design",
            "Probability",
            "Probability distributions & expected value",
        },

        "Precalculus": {
            "Composite and inverse functions",
            "Trigonometry",
            "Complex numbers",
            "Rational functions",
            "Conic sections",
            "Vectors",
            "Matrices",
            "Probability and combinatorics",
            "Series",
            "Limits and continuity",
        },

        "Calculus_1": {
            "Limits and continuity",
            "Derivatives: definition and basic rules",
            "Derivatives: chain rule and other advanced topics",
            "Applications of derivatives",
            "Analyzing functions",
            "Integrals",
            "Differential equations",
            "Applications of integrals",
        },

        "Calculus_2": {
            "Integrals review",
            "Integration techniques",
            "Differential equations",
            "Applications of integrals",
            "Parametric equations, polar coordinates, and vector-valued functions",
            "Series",
        },
    }

    higher_concepts = {
        "AP_College_Statistics": {
            "Exploring categorical data",
            "Exploring one-variable quantitative data: Displaying and describing",
            "Exploring one-variable quantitative data: Summary statistics",
            "Exploring one-variable quantitative data: Percentiles, z-scores, and the normal distribution",
            "Exploring two-variable quantitative data",
            "Collecting data",
            "Probability",
            "Random variables and probability distributions",
            "Sampling distributions",
            "Inference for categorical data: Proportions",
            "Inference for quantitative data: Means",
            "Inference for categorical data: Chi-square",
            "Inference for quantitative data: slopes",
            "Prepare for the 2022 AP Statistics Exam",
        },

        "College_Algebra": {
            "Linear equations and inequalities",
            "Graphs and forms of linear equations",
            "Functions",
            "Quadratics: Multiplying and factoring",
            "Quadratic functions and equations",
            "Complex numbers",
            "Exponents and radicals",
            "Rational expressions and equations",
            "Relating algebra and geometry",
            "Polynomial arithmetic",
            "Advanced function types",
            "Transformations of functions",
            "Rational exponents and radicals",
            "Logarithms",
        },

        "Differential_Calculus": {
            "Limits and continuity",
            "Derivatives: definition and basic rules",
            "Derivatives: chain rule and other advanced topics",
            "Applications of derivatives",
            "Analyzing functions",
            "Parametric equations, polar coordinates, and vector-va",
        },

        "Integral_Calculus": {
            "Integrals",
            "Differential equations",
            "Applications of integrals",
            "Parametric equations, polar coordinates, and vector-valued functions",
            "Series",
        },

        "AP_College_Calculus_AB": {
            "Limits and continuity",
            "Differentiation: definition and basic derivative rules",
            "Differentiation: composite, implicit, and inverse functions",
            "Contextual applications of differentiation",
            "Applying derivatives to analyze functions",
            "Integration and accumulation of change",
            "Differential equations",
            "Applications of integration",
            "AP Calculus AB solved free response questions from past exams",
            "AP Calculus AB Standards mappings",
        },

        "AP_College_Calculus_BC": {
            "Limits and continuity",
            "Differentiation: definition and basic derivative rules",
            "Differentiation: composite, implicit, and inverse functions",
            "Contextual applications of differentiation",
            "Applying derivatives to analyze functions",
            "Integration and accumulation of change",
            "Differential equations",
            "Applications of integration",
            "Parametric equations, polar coordinates, and vector-valued functions",
            "Infinite sequences and series",
            "AP Calculus BC solved exams",
            "AP Calculus BC Standards mappings",
        },

        "Multivariable_calculus": {
            "Thinking about multivariable functions",
            "Derivatives of multivariable functions",
            "Applications of multivariable derivatives",
            "Integrating multivariable functions",
            "Green's, Stokes', and the divergence theorems",
        },

        "Differential_equations": {
            "First order differential equations",
            "Second order linear equations",
            "Laplace transform",
        },

        "Linear_algebra": {
            "Vectors and spaces",
            "Matrix transformations",
            "Alternate coordinate systems (bases)",
        },

    }

    conceptDict = {
        "higher_edu": higher_concepts,
        "high_school": high_concepts,
        "middle_school": middle_concepts,
        "elementary_school": element_concepts,
    }

class Generator:

    def __init__(self, openai_api_key_account:str=None) -> None:
        
        self.qtypes_starter = {
            "word_problem": "Please create a word problem",
            "theorem_proving": "Please create a theorem proving problem", 
        }

        self.qlevels = {
            "higher_edu": "in the level of higher education",
            "high_school": "in the level of high school",
            "middle_school": "in the level of middle school",
            "elementary_school": "in the level of elementary school",
        }

        self.qlevel2name = {
            "elementary_school": "ELEM_",
            "middle_school": "MIDD_",
            "high_school": "HIGH_",
            "higher_edu": "HEDU_",
        }

        self.key_account = openai_api_key_account
        
        self.out_dir = "./saved_samples"
        if not os.path.isdir(self.out_dir): os.mkdir(self.out_dir)
        
        self.parser = Parser()
        self.server = LeanEnv()

        # self.baseline_gen = {
        #     "all": self._gen_question_all_at_once,
        #     "step": self._gen_question_step_by_step,
        # }


    # @retry(wait=wait_fixed(1))
    def _llm_gen(self, system_msg, human_msg):
        messages = [
                {"role": "system",  "content": system_msg},
                {"role": "user",  "content": human_msg},
            ]

        response = openai.ChatCompletion.create(
            model=LLM_API_KEY["model"].lower(), 
            messages=messages,
            api_key=LLM_API_KEY["key"], 
            organization=LLM_API_KEY["org"], 
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
        )

        response = response['choices'][0]['message']['content']
        return response

    def _selecting_keywords(self, num_keyword, qlevel, qtype, kw_mode:str, verbose:bool=True):
        """
        The strategy of randomly selecting knowledge concepts. 
        NOTE: We use ";" to split the keywords. 

        :param kw_mode: (1) kw_only, (2) kwg: in the format of "$keyword in $selected_group"
        """
        _keys = ConceptLibrary.conceptDict[qlevel]
        _keys = list(_keys.keys())
        
        if num_keyword >= 1 and num_keyword <= 5:  # "k: k keywords from 1 group, where k\in{1,2,..}. "
            selected_group = random.choice(_keys)
            selected_keywords = random.sample(
                list(ConceptLibrary.conceptDict[qlevel][selected_group]), 
                min(num_keyword, len(ConceptLibrary.conceptDict[qlevel][selected_group]))
            )
            selected_kwg = [(item, selected_group) for item in selected_keywords]

            assert len(selected_keywords) == len(selected_kwg) == num_keyword 
        
        elif num_keyword == 0:  # "0: k keywords from 1 group, where k\in{1,2,..} by random.choice. "
            selected_group = random.choice(_keys)
            selecting_num = random.choice(range(5)) + 1
            selected_keywords = random.sample(
                list(ConceptLibrary.conceptDict[qlevel][selected_group]), 
                min(selecting_num, len(ConceptLibrary.conceptDict[qlevel][selected_group]))
            )
            selected_kwg = [(item, selected_group) for item in selected_keywords]

            assert len(selected_keywords) == len(selected_kwg) == selecting_num
        

        elif num_keyword < -1 and num_keyword >= -5:  # "-k: k keywords from k group, where k\in{1,2,..}. " 
            selected_groups = random.choices(_keys, k=abs(num_keyword))
            selected_keywords, selected_kwg = [], []  # list[str], list[tuple[str]]
            for selected_group in selected_groups:
                kw = random.sample(
                        list(ConceptLibrary.conceptDict[qlevel][selected_group]), 
                        min(1, len(ConceptLibrary.conceptDict[qlevel][selected_group])))  # list[str]
                selected_keywords.extend(kw)
                selected_kwg.append((kw[0], selected_group))

            assert len(selected_keywords) == len(selected_groups) == len(selected_kwg) == abs(num_keyword), "\nselected_keywords: {}\nselected_group: {}\nselected_kwg: {}\nnum_keyword: {}".format(len(selected_keywords), len(selected_groups), len(selected_kwg), abs(num_keyword))
        
        elif num_keyword == -1:  # "-1: k keywords from k group, where k\in{1,2,..} by random.choice. "
            selecting_num = random.choice(range(5)) + 1
            selected_groups = random.choices(_keys, k=selecting_num)
            selected_keywords, selected_kwg = [], []
            for selected_group in selected_groups:
                kw = random.sample(
                        list(ConceptLibrary.conceptDict[qlevel][selected_group]), 
                        min(1, len(ConceptLibrary.conceptDict[qlevel][selected_group])))  # list[str]
                selected_keywords.extend(kw)  
                selected_kwg.append((kw[0], selected_group))

            assert len(selected_keywords) == len(selected_group) == len(selected_kwg) == selecting_num, "\nselected_keywords: {}\nselected_group: {}\nselected_kwg: {}\selecting_num: {}".format(len(selected_keywords), len(selected_groups), len(selected_kwg), selecting_num)
        
        else:
            raise Exception("Num_keyword {} is not implemented. Please check the help document for usage of the argument num_keyword.".format(len(num_keyword)))        
        
        selected_keywords_ = "; ".join(["*{}*".format(item) for item in selected_keywords])
        selected_kwg_ = "; ".join(["*{}* in {}".format(item[0], item[1]) for item in selected_kwg])

        if kw_mode == "kw":  # keyword only.
            kw_included = selected_keywords_
            kw_save = selected_keywords  # list
        elif kw_mode == "kwg":  # keyword with domain. 
            kw_included = selected_kwg_
            kw_save = selected_kwg  # list
        else:
            raise Exception
        
        if verbose:
            print("\nSelected {} keyword(s):\n{}".format(len(selected_keywords), kw_included))
        
        keyword_sec = " " + "based on the following knowledge point(s): " + kw_included
        
        return keyword_sec, kw_save


    def _process_preset_keywords(self, k:list[tuple], verbose:bool=True):
        assert k is not None
        if k[0][1] is None:
            k = [item[0] for item in k]
            kws = "; ".join(["*{}*".format(item) for item in k])
        else:
            kws = "; ".join(["*{}* in {}".format(item[0], item[1]) for item in k])

        kw_included = kws
        kw_save = k

        if verbose:
            print("\nPreset {} keyword(s):\n{}".format(len(k), kws))

        keyword_sec = " " + "based on the following knowledge point(s): " + kw_included
        
        return keyword_sec, kw_save        


#     def _gen_once(self, qtype, qlevel, human_msg_keyword_sec):
#         """ first-time gen. all-at-once generation. """
#         system_msg = "You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ``Problem: ''), a corresponding solution in natural language (indicated by ``Informal proof:''), and a translated formal solution in Lean 3 (indicated by ``Formal proof in Lean 3:''). Please note that the informal proof and the formal proof need to be identical."                
                    
#         human_msg = self.qtypes_starter[qtype]
#         if qlevel:
#             human_msg += " " + self.qlevels[qlevel]
#         human_msg += human_msg_keyword_sec  
#         human_msg += "."
#         human_msg += """

# You must respond in the following format: 

# # Problem: ...

# # Informal proof: ...

# # Formal proof in Lean 3: ...

# """
#         return system_msg, human_msg
    
    def kernel_gen_once(self, qtype, qlevel, keyword_sec):          
        # system_msg, human_msg = self._gen_once(qtype=qtype, qlevel=qlevel, human_msg_keyword_sec=human_msg_keyword_sec)
        """ first-time gen. all-at-once generation. 
        :returns
            - question
            - informal_ans
            - formal_ans
        """
        system_msg = "You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ``Problem: ''), a corresponding solution in natural language (indicated by ``Informal proof:''), and a translated formal solution in Lean 3 (indicated by ``Formal proof in Lean 3:''). Please note that the informal proof and the formal proof need to be identical."                
                    
        human_msg = self.qtypes_starter[qtype]
        if qlevel:
            human_msg += " " + self.qlevels[qlevel]
        human_msg += keyword_sec  
        human_msg += "."
        human_msg += """

You must respond in the following format: 

# Problem: ...

# Informal proof: ...

# Formal proof in Lean 3: ...

"""

        response = self._llm_gen(system_msg, human_msg)
        question, informal_ans, formal_ans = self.parser._parse_response2proof(response) 

        return question, informal_ans, formal_ans


    def kernel_gen_step(self, qtype, qlevel, keyword_sec):
        """ first-time gen. step-by-step generation. 
        returns:
            - question
            - informal_ans
            - formal_ans
        """
        # step 1: generate question
        system_msg_q = "You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ``Problem: ''), a corresponding solution in natural language (indicated by ``Informal proof: ''), and a translated formal solution in Lean 3 (indicated by ``Formal proof in Lean 3: ''). Please note that the informal proof and the formal proof need to be identical."
        system_msg_kw = self.qtypes_starter[qtype]
        if qlevel:
            system_msg_kw += " " + self.qlevels[qlevel]
        system_msg_kw += keyword_sec  
        system_msg_kw += "."
        system_msg_q += system_msg_kw
        
        human_msg_q = """Now please write the math problem step-by-step following the instruction below. Please first write the question part regardless of the other parts. You must write in the following format, filling in the ``# Problem: '' section, and leaving the other two sections empty. 

# Problem: ...

# Informal proof: ...

# Formal proof in Lean 3: ...

"""
        response_q = self._llm_gen(system_msg_q, human_msg_q)
        question, _, _ = self.parser._parse_response2proof(response_q) 

        # step 2: generate informal proof
        human_msg_ip = """Now please write the math problem step-by-step following the instruction below. Please then write the corresponding solution in natural language (indicated by ``Informal proof: '') given the ``# Problem: '', filling in the ``# Informal proof: '' section, and leaving the other section empty. 
                        
# Problem: {}

# Informal proof: ...

# Formal proof in Lean 3: ...

""".format(question.strip())
        response_ip = self._llm_gen(system_msg_q, human_msg_ip)
        informal_ans = self.parser._parse_step_ip(response_ip)        

        # step 3: generate formal proof 
        system_msg_fp = copy.deepcopy(system_msg_q).replace("a math expert", "a master in Lean")
        human_msg_fp = """Now please write the math problem step-by-step following the instruction below. Please then write the corresponding solution in natural language (indicated by ``Informal proof: '') given the ``# Problem: '', filling in the ``# Informal proof: '' section, and leaving the other section empty.
                        
# Problem: {}

# Informal proof: ...

# Formal proof in Lean 3: 
```lean
...
```

""".format(question.strip(), informal_ans.strip())
        response_fp = self._llm_gen(system_msg_fp, human_msg_fp)
        formal_ans = self.parser._parse_formal(response_fp)

        return question, informal_ans, formal_ans

    
    def _gen_correct(self, sample):
        """ self-correction. 
        :param sample['problem']: str. problem.
        :param sample['informal_proof']: str. informal proof.
        :param sample['errs']: dict. error info.  
        """
        system_msg = """In the following, you are given a ``Problem'', a pair of corresponding ``Informal proof'', and a series of ``Formal proof in Lean 3'' along with the corresponding error messages from a Lean prover. Now please carefully modify the last ``Formal proof in Lean 3'' so that it passes the Lean prover without error. You should write the modified complete proof in your response."""

        human_msg = """
# Problem: 
{}

# Informal proof:
{}
""".format(sample['problem'], sample['informal_proof'])
        
        for i, e in enumerate(sample['formal_proofs'][:]):
            proof, infos = e
            lined_proof = self.parser._add_line_id(proof)
            human_msg += f"""
# Formal proof ({i+1}) in Lean 3:
{lined_proof}

# Error messages for Formal proof ({i+1}) from Lean prover:
{infos['error']}
"""
        return system_msg, human_msg
    
    @func_set_timeout(300)
    def _verify(self, s):
        infos = self.server.verify_lean_file(s)
        return infos

    # def _gen_question_all_at_once(self, qtype, qlevel:None, 
    #                               n_iter:tuple, 
    #                               num_keyword: int=None, kw_mode: str=None,
    #                               num_correct: int=None,
    #                               ):
    def generate(self, params):

        qtype=params.qtype
        qlevel=params.qlevel
        n_iter=params.n_iter
        num_keyword=params.num_keyword
        kw_mode=params.kw_mode
        num_correct=params.num_correct
        baseline_type=params.baseline_type
        preset_keywords=params.preset_keywords
        
        if num_keyword and kw_mode and qlevel: 
            group_dir = "group_{}{}".format(kw_mode, num_keyword)
            sample_dir = "group_all_{}{}_{}".format(kw_mode, num_keyword, qlevel) 
            out_qt_dir = os.path.join(self.out_dir, group_dir, qtype, sample_dir)  
        else:
            raise Exception
                
        if not os.path.isdir(out_qt_dir): 
            os.makedirs(os.path.join(out_qt_dir, 'data')) 
            os.makedirs(os.path.join(out_qt_dir, 'code')) 

        pid = os.getpid()
        with trange(n_iter+1, position=pid, desc=f"#{pid}") as tbar:
            tbar.set_description_str(f"# {os.getpid()} {qlevel}_{qtype}")
            for i in tbar:
                
                _prefix_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                qid = f"{self.qlevel2name[qlevel] + qtype}_{_prefix_time}"
                f_data = os.path.join(out_qt_dir, 'data', f"{qid}.json")
                f_code = os.path.join(out_qt_dir, 'code', f"{qid}.lean")

                ### selecting keywords   # TODO: add given keywords from input files. 
                if preset_keywords is not None:
                    keyword_sec, kws = self._process_preset_keywords(preset_keywords)
                elif num_keyword:
                    keyword_sec, kws = self._selecting_keywords(num_keyword=num_keyword, qlevel=qlevel, qtype=qtype, kw_mode=kw_mode)
                else:
                    raise Exception("ERROR: Please specify the keywords.")

                """ Start generating. """
                if os.path.isfile(f_data):
                    raise Exception(f"File {f_data} already exists.")
                else:

                    self.server.reset()
                    sample = {
                        'problem': None,
                        'informal_proof': None,
                        'formal_proofs': []
                        } 

                    # system_msg, human_msg = self._gen_once(qtype=qtype, qlevel=qlevel, human_msg_keyword_sec=human_msg_keyword_sec)
                    # response = self._llm_gen(system_msg, human_msg)

                    # _question, _informal_ans, _formal_ans = self.parser._parse_response2proof(response) 
                    if baseline_type == "all":
                        question, informal_ans, formal_ans = self.kernel_gen_once(qtype=qtype, qlevel=qlevel, keyword_sec=keyword_sec)
                    elif baseline_type == "step":
                        question, informal_ans, formal_ans = self.kernel_gen_step(qtype=qtype, qlevel=qlevel, keyword_sec=keyword_sec)
                    else:
                        raise NotImplementedError

                    if sample['problem'] is None: sample['problem'] = question 
                    if sample['informal_proof'] is None: sample['informal_proof'] = informal_ans 
                    
                    # infos = self.server.verify_lean_file(formal_ans)
                    try:
                        infos = self._verify(informal_ans)
                    except:
                        infos = {'error': 'ERROR: The lean code does not halt. Skipping this case.', 'open_states': []}
                        print(infos['error'])
                        continue
                    sample['formal_proofs'].append((formal_ans, infos))

                    # self-correct. use & update info_error. 
                    if num_correct > 0:
                        for c in range(num_correct):

                            if len(sample['formal_proofs'][-1][-1]['error'].strip()) == 0:  # infos['error'], str. 
                                break

                            system_msg, human_msg = self._gen_correct(sample=sample)
                            response = self._llm_gen(system_msg, human_msg)
                            formal_ans = self.parser._parse_formal(response) 
                    
                            # infos = self.server.verify_lean_file(formal_ans)
                            try:
                                infos = self._verify(informal_ans)
                            except:
                                infos = {'error': 'ERROR: The lean code does not halt. Skipping this case.', 'open_states': []}
                                print(infos['error'])
                                continue
                            sample['formal_proofs'].append((formal_ans, infos))

                    # remove `sorry` cases. 
                    if len(sample['formal_proofs'][-1][-1]['open_states']) > 0:  # `sorry` in proof. 
                        print('Filtered proof with `sorry`.')
                        continue

                    # remove `reduce` cases. 
                    if '#reduce' in sample['formal_proofs'][-1][0]:  # `#reduce` in proof. 
                        print('Filtered proof with `#reduce`.')
                        continue
                    
                    # save.
                    if len(sample['formal_proofs'][-1][-1]['error'].strip()) == 0:  # infos['error'], str. 

                        lean_code = sample['formal_proofs'][-1][0]  # str
                        
                        output_sample = {
                            "problem_name": qid,
                            "informal_statement": sample['problem'],
                            "informal_proof": sample['informal_proof'],
                            "formal_proof": lean_code,  
                            "metadata": {
                                "qtype": qtype,
                                "qlevel": qlevel,
                                "keywords": kws, 
                                "baseline_type": "all",
                                "evolution": sample['formal_proofs'],
                                "num_evolve": len(sample['formal_proofs']) - 1,
                            },
                        }

                        with open(f_data, 'w') as f:
                            json.dump(output_sample, f, indent=4)
                        with open(f_code, 'w') as f:
                            f.write(lean_code)
                        print(f'===> Saved new data {qid}.')
                    else:
                        print('Filtered invalid proof.')


        self.server.close()


    # def generate(self, params):
    #     self.generate(n_iter=params.n_iter, 
    #                       qtype=params.qtype, qlevel=params.qlevel, 
    #                       num_keyword=params.num_keyword, kw_mode=params.kw_mode,
    #                       num_correct=params.num_correct,
    #                       baseline_type=params.baseline_type)
