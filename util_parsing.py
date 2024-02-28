# Author: Yinya Huang
# Date: 17/7/2023
#



import re
import copy


class Parser:

    def _replace_n_split(self, s:str, replacee:list):
        replacer = '~~~~~~~'
        s = s.strip() 
        for ee in replacee:
            s = s.replace(ee, replacer)
        return s.split(replacer)
    
    def _parse_keywords(self, __string:str):
        """
        Return: 
            - _keywords: list[str]
        """
        _prefix = "based on the following knowledge point(s):"
        _keywords = __string.split(_prefix)[1]
        _keywords = _keywords.strip().split("\n")[0]
        _keywords = _keywords.strip().split(";")
        return _keywords        
    
    def _get_splitter(self):
        """
        31/7/2023. For the updated prompts. 
        2/1/2024. canceling param: mode. 
        """
        major_splitters = ["======= Prompt =======", "======= Response ======="]        
        minor_splitters = [
            "# Informal proof:",
            "**Informal proof:**",

            "# Formal proof in Lean 3:",
            "**Formal proof in Lean 3:**",

            "# Problem:",
            "**Problem:**",
        ]
        return major_splitters, minor_splitters

    def _parse_major_lite(self, _string:str):
        major_splitters, minor_splitters = self._get_splitter()
        major_splitted = self._replace_n_split(_string, major_splitters)
        assert len(major_splitted) == len(major_splitters) + 1
        
        prompt, response = major_splitted[-2], major_splitted[-1]
        return prompt, response
    
    def _add_line_id(self, _code):
        pos_newlines = [m.start() for m in re.finditer("\n", _code)]
        pos_newlines = pos_newlines[:-1]
        _code_lines = copy.deepcopy(_code)
        for _pos_newline, _line_id in zip(list(reversed(pos_newlines)), range(len(pos_newlines), 0, -1)):
            _code_lines = _code_lines[:_pos_newline] + "\nline {} ".format(_line_id) + _code_lines[_pos_newline+1:]
        return _code_lines

    def _parse_formal(self, _string:str):
        """
        Parse a formal proof section into lean code. 
        """
        _formal_proof = re.findall(r'```[\S\s]*```', _string)
        if _formal_proof == []:
            _formal_proof = re.findall(r'```[\S\s]*', _string)
        if _formal_proof == []:
            _formal_proof = [_string]
        # assert len(_formal_proof) == 1                    
        _formal_proof = _formal_proof[0]
        _formal_proof = _formal_proof.strip()
        
        lean_code = "\n".join(_formal_proof.strip().split("\n")[1:-1])  # remove ```lean ```
        lean_code = re.sub(pattern=r'line [0-9]* ', repl='', string=lean_code)  # remove line *

        return lean_code
    

    def _parse_response2proof_dep(self, _string:str):
        """
        Parsing a response (without prompt) into sections and return.

        :return
            - question 
            - informal statement 
            - formal statement 
        """

        def sepa(minor_splitted):
            if len(minor_splitted) == 4:
                problem, solution_nl, solution_fm = minor_splitted[-3].strip(), minor_splitted[-2].strip(), minor_splitted[-1].strip()
                return problem.strip(), solution_nl.strip(), solution_fm.strip()
            elif len(minor_splitted) == 3 and "# Problem:" not in _minor_splitters:
                solution_nl, solution_fm = minor_splitted[-2].strip(), minor_splitted[-1].strip()
                return None, solution_nl, solution_fm
            elif len(minor_splitted) == 3 and "# Formal proof in Lean 3:" not in _minor_splitters:
                problem, solution_nl = minor_splitted[-2].strip(), minor_splitted[-1].strip()
                return problem, solution_nl, None
            elif len(minor_splitted) == 2:
                output = minor_splitted[-1].strip()
                return (output.strip(), )
            else:
                print("minor_splitted", minor_splitted, len(minor_splitted))
                raise Exception()

        major_splitters, minor_splitters = self._get_splitter() 

        _minor_splitters = [item for item in minor_splitters if item in _string] 

        minor_splitted = self._replace_n_split(_string, _minor_splitters) 
        assert len(minor_splitted) == len(_minor_splitters) + 1

        _question, _informal_ans, _formal_ans = sepa(minor_splitted)
        assert not len(_question) == 0
        assert not len(_informal_ans.strip()) == 0
        assert not len(_formal_ans.strip()) == 0 

        _formal_ans = self._parse_formal(_formal_ans)

        return _question, _informal_ans, _formal_ans

        
    def _parse_response2proof(self, _string:str):
        """
        """

        problem_pattern = re.compile('(\#|\*)*( )*(P|p)roblem( )*(:|\*|\#)*( )*(:|\*|\#)*')
        problem_ids = [m.start() for m in problem_pattern.finditer(_string)]

        ip_pattern = re.compile('(\#|\*)*( )*(I|i)nformal (P|p)roof( )*(:|\*|\#)*( )*(:|\*|\#)*')
        ip_ids = [m.start() for m in ip_pattern.finditer(_string)]

        fp_pattern = re.compile('(\#|\*)*( )*[^(I|i)n](F|f)ormal (P|p)roof( )*(:|\*|\#)*( )*(:|\*|\#)*')
        fp_ids = [m.start() for m in fp_pattern.finditer(_string)]

        if not (len(problem_ids) == 1 and len(ip_ids) == 1 or len(fp_ids) == 1): 
            raise Exception('Response can not be parsed.')
        
        _question = _string[problem_ids[0]:ip_ids[0]]
        _informal_ans = _string[ip_ids[0]:fp_ids[0]]
        _formal_ans = _string[fp_ids[0]:]

        _formal_ans = self._parse_formal(_formal_ans)

        return _question, _informal_ans, _formal_ans


    def _detect_sections(self, _string:str):
        problem_pattern = re.compile('(\#|\*)*( )*(P|p)roblem( )*(:|\*|\#)*( )*(:|\*|\#)*')
        problem_ids = [m.start() for m in problem_pattern.finditer(_string)]

        ip_pattern = re.compile('(\#|\*)*( )*(I|i)nformal (P|p)roof( )*(:|\*|\#)*( )*(:|\*|\#)*')
        ip_ids = [m.start() for m in ip_pattern.finditer(_string)]

        fp_pattern = re.compile('(\#|\*)*( )*[^(I|i)n](F|f)ormal (P|p)roof( )*(:|\*|\#)*( )*(:|\*|\#)*')
        fp_ids = [m.start() for m in fp_pattern.finditer(_string)]

        return problem_ids, ip_ids, fp_ids
        
    def _parse_response2proof(self, _string:str):
        """ """

        problem_ids, ip_ids, fp_ids = self._detect_sections(_string)    

        if not (len(problem_ids) == 1 and len(ip_ids) == 1 or len(fp_ids) == 1): 
            raise Exception('Response can not be parsed.')
        
        question = _string[problem_ids[0]:ip_ids[0]]
        informal_ans = _string[ip_ids[0]:fp_ids[0]]
        formal_ans = _string[fp_ids[0]:]

        formal_ans = self._parse_formal(formal_ans)

        return question, informal_ans, formal_ans

    def _parse_step_ip(self, _string:str):
        """  """

        problem_ids, ip_ids, fp_ids = self._detect_sections(_string)    
        if len(ip_ids) == 1 and len(fp_ids) == 1:
            informal_ans = _string[ip_ids[0]:fp_ids[0]]
        elif len(ip_ids) == 1 and len(fp_ids) == 0:
            informal_ans = _string[ip_ids[0]:]
        elif len(ip_ids) == 0 and len(fp_ids) == 0:
            informal_ans = _string
        else:
            raise NotImplementedError
        
        return informal_ans
    