import os
from pathlib import Path
import subprocess
import ujson as json
# from tqdm import tqdm, trange
import re
from leaven.src.lean_server import LeanEnv, LeanServerNoInfoError
# import pickle
import pickle
import networkx as nx
from leaven.src.file_tools import *

leaven_path = Path(__file__).parent.parent.resolve()
src_path = Path(__file__).parent.resolve()

def get_name_from_leanpkg_path(p: Path) -> str:
  """ get the package name corresponding to a source path """
  # lean core?
  if p.parts[-5:] == Path('bin/../lib/lean/library').parts:
    return "core"
  if p.parts[-3:] == Path('bin/../library').parts:
    return "core"
  return '<unknown>'

lean_path_basis = [Path(p) for p in json.loads(subprocess.check_output([f_join(leaven_path, 'elan', 'bin', 'lean'), '--path'], cwd=leaven_path).decode())['path']]
path_info = [(p.resolve(), get_name_from_leanpkg_path(p)) for p in lean_path_basis]
lean_paths = [p.resolve() for p in lean_path_basis if p.exists()]

def cut_path(path):
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve().with_suffix('')
    for lean_path in lean_paths:
        if path.is_relative_to(lean_path):
            return '.'.join(path.relative_to(lean_path).parts)
    return None

def object2Map(obj:object):
    m = obj.__dict__
    for k in m.keys():
        v = m[k]
        if hasattr(v, "__dict__"):
            m[k] = object2Map(v)
    return m

class GrammarRegistry:
    def __init__(self, grammarPath, multiline=True):
        self.multiline = multiline
        grammar = json.load(open(grammarPath,'r',encoding="utf-8"))
        assert isinstance(grammar['scopeName'], str) and grammar['scopeName'], f"Grammar missing required scopeName property: #{grammarPath}"
        self.begin_pattern = None
        self.end_pattern = None
        self.scope_name = grammar['scopeName']
        self.content_name = None
        self.patterns = [Rule(pat) for pat in grammar['patterns']]
        self.repository = {key : Rule(value) for key, value in grammar['repository'].items()} if 'repository' in grammar else None
        self.redirect_patterns(self.repository)
        self.redirect_patterns(self.patterns)

    def redirect_patterns(self, patterns):
        assert isinstance(patterns, (list, dict))
        for idx, rule in (enumerate(patterns) if isinstance(patterns, list) else patterns.items()):
            if isinstance(rule.include, str) and rule.include[1:] in self.repository:
                patterns[idx] = self.repository[rule.include[1:]]
                # if rule.match_first:
                #     patterns[idx].match_first = rule.match_first
            elif rule.patterns:
                self.redirect_patterns(rule.patterns)
    
    def get_begin_pattern(self, rule):
        if rule.begin_pattern:
            return [[rule.begin_pattern , rule]]
        elif rule.patterns:
            begin_pattern = []
            for pattern in rule.patterns:
                begin_pattern.extend(self.get_begin_pattern(pattern))
            return begin_pattern
        else:
            return []
        
    def tokenize_line(self, line, repository=None):
        tagging_log = []
        self.rule_apply(self if not repository else self.repository[repository], line, [], tagging_log)
        return tagging_log

    # def get_next_regex(self, line, begin_patterns):
    #     starts = []

    def rule_apply(self, rule, line, scope_name, tagging_log, outside_end=[]):
        begin_patterns = [(re.compile(item[0], re.M) if self.multiline else re.compile(item[0]), item[1]) for pattern in rule.patterns for item in self.get_begin_pattern(pattern) if not pattern.match_first] if rule.patterns else []
        match_first_patterns = [(re.compile(item[0], re.M) if self.multiline else re.compile(item[0]), item[1]) for pattern in rule.patterns for item in self.get_begin_pattern(pattern) if pattern.match_first] if rule.patterns else []
        # if not begin_patterns:
        #     return 0
        pos = 0
        last_pos = 0
        scope_name = scope_name + ([rule.scope_name] if rule.scope_name else [])
        content_name = scope_name + ([rule.content_name] if rule.content_name else [])
        # end_pattern = re.compile(rule.end_pattern) if rule.end_pattern else None
        if rule.end_pattern:
            end_pattern = re.compile(rule.end_pattern, re.M) if self.multiline else re.compile(rule.end_pattern)
        if outside_end:
            outside_end_pattern = [re.compile(i, re.M) if self.multiline else re.compile(i) for i in outside_end]
        while pos < len(line):
            matched = False
            for regex_pattern, pattern in match_first_patterns:
                if (match := regex_pattern.match(line, pos)):
                    matched = True
                    # if '#upper_ending' == pattern.end_pattern:
                    #     match = match
                    if pos > last_pos:
                        tagging_log.append([line[last_pos : pos], content_name])
                    if pattern.end_pattern:
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        if pattern.soft_ending:
                            apply_result = self.rule_apply(pattern, line[match.end():], content_name, tagging_log, ([rule.end_pattern] if rule.end_pattern else []) + outside_end)
                            last_pos = pos = match.end() + apply_result
                        else:
                            last_pos = pos = match.end() + self.rule_apply(pattern, line[match.end():], content_name, tagging_log)
                    else:
                        if pattern.soft_ending and rule.end_pattern and end_pattern.search(match.group(1)):
                            matched = False
                            continue
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        # tagging_log.append([line[pos + match.start() : pos + match.end()], content_name + [pattern.scope_name]])
                        last_pos = pos = match.end()
                    break
            if matched:
                continue
            if rule.end_pattern and rule.end_pattern != '#just_matching_upper_ending' and (match := end_pattern.match(line, pos)):
                if pos > last_pos:
                    tagging_log.append([line[last_pos : pos], content_name])
                if match.start() < match.end():
                    tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, scope_name, rule.end_captures if rule.end_captures else {}))
                return match.end()
            if outside_end:
                for pattern in outside_end_pattern:
                    if pattern.match(line, pos):
                        if line[last_pos : pos]:
                            tagging_log.append([line[last_pos : pos], content_name])
                        return pos
            for regex_pattern, pattern in begin_patterns:
                if (match := regex_pattern.match(line, pos)):
                    matched = True
                    # if '#upper_ending' == pattern.end_pattern:
                    #     match = match
                    if pattern.end_pattern:
                        if pos > last_pos:
                            tagging_log.append([line[last_pos : pos], content_name])
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        if pattern.soft_ending:
                            apply_result = self.rule_apply(pattern, line[match.end():], content_name, tagging_log, ([rule.end_pattern] if rule.end_pattern else []) + outside_end)
                            last_pos = pos = match.end() + apply_result
                        else:
                            last_pos = pos = match.end() + self.rule_apply(pattern, line[match.end():], content_name, tagging_log)
                    else:
                        if pattern.soft_ending and rule.end_pattern and end_pattern.search(match.group()):
                            matched = False
                            continue
                        if pos > last_pos:
                            tagging_log.append([line[last_pos : pos], content_name])
                        if match.start() < match.end():
                            tagging_log.extend(self.capture(line[match.start() : match.end()], pos, match.regs, content_name + ([pattern.scope_name] if pattern.scope_name else []), pattern.begin_captures if pattern.begin_captures else {}))
                        # tagging_log.append([line[pos + match.start() : pos + match.end()], content_name + [pattern.scope_name]])
                        last_pos = pos = match.end()
                    break
            if not matched:
                pos += 1
        if pos > last_pos:
            tagging_log.append([line[last_pos : pos], content_name])
        return pos

    def capture(self, line, pos, regs, scope_name, captures):
        tags = [scope_name.copy() for token in line]
        for i in captures:
            for idx in range(len(line)):
                if regs[int(i)][0] <= pos + idx and pos + idx < regs[int(i)][1] and captures[i]['name']:
                    tags[idx].append(captures[i]['name'])
        tokenized = []
        flag = 0
        last_tag = tags[0]
        for idx, tag in enumerate(tags):
            if not tag == last_tag:
                tokenized.append([line[flag : idx], last_tag])
                flag = idx
                last_tag = tag
        tokenized.append([line[flag : len(tags)], last_tag])
        return tokenized


class Rule:
    def __init__(self, pattern):
        self.include = pattern['include'] if 'include' in pattern and pattern['include'] else None
        self.scope_name = pattern['name'] if 'name' in pattern and pattern['name'] else None
        self.begin_captures = pattern['beginCaptures'] if 'beginCaptures' in pattern and pattern['beginCaptures'] else None
        self.end_captures = pattern['endCaptures'] if 'endCaptures' in pattern and pattern['endCaptures'] else None
        self.end_pattern = None
        self.content_name = pattern['contentName'] if 'contentName' in pattern and pattern['contentName'] else None
        self.begin_pattern = None
        self.soft_ending = pattern["soft_ending"] if "soft_ending" in pattern else False
        self.match_first = pattern["match_first"] if "match_first" in pattern else False
        if 'match' in pattern:
            self.begin_pattern = pattern['match']
        elif 'begin' in pattern:
            assert 'end' in pattern
            self.begin_pattern = pattern['begin']
            # if self.begin_captures:
            #     self.captures = self.begin_captures
            self.end_pattern = pattern['end']
        self.patterns = [Rule(pat) for pat in pattern['patterns']] if 'patterns' in pattern and pattern['patterns'] else None

class SyntaxParser:
    @classmethod
    def parsing_declarations(cls, lines):
        file_grammar = GrammarRegistry(src_path / 'lean_syntax' / 'lean_grammar.json')
        lines = file_grammar.tokenize_line(lines)
        lines = [[line[0], line[1][1:]] for line in lines]
        modifier_flag = []
        last_flag = 0
        blocks = []
        for idx, line in enumerate(lines):
            if not line[0].strip():
                continue
            else:
                break
        lines = lines[idx:]
        for idx, line in enumerate(lines):
            name = ','.join(line[1])
            if not line[0].strip() and modifier_flag and modifier_flag[-1] + 1 == idx:
                modifier_flag.append(idx)
            elif 'modifier' in name or "block.documentation" in name:
                if not idx or ((('modifier' in name or "block.documentation" in name)) and modifier_flag and modifier_flag[-1] + 1 == idx):
                    modifier_flag.append(idx)
                else:
                    head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:idx] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
                    if not head and 'storage.modifier.attribute' in  ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]):
                        head = ['attribute']
                    blocks.append(head + lines[last_flag:idx])
                    last_flag = idx
                    modifier_flag = [idx]
            elif not idx:
                continue
            elif ('codeblock' in name and not modifier_flag) or 'keyword.end' in name:
                head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:idx] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
                if not head and 'storage.modifier.attribute' in  ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]):
                    head = ['attribute']
                blocks.append(head + lines[last_flag:idx])
                last_flag = idx
            elif ('codeblock' in name and modifier_flag[-1] + 1 < idx) or 'keyword.end' in name:
                head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:idx] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
                if not head and 'storage.modifier.attribute' in  ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]):
                    head = ['attribute']
                blocks.append(head + lines[last_flag:idx])
                last_flag = idx
                modifier_flag = []
            elif 'codeblock' in name and modifier_flag:
                modifier_flag = []
        head = ["documentation"] if "codeblock.keyword.documentation" in ','.join([','.join(tree[1]) for tree in lines[last_flag:idx]]) else [tree[0] for tree in lines[last_flag:len(lines)] if 'codeblock' in ','.join(tree[1]) or 'keyword.end' in ','.join(tree[1])]
        blocks.append(head + lines[last_flag:len(lines)])
        return blocks
    
    @classmethod
    def parsing_steps(cls, text : str, remove_modifier=False, return_index=False):
        def merge_consecutive_strings(lst):
            merged = []
            current_str = ""
            for item in lst:
                if isinstance(item, str):
                    current_str += item
                else:
                    if current_str:
                        merged.append(current_str)
                        current_str = ""
                    merged.append(item)
            if current_str:
                merged.append(current_str)
            return merged

        def decoupling(l: list):
            for idx in range(len(l)):
                if isinstance(l[idx], list):
                    if len(l[idx]) == 2 and isinstance(l[idx][0], str) and isinstance(l[idx][1], list) and l[idx][0] in ['calc_block', 'proof_block', 'goal_block', 'begin_block']:
                        l[idx] = decoupling(l[idx][1])
                    else:
                        l[idx] = ''.join(dfs_print(l[idx][1]))
            result = merge_consecutive_strings(l)
            return result
        
        def merge_consecutive_list(lst):
            merged = []
            current_list = []
            for item in lst:
                if isinstance(item, tuple) and len(item) == 2 and all(isinstance(i, int) for i in item):
                    if not current_list:
                        current_list = item[ : ]
                    else:
                        assert current_list[1] == item[0]
                        current_list = (current_list[0], item[1])
                else:
                    if current_list:
                        merged.append(current_list)
                        current_list = []
                    merged.append(item)
            if current_list:
                merged.append(current_list)
            return merged
        
        def decoupling_index(text, flag, l: list):
            for idx in range(len(l)):
                if isinstance(l[idx], list):
                    if len(l[idx]) == 2 and isinstance(l[idx][0], str) and isinstance(l[idx][1], list) and l[idx][0] in ['calc_block', 'proof_block', 'goal_block', 'begin_block', 'tactic_block']:
                        l[idx], flag = decoupling_index(text, flag, l[idx][1])
                    else:
                        dfs_pt = ''.join(dfs_print(l[idx][1]))
                        str_len = len(dfs_pt)
                        assert text[flag : flag + str_len] == dfs_pt
                        l[idx] = (flag, flag + str_len)
                        flag += str_len
                else:
                    str_len = len(l[idx])
                    assert text[flag : flag + str_len] == l[idx]
                    l[idx] = (flag, flag + str_len)
                    flag += str_len
            result = merge_consecutive_list(l)
            return result, flag
        
        def dfs_print(fraction : list):
            target = []
            for i in range(len(fraction)):
                if isinstance(fraction[i], list):
                    assert len(fraction[i]) == 2 and isinstance(fraction[i][0], str)
                    target.extend(dfs_print(fraction[i][1]))
                else:
                    target.append(fraction[i])
            return target

        theorem = [[]]
        theorem_grammar = GrammarRegistry(src_path / 'lean_syntax' / 'proof_grammar.json', multiline=False)
        theorem_split = [[line[0], line[1][1:]] for line in theorem_grammar.tokenize_line(text) if len(line) > 1]
        for token in theorem_split:
            flag = theorem[-1]
            for label in token[1]:
                if not flag or flag[-1][0] != label:
                    flag.append([label, []])
                flag = flag[-1][1]
            flag.append(token[0])
        theorem = theorem[0]
        modifiers = [i for i, tree in enumerate(theorem) if tree[0] == 'storage.modifier']
        if remove_modifier:
            theorem = [line for idx, line in enumerate(theorem) if idx not in modifiers]
        if return_index:
            return decoupling_index(text, 0, theorem)[0]
        else:
            return decoupling(theorem)

def parse_export(decls, path=None, only_path=False):
    if only_path:
        assert path is not None
        file_list = [str(i) for i in Path(path).glob('**/*.lean')]
    from collections import defaultdict
        
    def separate_results(objs):
        file_map = defaultdict(list)
        loc_map = {}
        for obj in objs:
            i_name = obj['filename']
            if 'export_json' in i_name:
                continue  # this is doc-gen itself
            file_map[i_name].append(obj)
            loc_map[obj['name']] = i_name
            for (cstr_name, tp) in obj['constructors']:
                loc_map[cstr_name] = i_name
            for (sf_name, tp) in obj['structure_fields']:
                loc_map[sf_name] = i_name
            if len(obj['structure_fields']) > 0:
                loc_map[obj['name'] + '.mk'] = i_name
        return file_map, loc_map

    def linkify_efmt(f):
        def go(f):
            if isinstance(f, str):
                f = f.replace('\n', ' ')
                # f = f.replace(' ', '&nbsp;')
                return ''.join(
                    match[4] if match[0] == '' else
                    match[1] + match[2] + match[3]
                    for match in re.findall(r'\ue000(.+?)\ue001(\s*)(.*?)(\s*)\ue002|([^\ue000]+)', f))
            elif f[0] == 'n':
                return go(f[1])
            elif f[0] == 'c':
                return go(f[1]) + go(f[2])
            else:
                raise Exception('unknown efmt object')

        return go(['n', f])
    
    def mk_export_map_entry(decl_name, filename, kind, is_meta, line, args, tp, description, attributes):
        return {'decl_name': decl_name,
                'filename': cut_path(filename),
                'local_filename': str(filename),
                'kind': kind,
                'is_meta': is_meta,
                'line': line,
                'args': [{key : linkify_efmt(value) if key == 'arg' else value for key, value in item.items()} for item in args],
                'type': linkify_efmt(tp),
                'attributes': attributes,
                'description': description,
                # 'src_link': library_link(filename, line),
                # 'docs_link': f'{site_root}{filename.url}#{decl_name}'
                }

    file_map, loc_map = separate_results(decls['decls'])
    # for entry in decls['tactic_docs']:
    #     if len(entry['tags']) == 0:
    #         entry['tags'] = ['untagged']

    # mod_docs = {f: docs for f, docs in decls['mod_docs'].items()}
    # # ensure the key is present for `default.lean` modules with no declarations
    # for i_name in mod_docs:
    #     if 'export_json' in i_name:
    #         continue  # this is doc-gen itself
    #     file_map[i_name]

    # return file_map, loc_map, decls['notes'], mod_docs, decls['instances'], decls['instances_for'], decls['tactic_docs']
    export_db = {}
    appended_fields = []
    for _, decls in file_map.items():
        if only_path and decls[0]['filename'] not in file_list:
            continue
        for obj in decls:
            export_db[obj['name']] = mk_export_map_entry(obj['name'], obj['filename'], obj['kind'], obj['is_meta'], obj['line'], obj['args'], obj['type'], obj['doc_string'], obj['attributes'])
            # export_db[obj['name']]['decl_header_html'] = env.get_template('decl_header.j2').render(decl=obj)
            for (cstr_name, tp) in obj['constructors']:
                export_db[cstr_name] = mk_export_map_entry(cstr_name, obj['filename'], obj['kind'], obj['is_meta'], obj['line'], [], tp, obj['doc_string'], obj['attributes'])
                appended_fields.append(cstr_name)
            for (sf_name, tp) in obj['structure_fields']:
                export_db[sf_name] = mk_export_map_entry(sf_name, obj['filename'],  obj['kind'], obj['is_meta'], obj['line'], [], tp, obj['doc_string'], obj['attributes'])
                appended_fields.append(sf_name)
    return export_db, appended_fields, decls

def parsing_file(filename, decls):
    decl_pos = []
    for d in decls:
        decl_pos.append((d, decls[d]['line'] - 1))
    with open(next(iter(decls.values()))['local_filename']) as f:
        lines = f.readlines()
    line_breaks = [0] + sorted(list(set([x[1] for x in decl_pos]))) + [len(lines)]
    line_break_switches = {}
    for i, b in enumerate(line_breaks[1 : -1]):
        index = line_breaks.index(b)
        last_b = line_breaks[index - 1]
        if m := re.match(r'.*(/--.*-/\s*)$', ''.join(lines[last_b : b]), re.DOTALL):
            line_break_switches[b] = b - m.group(1).count('\n')
        else:
            line_break_switches[b] = b
    decl_pos = sorted([(i, line_break_switches[j]) for i, j in decl_pos], key=lambda x : (x[1], x[0]))
    line_breaks = sorted(list(set([x[1] for x in decl_pos])))
    # decl_blocks[cut_paths[file]] = {}
    # decl_blocks[cut_paths[file]]['header'] = ''.join(lines[ : line_breaks[0]])
    line_breaks = [0] + line_breaks + [len(lines)]
    # decl_blocks[cut_paths[file]]['blocks'] = {}
    for d, i in decl_pos:
        index = line_breaks.index(i)
        source = ''.join(lines[line_breaks[index] : line_breaks[index + 1]])
        # decl_blocks[cut_paths[file]]['blocks'][d] = (''.join(lines[line_breaks[max(index - 5, 1)] : line_breaks[index]]), source)
        decls[d]['source'] = ''.join([i[0] for i in SyntaxParser.parsing_declarations(source)[0][1:]])
        decls[d]['line'] = [j for i, j in decl_pos if i == d][0] + 1
    return filename, decls

def get_decl_source(decls_per_file, num_processes=30):
    import multiprocessing
    from tqdm import tqdm

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(decls_per_file)) as tbar:
        def get(result, decls_per_file):
            decls_per_file[result[0]] = result[1]
            tbar.update()
        
        results = [pool.apply_async(parsing_file, item) for item in decls_per_file.items()]
        results = [get(result.get(), decls_per_file) for result in results]

    return decls_per_file

def get_decl_info(path, only_path=False, max_memory_limit=102400):
    global lean_path_basis
    global lean_paths
    path = Path(path).resolve()
    revised_path = False
    with open(leaven_path / 'leanpkg.path') as f:
        leanpkg_path = f.read()
    if f'\npath {path}' not in leanpkg_path:
        with open(leaven_path / 'leanpkg.path', 'w') as f:
            f.write(leanpkg_path + f'\npath {path}')
        revised_path = True
        lean_path_basis = [Path(p) for p in json.loads(subprocess.check_output([f_join(leaven_path, 'elan', 'bin', 'lean'), '--path'], cwd=leaven_path).decode())['path']]
        lean_paths = [p.resolve() for p in lean_path_basis if p.exists()]
    file_list = [i for i in path.iterdir() if i.suffix == '.lean'] if path.is_dir() else [path]
    load_file_list = []
    for file in file_list:
        file_cut = cut_path(file)
        assert file_cut is not None
        load_file_list.append(file_cut)
    with open(src_path / 'entrypoint.lean', 'w') as f:
        f.write('\n'.join([f'import {i}' for i in load_file_list] + \
                            ["import .export_json\nimport all\nopen_all_locales"]
                            ))
    command = ["lean", "--run", src_path / 'entrypoint.lean', "-M" , str(max_memory_limit)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(leaven_path))
    # clear temp files
    os.remove(src_path / 'entrypoint.lean')
    if revised_path:
        with open(leaven_path / 'leanpkg.path', 'w') as f:
            f.write(leanpkg_path)
    return parse_export(json.loads(result.stdout), path=path, only_path=only_path)

def simplify_lean_code(code):
    import re
    from collections import defaultdict

    def combine_commands(lines):
        simplified_lines = []  # 存储简化后的代码行
        namespace_stack = [None]  # 栈用来存储当前的namespace层级
        opened_namespaces = defaultdict(list) 
        opened_namespaces[None] = list()
        universe_namespaces = defaultdict(list)
        universe_namespaces[None] = list()
        variable_namespaces = defaultdict(list)
        variable_namespaces[None] = list()
        theory_namespaces = defaultdict(list)
        theory_namespaces[None] = list()

        for line in lines:
            ns_match = re.match(r'\s*(?:namespace|section)\s*(.*)', line)
            end_match = re.match(r'\s*end\s*(.*)', line)
            open_match = re.match(r'\s*open\s*(.*)', line)
            universe_match = re.match(r'\s*universes* (.*)', line)
            variable_match = re.match(r'\s*variables* (.*)', line)

            if line.strip() == 'noncomputable theory':
                should_add = True
                for ns in reversed(namespace_stack):
                    if 'noncomputable theory' in theory_namespaces[ns]:
                        should_add = False
                        break
                if should_add:
                    theory_namespaces[namespace_stack[-1]].append('noncomputable theory')
                    simplified_lines.append(line)
            if ns_match:
                ns_name = ns_match.group(1).strip()
                namespace_stack.append(ns_name)
                simplified_lines.append(line)
            elif end_match:
                ns_name = end_match.group(1).strip()
                last_ns = namespace_stack.pop()
                assert last_ns == ns_name
                simplified_lines.append(line)
            elif open_match:
                open_ns = open_match.group(1).strip().split()
                ns_to_add = []
                should_add = True
                # 检查是否在当前namespace层级或上层中已经打开了这个命名空间
                for open_n in open_ns:
                    for ns in reversed(namespace_stack):
                        if open_n in opened_namespaces[ns]:
                            should_add = False
                            break
                    if should_add:
                        # 如果没有打开过这个命名空间，则添加这个open语句并更新opened_namespaces
                        opened_namespaces[namespace_stack[-1]].append(open_n)
                        ns_to_add.append(open_n)
                if ns_to_add:
                    simplified_lines.append('open ' + ' '.join(ns_to_add))
            elif universe_match:
                universes = universe_match.group(1).split()
                universe_to_add = []
                should_add = True
                for universe in universes:
                    for ns in reversed(namespace_stack):
                        if universe in universe_namespaces[ns]:
                            should_add = False
                            break
                    if should_add:
                        universe_namespaces[namespace_stack[-1]].append(universe)
                        universe_to_add.append(universe)
                if universe_to_add:
                    simplified_lines.append('universes ' + ' '.join(universe_to_add))
            elif variable_match:
                variable_command = variable_match.group(1).strip()
                should_add = True
                for ns in reversed(namespace_stack):
                    if variable_command in variable_namespaces[ns]:
                        should_add = False
                        break
                if should_add:
                    variable_namespaces[namespace_stack[-1]].append(variable_command)
                    simplified_lines.append(line)
            else:
                # 对于非namespace, end和open语句，直接添加到简化后的代码行
                simplified_lines.append(line)
        return simplified_lines
    
    def combine_namespaces(lines):
        simplified_lines = []
        for line in lines:
            ns_match = re.match(r'\s*(?:namespace|section)(.*)', line)
            end_match = re.match(r'\s*end(.*)', line)
            if ns_match:
                ns_name = ns_match.group(1).strip()
                # 查找simplified_lines中最后一个非空行
                for i in range(len(simplified_lines) - 1, -1, -1):
                    if simplified_lines[i].strip():
                        last_non_empty_line = simplified_lines[i]
                        break
                else:
                    last_non_empty_line = ''
                if ns_name:
                    end_match = re.match(r'\s*end ' + re.escape(ns_name), last_non_empty_line)
                else:
                    end_match = re.match(r'\s*end', last_non_empty_line)
                if end_match:
                    # 如果找到匹配的end语句，删除它
                    del simplified_lines[i]
                else:
                    # 否则，添加当前的namespace语句
                    simplified_lines.append(line)
            elif end_match:
                # 添加当前的end语句
                simplified_lines.append(line)
            else:
                # 对于非namespace和end语句，直接添加到简化后的代码行
                simplified_lines.append(line)
        return simplified_lines
    
    simplified_lines = combine_commands(combine_namespaces(combine_commands(code.split('\n'))))

    return re.sub(r'\n\n+', '\n\n', '\n'.join(simplified_lines))

def auto_complete_code(code_str):
    stack = []
    name_regex = re.compile(r'(?:namespace|section)\s*(.*)')
    for line in code_str.split("\n"):
        if line.startswith("namespace") or line.startswith("section"):
            stack.append(line)
    while len(stack) > 0:
        last_name = stack.pop()
        last_name = name_regex.match(last_name).group(1)
        code_str += "\nend " + last_name
    return code_str

def get_proving_environment(proving_environment):
    stack = []
    for i in proving_environment:
        if i[2] == 'end':
            while True:
                last_item = stack.pop()
                if last_item[2] not in ['namespace', 'section']:
                    continue
                if end_name := i[3][i[3].find(i[2]) + len(i[2]) : ].strip():
                    closed_name = last_item[3][last_item[3].find(last_item[2]) + len(last_item[2]) : ].strip()
                    assert end_name == closed_name, (end_name, closed_name)
                break
        else:
            stack.append(i)
    return stack

def calculate_end_position(lines, start_line, start_column):
    if isinstance(lines, str):
        lines = lines.split('\n')
    end_line = start_line + len(lines) - 1
    if len(lines) > 1:
        end_column = len(lines[-1])
    else:
        end_column = start_column + len(lines[0])
    return [end_line, end_column]

def simplify_environment(decl_list, proving_environment):
    def get_name(item):
        return item[3][item[3].find(item[2]) + len(item[2]) : ].strip()
    
    commands = sorted([[item['start'], item['end'], item['kind'], item['source']] if 'start' in item else [[item['line'], 0], calculate_end_position(item['source'], item['line'], 0), item['kind'], item['source']] for item in decl_list], key=lambda x : x[ : 2])
    commands = sorted(commands + [list(item) for item in proving_environment if (item[1] <= commands[-1][0] or item[2] in ['namespace', 'section', 'end']) and not any(c[0] <= item[0] and item[1] <= c[1] for c in commands)], key=lambda x : x[ : 2])
    stack = []
    closed_local_names = []
    last_theorem_id = [i for i, item in enumerate(commands) if item[2] == 'theorem'][-1]
    for i, item in enumerate(commands):
        if item[2] == 'end' and i > last_theorem_id:
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][2] in ['namespace', 'section']:
                    if end_name := get_name(item):
                        last_name = get_name(stack[j])
                        assert end_name == last_name, (end_name, last_name)
                        if stack[j][2] == 'namespace':
                            closed_local_names.append(last_name)
                    stack = stack[ : j]
                    break
                elif stack[j][2] in ['def', 'constant', 'axiom', 'theorem', 'definition', 'structure', 'inductive', 'class', 'abbreviation', 'instance', 'class_inductive', 'run_cmd', 'user_command', 'attribute', 'export', 'open']:
                    stack.append(item)
                    break
        # elif item[2] == 'open' and (open_name := get_name(item)) in closed_local_names:
        #     if any(i[2] == 'end' and get_name(i) == open_name for i in stack):
        #         stack.append(item)
        else:
            stack.append(item)
    return stack

def get_fraction(start, end, lines):
    if start[0] < end[0]:
        return lines[start[0] - 1][start[1] : ] + ''.join(lines[start[0] : end[0] - 1]) + lines[end[0] - 1][ : end[1]]
    else:
        return lines[start[0] - 1][start[1] : end[1]]

def parse_lean_file(file, decls, appended_fields, lines=None, debug=False):
    def get_instance_name(ast):
        if not 'children' in ast and ast['kind'] in ['ident', 'notation'] and ast['value']:
            if isinstance(ast['value'], list):
                return ast['value']
            elif isinstance(ast['value'], str):
                return [ast['value']]
            else:
                raise ValueError
        elif ast['children']:
            for i in ast['children']:
                if i is not None:
                    return get_instance_name(i)
            raise ValueError
        else:
            raise ValueError
    
    def folding_ast(all_asts, lines):
        kinds = {}
        for item in all_asts:
            if item is None:
                continue
            # graph.add_node(i, **item)
            # if 'children' in item:
            #     graph.add_edges_from([(i, j) for j in item['children']])
            if 'children' in item:
                # deps.extend([(i, j) for j in item['children']])
                item['children'] = [all_asts[i] for i in item['children'] if i is not None]
            if 'kind' in item:
                if item['kind'] not in kinds:
                    kinds[item['kind']] = []
                kinds[item['kind']].append(item)
        return kinds
    
    def get_before_lines(item, lines):
        return ''.join(lines[ : item['start'][0] - 1]) + lines[item['start'][0] - 1][ : item['start'][1]]
    
    def get_after_lines(item, lines):
        return (lines[item['start'][0] - 1][item['start'][1] : ] + ''.join(lines[item['start'][0] : ]))
    
    def get_content(all_asts, lines, start_pos = [0,0]):
        local_start_pos = None
        local_end_pos = None
        for item in all_asts:
            if item is None or (item['start'] is not None and item['start'] < start_pos):
                continue
            if item['kind'] in ['ident', 'notation', 'nat', 'string']:
                if ('content' not in item or not item['content']) and 'value' in item and item['end'][0] == item['start'][0]:
                    if isinstance(item['value'], list):
                        item['value'] = '.'.join(item['value'])
                    item['value'] = item['value'].replace('\r', '')
                    after_lines = get_after_lines(item, lines)
                    if not after_lines.startswith(item['value']) and not after_lines[1 : ].startswith(item['value']) and repr(item['value'].replace('\r', ''))[1:-1] in after_lines:
                        item['value'] = repr(item['value'])[1:-1]
                    if get_fraction(item['start'], item['end'], lines) != item['value']:
                        if get_after_lines({**item, **{'start' : [item['start'][0], item['start'][1] - 1]}}, lines).startswith(item['value']):
                            item['start'][1] -= 1
                        elif get_after_lines({**item, **{'start' : [item['start'][0], item['start'][1] + 1]}}, lines).startswith(item['value']):
                            item['start'][1] += 1
                    before_lines = get_before_lines(item, lines)
                    after_lines = get_after_lines(item, lines)
                    if (after_lines.startswith(item['value']) or (item['kind'] == 'string' and item['value'] in after_lines)) and (value_end := calculate_end_position(before_lines + after_lines[ : after_lines.find(item['value']) + len(item['value'])], 1, 0)) > item['end']:
                        item['end'] = value_end
                        if item['kind'] == 'string' and not get_fraction(item['start'], item['end'], lines).endswith('"') and (lines[item['end'][0] - 1][item['end'][1] : ] + ''.join(lines[item['end'][0] : ])).startswith('"'):
                            item['end'] = calculate_end_position('"', *item['end'])
            if 'children' in item and any(i for i in item['children'] if i is not None):
                child_start_pos, child_end_pos = get_content(item['children'], lines, start_pos)
                if child_start_pos is None or child_end_pos is None:
                    continue
                item['start'] = min(item['start'], child_start_pos) if 'start' in item else child_start_pos
                item['end'] = max(item['end'], child_end_pos) if 'end' in item else child_end_pos
            if 'start' in item and 'end' in item and item['start'] >= start_pos:
                item['content'] = get_fraction(item['start'], item['end'], lines)
            local_start_pos = item['start'] if local_start_pos is None else min(local_start_pos, item['start'])
            local_end_pos = item['end'] if local_end_pos is None else max(local_end_pos, item['end'])
            # if item['kind'] == 'instance' and 'content' in item and '\n  let N := 10, seq := r.unquot in\n   ' in item['content']:
            #     print()
        return local_start_pos, local_end_pos

    def merge_strings(str1, str2):
        # 检查str1的结束是否与str2的开始匹配，或者str2的结束是否与str1的开始匹配
        for i in range(1, min(len(str1), len(str2))+1):
            if str1[-i:] == str2[:i]:
                return str1 + str2[i:]
            elif str2[-i:] == str1[:i]:
                return str2 + str1[i:]
        # 如果没有找到匹配，抛出一个错误
        raise ValueError("Strings do not appear to be from the same source string")

    
    file = Path(file).resolve()
    if not os.path.exists(file.with_suffix('.ast.json')):
        os.system(' '.join([f_join(leaven_path, 'elan', 'bin', 'lean'), "-M", "20480", "--ast", "--tsast", "--tspp -q ", str(file)]))
    try:
        with open(file.with_suffix('.ast.json'), 'r') as f:
            all_asts = json.load(f)['ast']
    except:
        os.system(' '.join([f_join(leaven_path, 'elan', 'bin', 'lean'), "-M", "20480", "--ast", "--tsast", "--tspp -q ", str(file)]))
        with open(file.with_suffix('.ast.json'), 'r') as f:
            all_asts = json.load(f)['ast']
    if lines is None:
        with open(file, 'r') as f:
            lines = f.readlines()
        if lines[-1].endswith('\n'):
            lines.append('')
    if not debug:
        os.remove(file.with_suffix('.ast.json'))
    file_comment = re.search(r'\/-\s*Copyright(?:[^-]|-[^\/])*-\/', ''.join(lines))
    if file_comment:
        file_comment_span = ''.join(lines)[ : file_comment.span()[1]].split('\n')
        file_comment_span = [len(file_comment_span),len(file_comment_span[-1])]
    else:
        file_comment_span = [0,0]
    kinds = folding_ast(all_asts, lines)
    get_content(all_asts, lines, file_comment_span)
    decl_asts = {}
    # all_decl_names = set()
    # next_comment_regex = re.compile(r'\/-(?:[^-]|-[^\/])*-\/|--[^\n]*\n')

    proving_environment = sorted([(x['start'], x['end'], x['kind'], get_fraction(x['start'], x['end'], lines)) for x in kinds['commands'][0]['children'] if x['start'] >= file_comment_span if x['kind'] in ['namespace', 'end', 'section']])
    
    gathered_asts = []
    mdocs = kinds['mdoc'][0]['value'].replace('\r','').replace('> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.\n> Any changes to this file require a corresponding PR to mathlib4.','') if 'mdoc' in kinds else None

    traced_kinds = ['constant', 'axiom', 'theorem', 'definition', 'structure', 'inductive', 'class', 'abbreviation', 'instance', 'class_inductive', 'run_cmd', 'user_command', 'attribute']

    ast_list = sorted([i for k, v in kinds.items() if k in traced_kinds for i in v if i['start'] >= file_comment_span], key=lambda x : x['start'])
    for i in range(len(ast_list) - 1):
        if ast_list[i]['end'] > ast_list[i+1]['start']:
            ast_list[i]['end'] = ast_list[i+1]['start']
            ast_list[i]['content'] = get_fraction(ast_list[i]['start'], ast_list[i]['end'], lines)
    for ast in ast_list:
        matched_decls = [i for i in decls if ast['start'][0] <= decls[i]['line'] <= (ast['end'][0] if ast['end'][1] > 0 else (ast['end'][0] - 1))]
        # last_decl_content = ''
        for d in matched_decls:
            # last_decl_content = merge_strings(last_decl_content, decls[d]['source']) if last_decl_content else decls[d]['source']
            # assert ast['content'] in last_decl_content
            # assert len([i for i in matched_decls if i not in appended_fields]) == 1
            # if 'source' in decls[d]:
            #     if ast['kind'] == 'attribute' or decl_asts[d]['kind'] == 'attribute':
            #         continue
            #     else:
            #         raise Exception
            if (end_comment := list(re.finditer(r'\/-.*?-\/\s*', ast['content'], re.DOTALL))) and end_comment[-1].end() == len(ast['content']):
                ast['content'] = ast['content'][ : -len(end_comment[-1].group(0))]
                ast['end'] = calculate_end_position(ast['content'], ast['start'][0], ast['start'][1])
            decls[d]['source'] = ast['content']
            decls[d]['line'] = ast['start'][0]
            decls[d]['start'] = ast['start']
            decls[d]['end'] = ast['end']
            decl_asts[d] = ast
        # all_decl_names.append((decl, str(file)))
        # all_decl_names.add(decl)
        if matched_decls and ast not in gathered_asts:
            gathered_asts.append(ast)
    assert not [item for item in decls.items() if 'source' not in item[1]]
    env_asts = [i for i in kinds['commands'][0]['children'] if i['kind'] not in traced_kinds if i['start'] >= file_comment_span]
    # untraced_asts = [x for x in kinds['commands'][0]['children'] if x['start'] >= file_comment_span and (x['kind'] not in ['theorem', 'definition', 'structure', 'inductive', 'class', 'class_inductive', 'abbreviation', 'instance', 'mdoc'] or x not in gathered_asts)]
    decl_blocks = [((1,0),)] + sorted(list(set((tuple(i['start']), tuple(i['end'])) for i in ast_list + env_asts))) + [((len(lines), len(lines[-1])),)]
    proving_environment = sorted([
        (list(decl_blocks[i][-1]), list(decl_blocks[i + 1][0]), m.split()[0], m) 
        for i in range(len(decl_blocks) - 1) 
        if decl_blocks[i][-1] != decl_blocks[i + 1][0] and (m := get_fraction(decl_blocks[i][-1], decl_blocks[i + 1][0], lines)).strip() and ('mdoc' not in kinds or kinds['mdoc'][0]['value'].replace('\r','').strip() not in m)
        ] + [(x['start'], x['end'], x['kind'], get_fraction(x['start'], x['end'], lines)) for x in ast_list + env_asts if x not in gathered_asts])
    return decls, proving_environment, decl_asts, mdocs

# @classmethod
# def get_ast(file, lines=None, build_graph=False):
#     def folding_ast(all_asts, lines, start_pos = [0,0]):
#         kinds = {}

#         for item in all_asts:
#             if not item:
#                 continue
#             # graph.add_node(i, **item)
#             # if 'children' in item:
#             #     graph.add_edges_from([(i, j) for j in item['children']])
#             if 'children' in item:
#                 # deps.extend([(i, j) for j in item['children']])
#                 item['children'] = [all_asts[i] for i in item['children'] if i is not None]
#             if 'kind' in item:
#                 if item['kind'] not in kinds:
#                     kinds[item['kind']] = []
#                 kinds[item['kind']].append(item)
#             if 'start' in item and 'end' in item and item['start'] >= start_pos:
#                 if item['start'][0] < item['end'][0]:
#                     item['content'] = lines[item['start'][0] - 1][item['start'][1] : ] + ''.join(lines[item['start'][0] : item['end'][0] - 1]) + lines[item['end'][0] - 1][ : item['end'][1]]
#                 else:
#                     item['content'] = lines[item['start'][0] - 1][item['start'][1] : item['end'][1]]
#             # else:
#             #     item['content'] = ''
        
#         return kinds
    
#     file = Path(file).resolve()
#     graph = None
#     if not os.path.exists(file.with_suffix('.ast.json')):
#         os.system(' '.join([f_join(leaven_path, 'elan', 'bin', 'lean'), "-M", "20480", "--ast", "--tsast", "--tspp -q ", str(file)]))
#     with open(file.with_suffix('.ast.json'), 'r') as f:
#         all_asts = json.load(f)
#     if lines is None:
#         with open(file, 'r') as f:
#             lines = f.readlines()
#         if lines[-1].endswith('\n'):
#             lines.append('')
#     os.remove(file.with_suffix('.ast.json'))
#     file_comment = re.search(r'\/-\s*Copyright(?:[^-]|-[^\/])*-\/', ''.join(lines))
#     if file_comment:
#         file_comment_span = ''.join(lines)[ : file_comment.span()[1]].split('\n')
#         file_comment_span = [len(file_comment_span),len(file_comment_span[-1])]
#     else:
#         file_comment_span = [0,0]
    
#     kinds = folding_ast(all_asts, lines, file_comment_span)
#     if build_graph:
#         graph = nx.DiGraph()
#         graph.add_node(0, kind=None, start=None, end=None)
#         transverse_ast(kinds['file'], graph, 0, None, None)
#     return kinds, all_asts['ast'], graph

def transverse_ast(ast, graph : nx.DiGraph, parent, start, end):
    for b in ast:
        if b is None:
            continue
        b_start = b['start'] if 'start' in b else None
        b_end = b['end'] if 'end' in b else None
        if b['kind'] in ['file', '#eval', '#reduce', '#check', 'example', 'constants', 'constant', 'variables', 'variable', 'imports', 'definition', 'theorem', 'abbreviation', 'instance', 'structure', 'inductive', 'class', 'class_inductive', 'by', '{', 'tactic', 'begin'] and ((start is None or 'start' not in b or b['start'] > start) or (end is None or 'end' not in b or b['end'] < end)):
            node_name = len(graph.nodes())
            graph.add_node(node_name, kind=b['kind'], start=b_start, end=b_end)
            graph.add_edge(parent, node_name)
            if 'children' in b:
                transverse_ast(b['children'], graph, node_name, b_start, b_end)
        elif 'children' in b:
            transverse_ast(b['children'], graph, parent, b_start, b_end)

def document_probing(file=None, content=None, lean_server=None, do_logging=False):
    if lean_server is None:
        local_lean_server = LeanEnv(cwd=str(Path('.').resolve()), do_logging=do_logging)
    else:
        local_lean_server = lean_server
    if file is not None:
        with open(file, 'r') as f:
            lines = f.readlines()
    elif content is not None:
        lines = re.split(r'(?<=\n)', content)
    else:
        raise ValueError
    if header := re.search(r'\/-\s*Copyright(?:[^-]|-[^\/])*-\/\s*', ''.join(lines), re.DOTALL):
        start_line = header.group(0).count('\n')
    else:
        start_line = 0
    sep_lines = [sorted(list(set([m.end() for m in re.finditer(r'(?:\b|:=|,|;|:|\(|\{|\[|⟨|\.)\s*', line)]))) for line in lines]
    processed_line = {}
    last_ts = ''
    line_buffer = ''
    local_lean_server.reset(options={"filename": str(file)})
    for row, sep in enumerate(sep_lines):
        if row < start_line:
            continue
        line = lines[row]
        row += 1
        last_sep = 0
        if line and not sep:
            line_buffer += line
            continue
        processed_line[row] = {}
        for column in sep:
            if column == last_sep:
                continue
            try:
                ts = local_lean_server.render(options={"filename" : str(file), "line" : row, "col" : column})
                assert ts
            except:
                line_buffer += line[last_sep : column]
                last_sep = column
                continue
            if ts and ts != last_ts:
                processed_line[row][column] = object2Map(ts)
                processed_line[row][column]['precontent'] = line_buffer + line[last_sep : column]
                last_ts = ts
                line_buffer = ''
            else:
                line_buffer += line[last_sep : column]
            last_sep = column
        line_buffer += line[last_sep : ]
    if len(lines) not in processed_line:
        processed_line[len(lines)] = {}
    processed_line[len(lines)][len(lines[-1])] = {'precontent' : line_buffer}
    if lean_server is None:
        local_lean_server.close()
        del local_lean_server
    return processed_line

def get_information(file):
    file = Path(file).resolve()
    processed_line = document_probing(file)
    tactics = []
    commands = []
    lemmas = []
    for l, line in processed_line.items():
        for c, column in line.items():
            if 'source' in column and column['source'] and 'file' in column['source'] and column['source']['file'] and column['source']['file'] != file:
                if 'tactic' in column['source']['file']:
                    tactics.append([l, c, column['text']])
                elif 'interactive' in column['source']['file']:
                    commands.append([l, c, column['text']])
                elif 'full_id' in column and column['full_id']:
                    lemmas.append([l, c, column['full_id']])
    return tactics, commands, lemmas

def get_training_probes(file=None, content=None, lean_server=None, do_logging=False):
    if file is not None:
        file = Path(file).resolve()
        processed_line = document_probing(file=file, lean_server=lean_server, do_logging=do_logging)
    elif content is not None:
        processed_line = document_probing(content=content, lean_server=lean_server, do_logging=do_logging)
    else:
        raise ValueError
    #  or content is not None
    tactic_states = []
    lemmas = []
    for l, line in processed_line.items():
        for c, column in line.items():
            if 'source' in column and column['source']:
                if 'state' in column and column['state'] and column['state'].split('⊢')[0]and 'Type' not in column['state'].split('⊢')[0]:
                    tactic_states.append([l, c, column])
                if 'full_id' in column and column['full_id']:
                    lemmas.append([l, c, column])
    return tactic_states, lemmas

def get_all_info(file=None, content=None, lean_server=None, do_logging=False):
    if file is not None:
        file = Path(file).resolve()
        processed_line = document_probing(file=file, lean_server=lean_server, do_logging=do_logging)
    elif content is not None:
        processed_line = document_probing(content=content, lean_server=lean_server, do_logging=do_logging)
    else:
        raise ValueError
    return processed_line

def position_to_decl(decls, proving_environment, appended_fields, filename, line, column, full_id=None):
    if full_id is not None and full_id in decls:
        return full_id, decls[full_id]
    if result := {
        k : v 
        for k, v in decls.items() 
        if 'start' in v and 'end' in v and v['filename'] == filename and v['start'] <= [line, column] < v['end'] and k not in appended_fields
        }:
        return list(result.items())[0]
    if len(result := {
        f"{kind}_{start_line}_{start_column}_{end_line}_{end_column}" : 
        {'filename': filename, 
            'kind': kind, 
            'start': [start_line, start_column], 
            'end': [end_line, end_column], 
            'source': content} 
            for (start_line, start_column), (end_line, end_column), kind, content in proving_environment 
            if (start_line, start_column) <= (line, column) <= (end_line, end_column)
            }) == 1:
        return list(result.items())[0]
    return None

def get_dependency_graph_within_file(lines, all_decls, file, proving_environment, all_info, appended_fields):
    def get_length_of_substring(lines, start_line, start_column, end_line, end_column):
        if start_line == end_line:
            return end_column - start_column
        else:
            first_line_length = len(lines[start_line - 1]) - start_column
            middle_lines_length = sum(len(lines[i]) for i in range(start_line, end_line - 1))
            last_line_length = end_column
            return first_line_length + middle_lines_length + last_line_length

    filename = cut_path(file)
    file = str(file)
    decls = all_decls[filename]
    graph = nx.MultiDiGraph()
    for decl in decls:
        graph.add_node(decl, **decls[decl])
    for line, _item in all_info.items():
        for column, item in _item.items():
            if 'source' not in item or not item['source']:
                continue
            if item['source']['file'] is not None and item['source']['file'] != file:
                if item['full_id'] in decls and decls[item['full_id']]['start'] <= [item['source']['line'], item['source']['column']] < decls[item['full_id']]['end']:
                    item['source']['file'] = None
                else:
                    continue
            tail_result = position_to_decl(decls, proving_environment, appended_fields, filename, line, column)
            if tail_result is None:
                continue
            else:
                tail, tail_info = tail_result
            head, head_info = position_to_decl(decls, proving_environment, appended_fields, filename, item['source']['line'], item['source']['column'], full_id=item['full_id'])
            if head not in graph.nodes:
                if head in appended_fields:
                    head, head_info = appended_fields(head), decls[appended_fields(head)]
                graph.add_node(head, **head_info)
            if tail not in graph.nodes:
                if tail in appended_fields:
                    tail, tail_info = appended_fields(tail), decls[appended_fields(tail)]
                graph.add_node(tail, **tail_info)
            graph.add_edge(head, tail, pos=(line, column))
            assert graph.nodes[tail]['start'] <= [line, column] < graph.nodes[tail]['end']
    for i, (start, end, tp, content) in enumerate(proving_environment):
        if tp == 'user_command' and (m := re.match(r'\s*open_locale\s+(.*)\s*$', content)):
            tail, tail_info = f"{tp}_{start[0]}_{start[1]}_{end[0]}_{end[1]}", {'filename': filename, 'kind': tp, 'start': start, 'end': end, 'source': content} 
            dep_localized = []
            for j, (start_, end_, tp_, content_) in enumerate(proving_environment[ : i]):
                if tp_ == 'user_command' and (m_ := re.match(r'\s*localized\s+".*?"\s+in\s+(.*)\s*$', content)) and m.group(1) == m_.group(1):
                    dep_localized.append(proving_environment[j])
            for start_, end_, tp_, content_ in dep_localized:
                head, head_info = f"{tp_}_{start_[0]}_{start_[1]}_{end_[0]}_{end_[1]}", {'filename': filename, 'kind': tp_, 'start': start_, 'end': end, 'source': content_}
                graph.add_node(head, **head_info)
                graph.add_node(tail, **tail_info)
                graph.add_edge(head, tail, pos=(*start,))
                assert graph.nodes[tail]['start'] <= start < graph.nodes[tail]['end']
    return graph
