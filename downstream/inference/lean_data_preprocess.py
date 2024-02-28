import json
import random
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

GAPMAP = list("0ABCDEFGHIJKLMN")

def check_long_tail_data(data_file):
    def splitandstrip(x):
        x = x.split()
        x = [re.sub("[\(\)\[\]⟩⟨\{\},;]","", i) for i in x]
        return x

    df = pd.read_csv(data_file)
    df['splited_tactic'] = df['human_tactic_code'].apply(splitandstrip)
    train_tactic = df[df['split'] == 'train']['splited_tactic']
    test_tactic = df[df['split'] == 'test']['splited_tactic']
    train_sig = {}
    test_sig = {}
    for line in tqdm(train_tactic):
        for item in line:
            train_sig[item] = train_sig.get(item, 0) + 1
    for line in tqdm(test_tactic):
        for item in line:
            test_sig[item] = test_sig.get(item, 0) + 1
    train_df = pd.DataFrame({'keywords': train_sig.keys(), 'train_counts': train_sig.values()})
    test_df = pd.DataFrame({'keywords': test_sig.keys(), 'test_counts': test_sig.values()})
    out_df = test_df.merge(train_df, how='outer', on='keywords')
    out_df = out_df.fillna(0)
    out_df['train_counts'] = out_df['train_counts'].astype(int)
    out_df['test_counts'] = out_df['test_counts'].astype(int)
    out_df = out_df.sort_values('test_counts')
    out_df.to_csv('out.csv', index=None)

    with open('ll.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    declnames = []
    df = df[df['split'] == 'test'][['decl_name', 'human_tactic_code']]
    for idx, row in tqdm(df.iterrows()):
        for line in lines:
            if line in row['human_tactic_code']:
                declnames.append(row['decl_name'])
    declnames = list(set(declnames))
    for d in declnames:
        print(d)
    
    # train_tactic = train_tactic.apply(splitandstrip)
    # test_tactic = test_tactic
        


def create_training_line(decl_nm, head, end, tgt, gap):
    head = head.strip().replace('\n', ' ').replace('\t', ' ')
    end = end.strip().replace('\n', ' ').replace('\t', ' ')
    tgt = tgt.strip().replace('\n', ' ').replace('\t', ' ')
    skip_line =  f"DEC {decl_nm} GOAL {head} SKIP{GAPMAP[gap]} {end}"
    train_line = f"DEC {decl_nm} SKIP{GAPMAP[gap]} {end} GOAL {head} PROOFSTEP {tgt}"
    return skip_line, train_line


def create_skip_data(input_index, input_tgt, output_skip, output_tactic_skip, max_gap=3):
    output_skip = open(output_skip, 'w', encoding='utf8')
    output_tactic_skip = open(output_tactic_skip, 'w', encoding='utf8')

    with open(input_index, 'r', encoding='utf8') as input:
        lines = input.readlines()
    with open(input_tgt, 'r', encoding='utf8') as input:
        tgts = input.readlines()

    new_lines = []
    for line in lines:
        # print(line.strip())
        line = line.strip().replace('\t', ' ').replace('\n', ' ')
        src = line[9:line.index('decl_nm')-4]
        decl_nms = line[line.index('decl_nm')+11:-1]
        new_line = {'src': src, 'decl_nm':decl_nms}
        new_lines.append(new_line)

    lines = new_lines

    current_decl_nm = lines[0]['decl_nm']
    total_idx = 0
    while total_idx < len(lines):
        print(f'\r{total_idx}', end='')
        # collect current_decl_nm
        current_set = []
        while total_idx<len(lines) and lines[total_idx]['decl_nm'] == current_decl_nm:
            lines[total_idx]['tgt'] = tgts[total_idx].strip()
            current_set.append(lines[total_idx])
            total_idx += 1

        for gap in range(1, max_gap+1):
            for idx, head in enumerate(current_set):
                if idx + gap > len(current_set):
                    continue
                head_src = head['src']
                head_tgt = head['tgt']
                if idx + gap == len(current_set):
                    end_src = 'no goals'
                else:
                    end_src = current_set[idx+gap]['src']
                skip_line, train_line = create_training_line(current_decl_nm, head_src, end_src, head_tgt, gap)
                output_skip.write(skip_line.strip() + '\n')
                output_tactic_skip.write(train_line.strip() + '\n')

        # end loop
        if total_idx < len(lines):
            current_decl_nm = lines[total_idx]['decl_nm']

def mix_file(data_paths, out_file):
    all_lines = []
    for data_path in data_paths:
        with open(data_path, 'r', encoding='utf8') as f:
            print('reading lines')
            all_lines.extend(f.readlines())
    random.shuffle(all_lines)
    lines = all_lines
    with open(out_file, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line)

def process_expert_iteration_data(folder, out_file):
    proofsize = []
    proofstep = []
    for filename in glob.iglob(folder + '**/*.csv', recursive=True):
        df = pd.read_csv(filename, encoding='utf8')
        if 'proofsize' in df.keys():
            proofsize.append(df)
        else:
            proofstep.append(df)
    proofsize = pd.concat(proofsize).drop_duplicates()
    proofstep = pd.concat(proofstep).drop_duplicates()

    out = open(out_file, 'w', encoding='utf8')
    lines_a, lines_b_k = [], []
    for _, row in tqdm(proofsize.iterrows()):
        decl_nm = row['decl_name']
        goal = row['goal'].replace('\t', ' ').replace('\n', ' ')
        proof_size = row['proofsize']
        if proof_size == 'A':
            lines_a.append((decl_nm, goal, proof_size))
        else:
            lines_b_k.append((decl_nm, goal, proof_size))
    print("lines a:", len(lines_a), "lines b ~ k", len(lines_b_k))
    lines_a = random.sample(lines_a, len(lines_b_k))
    lines = lines_a + lines_b_k
    for line in lines:
        decl_nm, goal, proof_size = line
        printline = f"DEC {decl_nm} GOAL {goal} PROOFSIZE {proof_size}"
        out.write(printline + '\n')

    for _, row in tqdm(proofstep.iterrows()):
        decl_nm = row['decl_name']
        goal = row['goal'].replace('\t', ' ').replace('\n', ' ')
        proof_step = row['proofstep'].replace('\t', ' ').replace('\n', ' ')
        line = f"DEC {decl_nm} GOAL {goal} PROOFSTEP {proof_step}"
        out.write(line + '\n')
    out.close()

def how_much_more(dir='local_testing/proof_step_multi_path'):
    for filename in os.listdir(dir):
        if 'proofsize' in filename:
            continue
        filepath = os.path.join(dir, filename)
        df = pd.read_csv(filepath, encoding='utf8')
        adds = []
        current_decl_name = "---"
        acc = 0
        current_set = set()
        add_set = set()
        flag_first = True
        for idx, row in df.iterrows():
            if row['decl_name'] == '---' and df.loc[idx+1, 'decl_name'] == current_decl_name:
                flag_first = False
                continue
            elif row['decl_name'] == '---':
                continue
            if current_decl_name != row['decl_name']:
                current_decl_name = row['decl_name']
                adds.append(len(add_set))
                add_set = set()
                flag_first = True
                current_set.add(row['goal'])
            else:
                if flag_first:
                    current_set.add(row['goal'])
                else:
                    if not row['goal'] in current_set:
                        add_set.add(row['goal'])
        print(f"{filepath} n_decl_names {df['decl_name'].nunique()} lenadds: {len(adds)} adds: {sum(adds)} basic: {len(current_set)}")


def get_dir(filepath, mode='from_root'):
    """Helper function for applying transformations to pandas data frames."""
    if filepath.endswith(os.sep):
        filepath = filepath[:-len(os.sep)]
    folders = filepath.split(os.sep)
    if mode == 'parent':
        return os.path.join(*(folders[:-1]))
    elif mode == 'from_root':
        return os.path.join(*(folders[:3]))

def process_proofstep_by_domain(
        data_and_metadata_path='data/lean/original/mathlib/data_and_metadata.csv',
        output_dir="data/lean/processed/by_domain/",
        with_decl_name=False,
        ):
    df = pd.read_csv(data_and_metadata_path)
    df = df[~df['cleaned_goal'].isna()]
    df['dir'] = df['filename'].apply(get_dir)
    gb = df.groupby('dir')
    for gn in gb.groups:
        g = gb.get_group(gn)
        data_dir = os.path.join(output_dir, gn)
        os.makedirs(data_dir, exist_ok=True)
        train_file = open(os.path.join(data_dir, 'train.txt'), 'w')
        valid_file = open(os.path.join(data_dir, 'valid.txt'), 'w')
        test_file = open(os.path.join(data_dir, 'test.txt'), 'w')
        train_names = open(os.path.join(data_dir, 'train.names'), 'w')
        valid_names = open(os.path.join(data_dir, 'valid.names'), 'w')
        test_names = open(os.path.join(data_dir, 'test.names'), 'w')
        for _, row in tqdm(g.iterrows()):
            goal = row['cleaned_goal'].replace('\n', ' ').replace('\t',' ').strip()
            tactic = row['human_tactic_code'].replace('\n', ' ').replace('\t',' ').strip()
            dec_name = row['decl_name'].replace('\n', ' ').replace('\t',' ').strip()
            namespaces = str(row['open_namespaces']).replace('\n', ' ').replace('\t',' ').strip()

            line = 'GOAL ' + goal + ' PROOFSTEP ' + tactic
            if with_decl_name:
                line = 'DEC ' + dec_name + ' ' + line

            names_line = dec_name + ' ' + namespaces
            
            if row['split'] == 'train':
                train_file.write(line + '\n')
                train_names.write(names_line + '\n')
            elif row['split'] == 'test':
                test_file.write(line + '\n')
                test_names.write(names_line + '\n')
            elif row['split'] == 'valid':
                valid_file.write(line + '\n')
                valid_names.write(names_line + '\n')
            else:
                print('not in split!')
        
        for f in [train_file, valid_file, test_file, 
                  train_names, valid_names, test_names]:
            f.close()
        


def process_proofstep_data(data_and_metadata_path='data/lean/original/mathlib/data_and_metadata.csv',
        with_decl_name=False):
    data_df = pd.read_csv(data_and_metadata_path)
    data_df = data_df[~data_df['goal_pp'].isna()]
    train_file = open('/cache/lean_tactic/raw/train.txt', 'w')
    valid_file = open('/cache/lean_tactic/raw/valid.txt', 'w')
    test_file = open('/cache/lean_tactic/raw/test.txt', 'w')
    for _, row in tqdm(data_df.iterrows()):
        goal = row['goal_pp'].replace('\n', ' ').replace('\t',' ').strip()
        tactic = row['human_tactic_code'].replace('\n', ' ').replace('\t',' ').strip()
        dec_name = row['decl_name'].replace('\n', ' ').replace('\t',' ').strip()

        line = 'GOAL ' + goal + ' PROOFSTEP ' + tactic
        if with_decl_name:
            line = 'DEC ' + dec_name + ' ' + line
        
        if row['split'] == 'train':
            train_file.write(line + '\n')
        elif row['split'] == 'test':
            test_file.write(line + '\n')
        elif row['split'] == 'valid':
            valid_file.write(line + '\n')
        else:
            print('not in split!')
    for f in [train_file, valid_file, test_file]:
        f.close()

def gen_pact(input_folder, output_file):
    out = open(output_file, 'w', encoding='utf8')

    for file in os.listdir(input_folder):
        filepath = os.path.join(input_folder, file)
        f1 = open(filepath,'r', encoding='utf8')
        print(filepath)
        for line in tqdm(f1):
            line = json.loads(line)
            raw_text = pact_patten_matching(list(line.keys()), line)
            out.write(raw_text + '\n')
    
    out.close()


def pact_patten_matching(keys, example):
    # skip_proof.json
    if keys[0] == 'skip_proof' and keys[1] == 'proof_term':
        raw_text = f"RESULT {example['skip_proof']} SKIPPROOF {example['proof_term']}"

    # next lemma prediction
    elif keys[0] == 'goal' and keys[1] == 'next_lemma':
        raw_text = f"GOAL {example['goal']} NEXTLEMMA apply ({example['next_lemma']})"

    # premise_classification
    elif keys[0] == 'goal' and keys[1] == 'classify_premise':
        raw_text = f"GOAL {example['goal']} CLASSIFYPREMISE {example['classify_premise']}"

    # proof_step_classification -> local context classification
    elif keys[0] == 'goal' and keys[1] == 'classify_locals':
        raw_text = f"GOAL {example['goal']} CLASSIFYLOCALS {example['classify_locals']}"

    # proof_term_elab
    elif keys[0] == 'proof_term' and keys[1] == 'elab_proof_term':
        raw_text = f"PROOFTERM {example['proof_term']} ELABPROOFTERM {example['elab_proof_term']}"

    # proof_term_prediction
    elif keys[0] == 'goal' and keys[1] == 'proof_term':
        raw_text = f"GOAL {example['goal']} PROOFTERM exact ({example['proof_term']})"

    # result_elab
    elif keys[0] == 'result' and keys[1] == 'result_elab':
        raw_text = f"RESULT {example['result']} ELABRESULT {example['result_elab']}"

    # theorem_name_prediction
    elif keys[0] == 'type' and keys[1] == 'name':
        raw_text = f"TYPE {example['type']} NAME {example['name']}"

    # ts_elab.json
    elif keys[0] == 'goal' and keys[1] == 'elab_goal':
        raw_text = f"GOAL {example['goal']} ELABGOAL {example['elab_goal']}"

    # type_prediction
    elif keys[0] == 'skip_proof' and keys[1] == 'goal':
        raw_text = f"RESULT {example['skip_proof']} PREDICTTYPE {example['goal']}"
    else:
        raise NotImplementedError()

    return raw_text

def process_proof_size(df_path, output_path, additional_data_path=None):
    df = pd.read_csv(df_path)
    df['label'] = df['proofsize'] != -1
    df = df.drop(columns=['proofstep', 'proofsize'])
    if additional_data_path is not None:
        df_add = pd.read_csv(additional_data_path)
        df = pd.concat([df, df_add])

    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)

def process_proof_step(df_path, output_path, additional_data_path=None):
    df = pd.read_csv(df_path)
    df = df.drop(columns=['proofsize'])
    if additional_data_path is not None:
        df_add = pd.read_csv(additional_data_path)
        df = pd.concat([df, df_add])
    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)

def from_yuanye_to_csv(path, out_path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split("#####") for line in lines]
    goal = [line[0] for line in lines]
    proofstep = [line[1] for line in lines]
    df = pd.DataFrame({'goal':goal,'proofstep':proofstep})
    df.to_csv(out_path, index=False)

def merge_train_data(ori_path="data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.csv",
        other_path="data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_processed_cleaned.csv",
        output_path="data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged.csv",
        upsampling_multipliers=(1,1)):
    df_ori = pd.read_csv(ori_path)

    df_proofstep = pd.read_csv(other_path)
    df_proofstep = df_proofstep.drop(columns=['decl_name'])

    um_ori, um_other = upsampling_multipliers
    df_proofstep = pd.concat([df_ori] * um_ori + [df_proofstep] * um_other)
    df_proofstep.to_csv(output_path)

def gen_gpt_train_data(proofstep_path, proofsize_path, out_path):
    df_proofstep = pd.read_csv(proofstep_path)
    df_proofsize = pd.read_csv(proofsize_path)
    out_file = open(out_path, 'w')
    for idx, row in df_proofstep.iterrows():
        out_line = f"GOAL {row['goal']} PROOFSTEP {row['proofstep']}"
        out_file.write(out_line + '\n')
    for idx, row in df_proofsize.iterrows():
        out_line = f"GOAL {row['goal']} PROOFSIZE {row['label']}"
        out_file.write(out_line + '\n')
    out_file.close()

def gen_gpt_train_data_for_matchup(proofstep_path, out_path, 
        with_error_msg=False,
        with_result_msg=False):
    df_proofstep = pd.read_csv(proofstep_path)
    out_file = open(out_path, 'w')
    for idx, row in df_proofstep.iterrows():
        ps = f"PROOFSTEP {row['proofstep']}"
        ps = f"GOAL {row['goal']} " + ps
        if with_error_msg:
            if not 'error' in row or pd.isna(row['error']):
                msg = None
            else:
                msg = row['error']
            ps = f"ERROR {msg} " + ps
        if with_result_msg:
            if not 'result' in row or pd.isna(row['result']):
                res = 'SUCCESS'
            else:
                res = row['result']
            ps = f"RESULT {res} " + ps
        out_line = ps
        out_file.write(out_line + '\n')
    out_file.close()

def clean_data():
    df = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/proofsize_train_processed_final.csv")
    test_decl_names = []
    with open("data/lean/original/mathlib/train2.names") as f:
        lines = f.readlines()
        test_decl_names = [line.split()[0] for line in lines]
    test_decl_names = set(test_decl_names)
    df = df[df['decl_name'].isin(test_decl_names)]
    df.to_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/proofsize_train_processed_cleaned.csv")

def gen_validation_data():
    # part 1
    src = []
    tgt = []
    with open("data/lean/original/mathlib/filtered_training_data/filtered_valid.src", 'r') as f:
        src = f.readlines()
    with open("data/lean/original/mathlib/filtered_training_data/filtered_valid.tgt", 'r') as f:
        tgt = f.readlines()

    with open("data/lean/processed/valid_lean_filtered.txt", 'w') as f:
        for i in range(len(src)):
            line = f"GOAL {src[i].strip()} PROOFSTEP {tgt[i].strip()}"
            f.write(line + '\n')

    # part 2
    with open("data/lean/original/mathlib/valid2.names", 'r') as f:
        valid_names = f.readlines()
        valid_names = [vn.split()[0] for vn in valid_names]

    df = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofstep_valid.csv")
    df = df[df['decl_name'].isin(valid_names)]
    df = df.drop(columns=['decl_name', 'proofsize'])
    df = df.drop_duplicates()
    with open("data/lean/processed/valid_lean_exp.txt", 'w') as f:
        for _, row in df.iterrows():
            line = f"GOAL {row['goal']} PROOFSTEP {row['proofstep']}"
            f.write(line + '\n')

    # with open("data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.txt", 'r') as f:
    #     train_lines = f.readlines()
    # random.shuffle(train_lines)

    # train_data = train_lines[:int(len(train_lines) * 0.01)]
    # added_valid_data = train_lines[int(len(train_lines) * 0.01):]


def data_to_df(filepath):
    pattern = r"DEC (\S*) RESULT (\S*) ERROR (.*) GOAL (.*) PROOFSTEP (.*)"
    decl_names = []
    results = []
    errors = []
    goals = []
    proofsteps = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            s = re.search(pattern, line)
            if not s is None:
                decl_name = s.group(1)
                result = s.group(2)
                error = s.group(3)
                goal = s.group(4)
                proofstep = s.group(5)
                decl_names.append(decl_name)
                results.append(result)
                errors.append(error)
                goals.append(goal)
                proofsteps.append(proofstep)
    df = pd.DataFrame({
        'decl_name': decl_names,
        'result': results,
        'error': errors,
        'goal': goals,
        'proofstep': proofsteps
    })
    return df


def process_expert_iter_data():
    # df_main = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofsize.csv")
    # df_main2 = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofsize (1).csv")
    # with open("data/lean/original/mathlib/train2.names") as f:
    #     lines = f.readlines()
    #     train_decl_names = [line.split()[0] for line in lines]
    # train_decl_names = set(train_decl_names)
    # df_main = df_main[df_main['decl_name'].isin(train_decl_names)]
    # df_main2 = df_main2[df_main2['decl_name'].isin(train_decl_names)]
    # df = pd.concat([df_main, df_main2])
    # df.to_csv("Mainprocess_proofsize_train.csv")

    df = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofsize_train.csv")
    decl_names = df['decl_name'].unique().tolist()

    results = []
    for ix, name in tqdm(enumerate(decl_names), total=len(decl_names)):
        theorem_df = df[df['decl_name'] == name]
        # goals = theorem_df[theorem_df['proofsize'] >= 1]['goal'].value_counts()

        trajectories = []
        cur_traj = []
        for idx, row in theorem_df.iterrows():
            if row['proofsize'] == -1:
                continue
            if row['proofsize'] == 1:
                if len(cur_traj) > 0:
                    cur_traj.reverse()
                    trajectories.append(cur_traj)
                cur_traj = [row]
            else:
                cur_traj.append(row)
        if len(cur_traj) > 0:
            cur_traj.reverse()
            trajectories.append(cur_traj)
        assert len(set([t[0]['goal'] for t in trajectories])) == 1

        dedup_traj = set()
        dedup_trajectories = []
        for t in trajectories:
            traj = []
            for step in t:
                traj.append(f"GOAL {step['goal']} PROOFSTEP {step['proofstep']} ")
            traj = " ".join(traj)
            if traj not in dedup_traj:
                dedup_traj.add(traj)
                dedup_trajectories.append(t)
        # print(dedup_trajectories)
        results.extend(sum(dedup_trajectories, []))

        # # get the shortest answer
        # shortest_length = 100000
        # shortest_solution = None
        # shortest_text = None
        # for t in dedup_trajectories:
        #     tactics = [state["proofstep"] for state in t]
        #     solution = "\n".join(tactics)
        #     # print(len(solution))
        #     if len(solution) < shortest_length:
        #         shortest_length = len(solution)
        #         shortest_solution = t
        #         shortest_text = solution
        #
        # # print(f"Shortest solution: \n{shortest_text}\n =============\n\n")
        #
        # results.extend(shortest_solution)

    pd.DataFrame(results).to_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/allproofs.csv",
                                 index=False,
                                 encoding='utf8')

        # print(trajectories)

def merge_with_ext5_twopath():
    ext5_twopath = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.csv")
    merge_proofs = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/allproofs.csv")
    merge_proofs = merge_proofs[["goal", "proofstep"]]
    all_data = pd.concat([merge_proofs, ext5_twopath])
    all_data = all_data.drop_duplicates()

    out_file = open("data/lean/original/mathlib/expert_iteration/expert_iter7/all_paths_ext_5_train.txt", 'w')
    for idx, row in all_data.iterrows():
        out_line = f"GOAL {row['goal']} PROOFSTEP {row['proofstep']}"
        out_file.write(out_line + '\n')

    # print(ext5_twopath)


def fix_long_tail():
    all_correct_train_names = []
    with open("data/lean/original/mathlib/filtered_training_data/filtered_train.names") as f:
        all_correct_train_names = f.readlines()
        all_correct_train_names = [line.split()[0] for line in all_correct_train_names]

    df = pd.read_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofsize_train.csv")
    decl_names = df['decl_name'].unique().tolist()
    print(all_correct_train_names)
    left_theorem = set(all_correct_train_names) - set(decl_names)
    #
    # # theorem level
    # with open("data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.txt") as f:
    #     lines = f.readlines()
    # for line in lines:
    #     goal, proofstep = line.split("PROOFSTEP")
    #     goal = goal[len("GOAL"):].strip()
    #     proofstep = proofstep.strip()







# match_proofstep_data()
# process_expert_iteration_data('data/lean/original/mathlib/expert_iteration/', 'data/lean/processed/expiter.txt')
# process_proofstep_data()
# gen_pact('/cache/outputs/mix2/', '/cache/outputs/mix2.txt')
# create_skip_data('data/lean/original/mathlib/filtered_training_data/filtered_train.index', 'data/lean/original/mathlib/filtered_training_data/filtered_train.tgt','/cache/outputs/skip.txt', '/cache/outputs/tactic_skip.txt')
# mix_file(['data/lean/processed/tactic.txt', 'data/lean/processed/expiter.txt'], '/cache/outputs/tactic_expiter_4.txt')

# dataset = 'train'
# proccess_src_and_tgt(f'{dataset}.src', f'{dataset}.tgt', f'data/{dataset}.txt')

# dataset = 'valid'
# proccess_src_and_tgt(f'{dataset}.src', f'{dataset}.tgt', f'data/{dataset}.txt')

# dataset = 'test'
# proccess_src_and_tgt(f'{dataset}.src', f'{dataset}.tgt', f'data/{dataset}.txt')


# os.system('cp datasets/cleaned_training_data/test.names data/test.names')
# os.system('cp datasets/cleaned_training_data/valid.names data/valid.names')


# how_much_more()
# check_long_tail_data('data/lean/original/mathlib/data_and_metadata.csv')

if __name__ == '__main__':
    # gen_validation_data()
    # fix_long_tail()
    # clean_data()
    # process_expert_iter_data()
    # merge_train_data()
    process_proofstep_data("/cache/new_data_and_metadata-230910.csv",False)
    # merge_with_ext5_twopath()
    # gen_gpt_train_data_for_matchup(proofstep_path="data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.csv",
    #                out_path="data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.txt")
    # gen_gpt_train_data(proofstep_path="data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged.csv",
    #                    proofsize_path="data/lean/original/mathlib/expert_iteration/expert_iter7/proofsize_train_processed_cleaned.csv",
    #                    out_path="data/lean/original/mathlib/expert_iteration/expert_iter7/ext6_train_w_proofsize.txt")
    # gen_gpt_train_data()
    # from_yuanye_to_csv("data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.data",
    #                    "data/lean/original/mathlib/expert_iteration/expert_iter7/ext_5_train_twopath.csv")
    # process_proof_step("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofstep (1).csv",
    #                    output_path='data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_processed_final.csv',
    #                    additional_data_path="data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_processed.csv")
    # process_proof_step("data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train.csv",
    #                    output_path='data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_processed.csv')


    # from_yuanye_to_csv("/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath.data",
    #                    "/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath.csv")
    # yuanye_path = "/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath.data"
    # gpt_path = "/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath.txt"
    # csv_path = "/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath.csv"
    # csv_path = "/cache/bucket-multimodal/haiming/hwatp/data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged.csv"
    # gpt_path = "/cache/bucket-multimodal/haiming/hwatp/data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged.txt"
    # gen_gpt_train_data_for_matchup(csv_path, gpt_path)
    # gpt_error_result_path = "/cache/bucket-multimodal/haiming/hwatp/data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged_ei5_error_result.txt"

    # gpt_error_result_path = "/cache/bucket-multimodal/haiming/hwatp/data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged_ei5_error_result.txt"
    # gen_gpt_train_data_for_matchup(csv_path, gpt_error_result_path, with_error_msg=True, with_result_msg=True)
    # process_proof_step("data/lean/original/mathlib/expert_iteration/expert_iter7/MainProcess_proofstep (1).csv",
    #                    output_path='data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_processed_final.csv',
    #                    additional_data_path="data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_processed.csv")
    # process_proof_step("/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath.csv",
    #                    output_path='/cache/bucket-multimodal/yuanye/to_haiming/ext_5_train_twopath_processed.csv')

    # yuanye_test_error_result_path = "/home/ma-user/work/zhengying/datasets/lean/error_message/train-error-result.txt"
    # df = data_to_df(yuanye_test_error_result_path)
    # df.to_csv('test-error-result.csv')
    # print(df)

    # ori_path = "/cache/bucket-multimodal/haiming/hwatp/data/lean/original/mathlib/expert_iteration/expert_iter7/proofstep_train_merged.csv"
    # other_path = "/home/ma-user/work/zhengying/hwatp/src/utils/test-error-result.csv"
    # output_path = "ei5-test-error-result-merged.csv"
    # merge_train_data(ori_path, other_path, output_path,
    #     upsampling_multipliers=(1,1))

    # csv_path = "ei5-test-error-result-merged.csv"
    # gpt_error_result_path = "ei5-test-error-result-merged.txt"
    # gen_gpt_train_data_for_matchup(csv_path, gpt_error_result_path, with_error_msg=True, with_result_msg=True)

    # data_and_metadata_path = "/cache/bucket-multimodal/haiming/hwatp/data/lean/original/mathlib/data_and_metadata.csv"
    # # process_proofstep_data(data_and_metadata_path)

    # process_proofstep_by_domain(data_and_metadata_path)