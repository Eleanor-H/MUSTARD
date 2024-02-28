import argparse
import datetime
import json
import logging
import os
import re
import shutil
import sys
import time
from pprint import pprint

import jsonlines
import torch
import yaml
from func_timeout import FunctionTimedOut, func_timeout

from environment_base import FormalSystemEnvironment
from formal_system_client import FormalSystemFatalError
from lean_environment import LeanEnvironment
from search import BestFirstSearch, SearchReturnState

from utils import cal_on_track_info

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.setrecursionlimit(1500)

WORK_DIR = os.environ.get('WORK_DIR')
HERE = os.path.realpath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--devices', type=str, default='0+1',
    help='which devices to use on local machine. Several devices should be divided by `+`.')
parser.add_argument(
    '--base_port', type=int, default=8000,
    help='base port for communicating with flask server'
)
parser.add_argument(
    '--config', type=str, default='configs/eval_atp_lean.yaml',
    help='configuration file for eval search'
)
parser.add_argument(
    '--formal_system', type=str, default='lean'
)
parser.add_argument(
    '--mm_filepath', type=str, default=f'{WORK_DIR}/data/atp/set.mm'
)
parser.add_argument(
    '--model_name_or_path', type=str, default=f'{WORK_DIR}/model_hub/gpt2-large'
)
parser.add_argument(
    '--model_type', type=str, default=f'{WORK_DIR}/model_hub/gpt2-large'
)
parser.add_argument(
    '--dec_names_path', type=str, default=f'{WORK_DIR}/data/mathlib/test.names'
)
parser.add_argument(
    '--pass_at', type=int, default=1,
    help="Number of runs, to compute pass@n.")
parser.add_argument(
    '--search_method', type=str, default='bfs'
)
parser.add_argument(
    '--re_const', type=float, default=1.0
)
parser.add_argument(
    '--const_exp', type=float, default=1.0
)
parser.add_argument(
    "--c_prime", type=float, default=1.0
)
parser.add_argument(
    '--use_max_q', action='store_true',
)
parser.add_argument(
    '--n_success_proof', type=int, default=1
)
parser.add_argument(
    '--prompt_mode', type=int, default=2
)
parser.add_argument("--model_class", default="GPT2LMHeadModel", type=str)
parser.add_argument('--lean_gym_dir', default=f'{WORK_DIR}/package/lean_gym/',
                    type=str, required=False, help='lean-gym 路径')
parser.add_argument('--n_expansion', default=8, type=int,
                    help="number of expansion per node, denoted as `e` in original paper")
parser.add_argument('--max_search_steps', default=128, type=int)
parser.add_argument('--use_skip', action='store_true',
                    help="whether to use skip inference, which generate future subgoal to guide tactic generation")
parser.add_argument('--decoding_method', default='sampling', type=str,
                    help="choose decoding method for tactic generation, choose from `sampling` and `beam_search`")
parser.add_argument('--use_value_function', action='store_true',
                    help="whether to use value function to re-evaluate tactic state")
parser.add_argument('--max_seq_length', default=1024, type=int,
                    help="max sequence length for generation")
parser.add_argument("--value_function_temperature", default=1.0, type=float)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--random_value", action="store_true")
parser.add_argument("--value_function_model_type", default='gpt', type=str)
parser.add_argument("--value_function_model_name_or_path", default="", type=str)
parser.add_argument("--accumulate_log_prob", action='store_true')
parser.add_argument("--disable_global_timeout", action='store_true')
parser.add_argument("--global_timeout_seconds", default=300, type=int)
parser.add_argument("--use_dummy_tactic_generator", action='store_true')

parser.add_argument('--init_method', type=str, default=None, help='data dir on s3')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")

# online training args
parser.add_argument(
    '--online_training', action='store_true', help='Whether use online-training.'
)
parser.add_argument('--online_eval_batchsize', type=int, default=4,
                    help="the number of decl names of each batch for evaluation")
parser.add_argument('--online_eval_num_batch', type=int, default=0,
                    help="the number of batches for evaluation in total. If set to 0 it never ends")
parser.add_argument('--online_batch_dir', type=str, default=f"{WORK_DIR}/output/online_batch",
                    help="the path to decl names of batch data.")
parser.add_argument('--online_expert_dir', type=str, default=f"{WORK_DIR}/cache/online_expert",
                    help="the local path to store files and then transfer to s3")
parser.add_argument("--create_gt_rlhf_data", action="store_true")


def proof_search(rank, environment: FormalSystemEnvironment, dataloader, params, device):
    logger = logging.getLogger('eval_search')
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(f'{WORK_DIR}/debug/debug.log'))

    total_proofs = len(dataloader)
    search_state_counter = {}
    on_track_rates = []
    on_track_reexpansion_rates = []
    for idx, data in enumerate(dataloader):
        # data = ['nat.mod_zero', 'nat']
        logger.info('\n\n[SEARCH] Start proof search')
        # perform search
        proofsearch = None
        proof_start_time, proof_end_time = None, None
        try:
            if params['search_method'] == 'bfs':
                proofsearch = BestFirstSearch(
                    rank=rank,
                    environment=environment,
                    data=data,
                    n_expansion=params['n_expansion'],
                    max_search_steps=params['max_search_steps'],
                    n_retry=params['n_retry'],
                    n_success_proof=params['n_success_proof'],
                    max_proof_steps=params['max_proof_steps'],
                    global_timeout=params['global_timeout'],
                    step_timeout=params['step_timeout'],
                    expert_iter_path=params['expert_iter_path'],
                    use_skip=params['use_skip'],
                    accumulate_log_prob=params['accumulate_log_prob'],
                    create_gt_rlhf_data=params["create_gt_rlhf_data"]
                )
            else:
                raise NotImplementedError("""Search method not implemented""")

            proof_start_time = time.time()
            if params['disable_global_timeout']:
                state, search_result, proof_log = proofsearch.search()
            else:
                state, search_result, proof_log = func_timeout(params['global_timeout_seconds'], proofsearch.search)
            
            # currently, only bfs support rl data generation...
            if params['search_method'] in ["bfs"] and "iteration_id" in params:
                proofsearch.generate_rlhf_data(params['iteration_id'], params['eval_batch_id'])
            proofsearch.on_search_end()
            proof_end_time = time.time()

            # only in the case of init search failed, the proof_log will be none
            if proof_log is not None:
                with jsonlines.open(f"outputs/search_logs/eval_search_shard_{rank}.json", mode='a') as writer:
                    writer.write(proof_log)
        except FunctionTimedOut:
            # currently, only bfs support rl data generation...
            if params['search_method'] in ["bfs"] and "iteration_id" in params:
                proofsearch.generate_rlhf_data(params['iteration_id'], params['eval_batch_id'])
            proofsearch.on_search_end()
            if params['search_method'] in ['bfs', 'bfsre'] and proofsearch is not None and \
                    len(proofsearch.success_nodes) >= 1:
                state, search_result, _ = proofsearch.get_return_state(state=SearchReturnState.GLOBAL_TIMEOUT)
            else:
                state = SearchReturnState.GLOBAL_TIMEOUT
                search_result = {'success': False, 'empty_queue': False, 'max_step_reached': False,
                                 'global_timeout': True, 'init_failed': False, 'truncation_flag': proofsearch.truncation_flag, 
                                 'on_track_rate': -1.0, 'tactics': [], 'decl_nms': data}
        except FormalSystemFatalError:
            environment.reset_formal_system()
            state = SearchReturnState.SYSTEM_ERROR
            search_result = {'success': False, 'empty_queue': False, 'max_step_reached': False, 'global_timeout': False,
                             'on_track_rate': -1.0, 'init_failed': False, 'system_error': True, 'tactics': [], 'decl_nms': data}

        # update counter
        search_state_counter[state] = \
            search_state_counter.get(state, 0) + 1
        acc = search_state_counter.get(SearchReturnState.SUCCESS, 0) / \
              (idx - search_state_counter.get(SearchReturnState.INIT_FAILED, 0) + 0.001)
        if 'on_track_rate' in search_result:
            on_track_rates.append(search_result['on_track_rate'])
        if 'on_track_reexpansion_rate' in search_result:
            on_track_reexpansion_rates.append(search_result['on_track_reexpansion_rate'])
        if proof_start_time is not None and proof_end_time is not None:
            time_expanded = proof_end_time - proof_start_time
            search_result['time'] = time_expanded

        # log progress
        logger.info(f"[{state}] {search_result}")
        logger.info(f"[STATUS] progress: {idx / total_proofs:.3f} acc: {acc}")
        logger.info(f"[STATUS] {search_state_counter}")

        # TODO: print progress
        print(search_result)

    new_stats = torch.tensor([search_state_counter.get(SearchReturnState.SUCCESS, 0),
                              search_state_counter.get(SearchReturnState.EMPTY_QUEUE, 0),
                              search_state_counter.get(SearchReturnState.GLOBAL_TIMEOUT, 0),
                              search_state_counter.get(SearchReturnState.MAX_STEP_REACHED, 0),
                              search_state_counter.get(SearchReturnState.INIT_FAILED, 0),
                              search_state_counter.get(SearchReturnState.SYSTEM_ERROR, 0)
                              ]).to(device)

    on_track_rates = cal_on_track_info(on_track_rates).to(device)
    on_track_reexpansion_rates = cal_on_track_info(on_track_reexpansion_rates).to(device)
    return new_stats, on_track_rates, on_track_reexpansion_rates


def main(params, args):
    print('main.py args:\n' + args.__repr__())

    # initialize distributed environment
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    print(f'device: {device}\targs.rank: {args.rank}\t args.world_size: {args.world_size}')
    print(f'init_method: {args.init_method}')

    # time out is set to 10 hour
    if args.world_size >= 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', world_size=args.world_size,
                                             rank=args.rank, init_method=args.init_method,
                                             timeout=datetime.timedelta(seconds=36000))

    # initialize logger
    log_dir = os.path.join(HERE, "outputs/search_logs/")
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, f"eval_search_shard_{args.rank}.out")
    # logger = setup_log(filepath, "eval_search")

    # create folder for expert iteration
    if args.rank == 0:
        if os.path.exists(params['expert_iter_path']):
            shutil.rmtree(params['expert_iter_path'])
        os.makedirs(params['expert_iter_path'])

    # initialized environment
    environment = None
    port = args.base_port + args.rank
    if args.formal_system in ['lean', 'new-lean']:
        environment = LeanEnvironment(try_intros=params['try_intros'],
                                      prompt_mode=params['prompt_mode'],
                                      use_value_function=params['use_value_function'],
                                      server_port=port,
                                      random_value=params['random_value'],
                                      pass_at=params['pass_at'])
    else:
        raise NotImplementedError(f"""
            The formal system "{args.formal_system}" has not yet been implemented
        """)
    assert environment is not None
    environment.initialized_environment(params, device)
    if args.online_training:
        environment.reset_formal_system()
    dataloader = environment.setup_data(params, args.rank, args.world_size)

    print(f" > Eval search on rank {args.rank}")
    print(f"    > decl name path: {params['dec_names_path']}")
    print(f"    > model name or path: {params['model_name_or_path']}")
    print(f"    > model type: {params['model_type']}")
    print(f"    > model class: {params['model_class']}")
    print(f"    > length of dataloader: {len(dataloader)}")


    result_queue = torch.zeros(6).to(device)
    print(f'Starting `{params["search_method"]}` search')
    result_queue, on_track_rate, on_track_reexpansion_rate = \
        proof_search(args.rank, environment, dataloader, params, device)
    torch.distributed.barrier()

    print(f'result_queue: {result_queue}')
    print('gather results!!!')
    torch.distributed.all_reduce(result_queue, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(on_track_rate, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(on_track_reexpansion_rate, op=torch.distributed.ReduceOp.SUM)
    total_stats = None
    if args.local_rank == 0:
        print(f'result_queue: {result_queue}')
        result_queue = result_queue.cpu()
        total_stats = {SearchReturnState.SUCCESS: result_queue[0].item(),
                       SearchReturnState.EMPTY_QUEUE: result_queue[1].item(),
                       SearchReturnState.GLOBAL_TIMEOUT: result_queue[2].item(),
                       SearchReturnState.MAX_STEP_REACHED: result_queue[3].item(),
                       SearchReturnState.INIT_FAILED: result_queue[4].item(),
                       SearchReturnState.SYSTEM_ERROR: result_queue[5].item()}
        ks = [k for k in total_stats]
        n_proofs = 0
        for k in ks:
            n_proofs += total_stats[k]
        total_stats['n_proofs'] = n_proofs

        # Calculate rates
        for k in ks:
            total_stats['rate of {}'.format(k)] = total_stats[k] / n_proofs

        total_stats["pass_rate"] = total_stats[SearchReturnState.SUCCESS] / (
                total_stats['n_proofs'] - total_stats[SearchReturnState.INIT_FAILED] + 1e-5)
        print("Total stats:", total_stats)

        if on_track_rate[1].item() != 0:
            print(f"on track rates: {on_track_rate[2].item()/on_track_rate[1].item()}, "
                  f"no full correct on track rates: "
                  f"{on_track_rate[3].item()/(1+on_track_rate[1].item()-on_track_rate[0].item())}, "
                  f"full correct {on_track_rate[0].item()}/{on_track_rate[1].item()} = "
                  f"{on_track_rate[0].item()/on_track_rate[1].item()}")
        if on_track_reexpansion_rate[1].item() != 0:
            print(f"on track re_expansion rates: {on_track_reexpansion_rate[2].item() / on_track_reexpansion_rate[1].item()}, "
                  f"no full correct on track re_expansion rates: "
                  f"{on_track_reexpansion_rate[3].item()/(1+on_track_reexpansion_rate[1].item()-on_track_reexpansion_rate[0].item())}, "
                  f"full correct re_expansion rates {on_track_reexpansion_rate[0].item()}/{on_track_reexpansion_rate[1].item()} = "
                  f"{on_track_reexpansion_rate[0].item() / on_track_reexpansion_rate[1].item()}")

        result_name = datetime.datetime.now().strftime(
            "eval_search_{}_e{}_d{}_%m%d_%H%M.txt".format(params['model_name_or_path'], params['n_expansion'],
                                                          params['max_search_steps']))
        result_dir = 'outputs/search_logs/'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        path = os.path.abspath(os.path.join(result_dir, result_name))
        with open(path, 'w') as f_res:
            f_res.write(str(params))
            f_res.write('\n\n')
            f_res.write(str(total_stats))
        print("[+] Saved results to {}.".format(path))

    torch.distributed.barrier()
    return total_stats


def env_var_constructor(loader, node):
    value = loader.construct_scalar(node) 
    env_var = re.findall(r'\$\{(.*?)\}', value)[0]
    return value.replace(f'${{{env_var}}}', os.environ[env_var])


if __name__ == '__main__':
    args, unparsed = parser.parse_known_args()
    yaml.SafeLoader.add_constructor('!env', env_var_constructor)
    with open(args.config, 'r', encoding='utf8') as stream:
        params = yaml.safe_load(stream)

    # replace arguments
    for arg in vars(args):
        if arg in params:
            params[arg] = getattr(args, arg)

    pprint(params)

    main(params, args)
    exit(0)
