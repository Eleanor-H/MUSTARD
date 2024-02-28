import argparse
import json
import operator
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, request
from func_timeout import FunctionTimedOut, func_set_timeout

app = Flask(__name__)

LEAN_PATH = os.path.expanduser('~/.elan/bin/lean')
LEAN_GYM_DIR = None


class LeanFatalErrorServer(Exception):
    """Raise when Lean fatal error ocurred"""
    pass


class LeanServer(object):
    def __init__(self,
                 lean_path=LEAN_PATH,
                 lean_gym_dir=LEAN_GYM_DIR,
                 normalize_tab=True):
        if lean_path == 'lean':
            lean_path = LEAN_PATH
        self.proc = subprocess.Popen([lean_path, '--run', 'src/repl.lean'],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     cwd=lean_gym_dir)
        self.normalize_tab = normalize_tab

    @func_set_timeout(120)
    def init_search(self, dec_name, namespaces=""):
        inputs = f'["init_search", ["{dec_name}","{namespaces}"]]\n'
        return self.__run(inputs)

    @func_set_timeout(60)
    def run_tac(self, search_id, tactic_id, tactic):
        tactic = "".join(re.findall("[^\n\t\a\b\r]+",tactic))
        inputs = f'["run_tac", ["{search_id}", "{tactic_id}", "{tactic}"]]\n'
        result = self.__run(inputs)
        return result

    def clear_search(self, search_id):
        inputs = f'["clear_search",["{search_id}"]]\n'
        return self.__run(inputs)

    def __output_parse(self, output):
        null = None
        if len(output) <= 0:
            app.logger.info("Reset formal system server.")('No output')
            raise LeanFatalErrorServer
        output = eval(output)
        if output['search_id'] is not None:
            output['search_id'] = int(output['search_id'])
        if output['tactic_state_id'] is not None:
            output['tactic_state_id'] = int(output['tactic_state_id'])
        if self.normalize_tab:
            if output['tactic_state'] is not None:
                # assert not '\t' in output['tactic_state']
                output['tactic_state'] = output['tactic_state'].replace(
                    '\t', ' ')
        return output

    def __run(self, inputs: str):
        try:
            self.proc.stdin.write(inputs.encode())
            self.proc.stdin.flush()
        except BrokenPipeError:
            # return {'error':'broken_pipe',
            #         'search_id':None,
            #         'tactic_state':None,
            #         'tactic_state_id':None}
            app.logger.info("Reset formal system server.")("Broken pipe")
            raise LeanFatalErrorServer
        return self.__output_parse(self.proc.stdout.readline().decode())

    def kill(self):
        # self.proc.terminate()
        os.kill(self.proc.pid, signal.SIGKILL)


@app.route('/run_cmd')
def run_cmd():
    cmd = request.args.get('cmd')
    if cmd == 'TEST_CONNECTION':
        return 'SUCCESS'
    if cmd == "reset_server":
        app.logger.info("Reset formal system server.")
        global formal_system_server
        global args
        operator.methodcaller('kill')(formal_system_server)
        del formal_system_server
        if args.formal_system == 'lean':
            formal_system_server = LeanServer('lean', lean_gym_dir=args.lean_gym_dir)
        return "Reset done."

    cmd_args = request.args.get('args')
    # app.logger.debug("str_cmd_args: " + cmd_args)
    cmd_args = json.loads(cmd_args)
    # app.logger.info("CMD:")
    # app.logger.info(cmd)
    # app.logger.info("Args: ")
    # app.logger.info(cmd_args)
    try:
        res = str(operator.methodcaller(cmd, *cmd_args)(formal_system_server))
    except (LeanFatalErrorServer, FunctionTimedOut):
        res = "FormalSystemFatalError"
    #     app.logger.info("FormalSystemFatalError occurred.")
    # app.logger.info("Response from this request:")
    # app.logger.info(res)
    sys.stdout.flush()
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_port", type=int, default=8000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--formal_system", type=str, default="lean")
    parser.add_argument("--lean_gym_dir", type=str, default="lean_gym")
    parser.add_argument("--pool_size", type=int, default=1)

    args = parser.parse_args()
    if args.formal_system == 'lean':
        formal_system_server = LeanServer('lean', lean_gym_dir=args.lean_gym_dir)
    else:
        raise NotImplementedError()

    app.run(port=args.port, debug=True, host="0.0.0.0", use_reloader=False, threaded=False, processes=1)