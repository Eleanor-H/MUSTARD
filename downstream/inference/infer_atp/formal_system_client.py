import os
import time
from abc import ABC, abstractmethod

import requests
import json
import traceback
from requests.exceptions import ConnectionError
LEAN_PATH = os.path.expanduser('~/.elan/bin/lean')


class FormalSystemFatalError(Exception):
    """Raise when Lean fatal error ocurred"""
    pass


class BaseClient(ABC):
    def __init__(self, port):
        self.port = port
        succeed = False

        # wait for 10 mins
        print("Waiting formal system server...")
        for _ in range(600):
            try:
                url = "http://127.0.0.1:%d/run_cmd" % self.port
                response = requests.get(url, params={"cmd": "TEST_CONNECTION", "args": json.dumps([])})
                if response.text == "SUCCESS":
                    succeed = True
                    break

                # you should never reach here
                raise ConnectionError
            except ConnectionError:
                time.sleep(1)
        if succeed:
            print("Formal system server ready to go")
        else:
            raise ConnectionError



    @abstractmethod
    def init_search(self):
        pass

    def run_tac(self, search_id, tactic_id, tactic):
        cmd = "run_tac"
        args = [search_id, tactic_id, tactic]
        return self.exe_cmd(cmd, args)

    def clear_search(self, search_id):
        cmd = "clear_search"
        args = [search_id]
        return self.exe_cmd(cmd, args)
    
    def reset_server(self):
        cmd = "reset_server"
        args = []
        return self.exe_cmd(cmd, args)
    
    def exe_cmd(self, cmd, args):
        try:
            assert json.loads(json.dumps(args)) == args, (args, json.loads(json.dumps(args)))
        except Exception:
            print(args)
            print(json.dumps(args))
            exit(-1)
        url = "http://127.0.0.1:%d/run_cmd" % self.port
        response = requests.get(url, params={"cmd": cmd, "args": json.dumps(args)})
        if response.text == "FormalSystemFatalError":
            raise FormalSystemFatalError
        if response.text == "Reset done.":
            return
        res = None
        try:
            res = eval(response.text)
        except:
            traceback.print_exc()
            print(response.text)
            exit(-1)
        return res


class LeanClient(BaseClient):

    def init_search(self, dec_name, namespaces=""):
        cmd = "init_search"
        args = [dec_name, namespaces]
        return self.exe_cmd(cmd, args)
