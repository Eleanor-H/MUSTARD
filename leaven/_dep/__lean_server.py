import time
from typing import SupportsFloat, Any, Tuple, Dict, List
import re
import sys
import gc
from typing import List
import psutil
import subprocess
import logging
import threading
from datetime import datetime
import pytz
from dataclasses import dataclass
from typing import Optional, List, NewType, ClassVar, Union, Type
from enum import Enum
import json
from pathlib import Path
from leaven.src.file_tools import *

parent_path = Path(__file__).parent.parent.resolve()



def dict_to_dataclass(cls, dic: dict):
    dic = {k: dic[k] for k in cls.__dataclass_fields__ if k in dic}
    return cls(**dic)

class Request:
    command: ClassVar[str]
    expect_response: ClassVar[bool]

    def __post_init__(self):
        self.seq_num = 0

    def to_json(self) -> str:
        dic = self.__dict__.copy()
        dic['command'] = self.command
        return json.dumps(dic)

class Response:
    response: ClassVar[str]

    @staticmethod
    def parse_response(data: str) -> Union['AllMessagesResponse', 'CurrentTasksResponse', 'OkResponse', 'ErrorResponse']:
        dic = json.loads(data)
        response = dic.pop('response')

        for cls in [AllMessagesResponse, CurrentTasksResponse, OkResponse, ErrorResponse]:
            if response == cls.response:  # type: ignore
                return cls.from_dict(dic)  # type: ignore
        raise ValueError("Couldn't parse response string.")



Severity = Enum('Severity', 'information warning error')


@dataclass
class Message:
    file_name: str
    severity: Severity
    caption: str
    text: str
    pos_line: int
    pos_col: int
    end_pos_line: Optional[int] = None
    end_pos_col: Optional[int] = None

    @classmethod
    def from_dict(cls, dic):
        dic['severity'] = getattr(Severity, dic['severity'])
        return dict_to_dataclass(cls, dic)


@dataclass
class AllMessagesResponse(Response):
    response = 'all_messages'
    msgs: List[Message]

    @classmethod
    def from_dict(cls, dic):
        return cls([Message.from_dict(msg) for msg in dic['msgs']])


@dataclass
class Task:
    file_name: str
    pos_line: int
    pos_col: int
    end_pos_line: int
    end_pos_col: int
    desc: str


@dataclass
class CurrentTasksResponse(Response):
    response = 'current_tasks'
    is_running: bool
    tasks: List[Task]
    cur_task: Optional[Task] = None

    @classmethod
    def from_dict(cls, dic):
        dic['tasks'] = [dict_to_dataclass(Task, task) for task in dic.pop('tasks')]
        return dict_to_dataclass(cls, dic)


@dataclass
class ErrorResponse(Response):
    response = 'error'
    message: str
    seq_num: Optional[int] = None

    @classmethod
    def from_dict(cls, dic):
        return dict_to_dataclass(cls, dic)

@dataclass
class CommandResponse(Response):
    """
    Parent class for all 'ok' responses directly tied to a specific request.
    """
    command: ClassVar[str]
    response = 'ok'
    seq_num: int

    @classmethod
    def from_dict(cls, dic):
        return dict_to_dataclass(cls, dic)


@dataclass
class OkResponse(Response):
    """
    Intermediate representation of a CommandResponse which can be constructed from the
    JSON alone.  It can later be converted to a CommandResponse.
    """
    response = 'ok'
    seq_num: int
    data: dict

    @classmethod
    def from_dict(cls, dic) -> 'OkResponse':
        return OkResponse(seq_num=dic['seq_num'], data=dic)

    def to_command_response(self, command: str) -> CommandResponse:
        response_types: List[Type[CommandResponse]] = [
            CompleteResponse, InfoResponse, HoleCommandsResponse, SyncResponse,
            SearchResponse, AllHoleCommandsResponse, HoleResponse, RoiResponse
        ]
        for cls in response_types:
            if cls.command == command:
                self.data['seq_num'] = self.seq_num
                return cls.from_dict(self.data)
        raise ValueError("Couldn't parse response string.")


@dataclass
class SyncRequest(Request):
    command = 'sync'
    expect_response = True
    file_name: str
    content: Optional[str] = None

    def to_json(self):
        dic = self.__dict__.copy()
        dic['command'] = 'sync'
        if dic['content'] is None:
            dic.pop('content')
        return json.dumps(dic)


@dataclass
class SyncResponse(CommandResponse):
    command = 'sync'
    message: Optional[str] = None


@dataclass
class CompleteRequest(Request):
    command = 'complete'
    expect_response = True
    file_name: str
    line: int
    column: int
    skip_completions: bool = False


@dataclass
class Source:
    line: Optional[int] = None
    column: Optional[int] = None
    file: Optional[str] = None


@dataclass
class CompletionCandidate:
    text: str
    type_: Optional[str] = None
    tactic_params: Optional[str] = None
    doc: Optional[str] = None
    source: Optional[Source] = None

    @classmethod
    def from_dict(cls, dic):
        dic['type_'] = dic.pop('type')
        if 'source' in dic:
            dic['source'] = dict_to_dataclass(Source, dic.pop('source'))
        return dict_to_dataclass(cls, dic)


@dataclass
class CompleteResponse(CommandResponse):
    command = 'complete'
    prefix: Optional[str] = None
    completions: Optional[List[CompletionCandidate]] = None

    @classmethod
    def from_dict(cls, dic):
        if 'completions' in dic:
            dic['completions'] = [CompletionCandidate.from_dict(cdt)
                                  for cdt in dic.pop('completions')]
        return dict_to_dataclass(cls, dic)


@dataclass
class InfoRequest(Request):
    command = 'info'
    expect_response = True
    file_name: str
    line: int
    column: int


GoalState = NewType('GoalState', str)


@dataclass
class InfoRecord:
    full_id: Optional[str] = None
    text: Optional[str] = None
    type_: Optional[str] = None
    doc: Optional[str] = None
    source: Optional[Source] = None
    state: Optional[GoalState] = None
    tactic_param_idx: Optional[int] = None
    tactic_params: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, dic):
        if 'full-id' in dic:
            dic['full_id'] = dic.pop('full-id')
        if 'type' in dic:
            dic['type_'] = dic.pop('type')
        if 'source' in dic:
            dic['source'] = dict_to_dataclass(Source, dic.pop('source'))
        return dict_to_dataclass(cls, dic)


@dataclass
class InfoResponse(CommandResponse):
    command = 'info'
    record: Optional[InfoRecord] = None

    @classmethod
    def from_dict(cls, dic):
        if 'record' in dic:
            dic['record'] = InfoRecord.from_dict(dic.pop('record'))
        return dict_to_dataclass(cls, dic)


@dataclass
class SearchRequest(Request):
    command = 'search'
    expect_response = True
    query: str


@dataclass
class SearchItem:
    text: str
    type_: str
    source: Optional[Source] = None
    doc: Optional[str] = None

    @classmethod
    def from_dict(cls, dic):
        dic['type_'] = dic.pop('type')
        if 'source' in dic:
            dic['source'] = dict_to_dataclass(Source, dic.pop('source'))
        return dict_to_dataclass(cls, dic)


@dataclass
class SearchResponse(CommandResponse):
    command = 'search'
    results: List[SearchItem]

    @classmethod
    def from_dict(cls, dic):
        dic['results'] = [SearchItem.from_dict(si)
                          for si in dic.pop('results')]
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleCommandsRequest(Request):
    command = 'hole_commands'
    expect_response = True
    file_name: str
    line: int
    column: int


@dataclass
class HoleCommandAction:
    name: str
    description: str


@dataclass
class Position:
    line: int
    column: int


@dataclass
class HoleCommands:
    file: str
    start: Position
    end: Position
    results: List[HoleCommandAction]

    @classmethod
    def from_dict(cls, dic):
        dic['results'] = [dict_to_dataclass(HoleCommandAction, hc)
                          for hc in dic.pop('results')]
        dic['start'] = dict_to_dataclass(Position, dic.pop('start'))
        dic['end'] = dict_to_dataclass(Position, dic.pop('end'))
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleCommandsResponse(CommandResponse):
    command = 'hole_commands'
    message: Optional[str] = None
    file: Optional[str] = None
    start: Optional[Position] = None
    end: Optional[Position] = None
    results: Optional[List[HoleCommandAction]] = None

    @classmethod
    def from_dict(cls, dic):
        if 'results' in dic:
            dic['results'] = [dict_to_dataclass(HoleCommandAction, hc)
                              for hc in dic.pop('results')]
            dic['start'] = dict_to_dataclass(Position, dic.pop('start'))
            dic['end'] = dict_to_dataclass(Position, dic.pop('end'))

        return dict_to_dataclass(cls, dic)


@dataclass
class AllHoleCommandsRequest(Request):
    command = 'all_hole_commands'
    expect_response = True
    file_name: str


@dataclass
class AllHoleCommandsResponse(CommandResponse):
    command = 'all_hole_commands'
    holes: List[HoleCommands]

    @classmethod
    def from_dict(cls, dic):
        dic['holes'] = [HoleCommands.from_dict(hole)
                          for hole in dic.pop('holes')]
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleRequest(Request):
    command = 'hole'
    expect_response = True
    file_name: str
    line: int
    column: int
    action: str


@dataclass
class HoleReplacementAlternative:
    code: str
    description: str


@dataclass
class HoleReplacements:
    file: str
    start: Position
    end: Position
    alternatives: List[HoleReplacementAlternative]

    @classmethod
    def from_dict(cls, dic):
        dic['alternatives'] = [dict_to_dataclass(HoleReplacementAlternative, alt)
                               for alt in dic.pop('alternatives')]
        dic['start'] = dict_to_dataclass(Position, dic.pop('start'))
        dic['end'] = dict_to_dataclass(Position, dic.pop('end'))
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleResponse(CommandResponse):
    command = 'hole'
    replacements: Optional[HoleReplacements] = None
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, dic):
        if 'replacements' in dic:
            dic['replacements'] = HoleReplacements.from_dict(
                    dic.pop('replacements'))
        return dict_to_dataclass(cls, dic)


CheckingMode = Enum('CheckingMode',
    'nothing visible-lines visible-lines-and-above visible-files open-files')


@dataclass
class RoiRange:
    begin_line: int
    end_line: int


@dataclass
class FileRoi:
    file_name: str
    ranges: List[RoiRange]

    def to_dict(self):
        return {'file_name': self.file_name,
                'ranges': [rr.__dict__ for rr in self.ranges] }


@dataclass
class RoiRequest(Request):
    command = 'roi'
    expect_response = True
    mode: CheckingMode
    files: List[FileRoi]

    def to_json(self) -> str:
        dic = self.__dict__.copy()
        dic['command'] = 'roi'
        dic['mode'] = dic['mode'].name
        dic['files'] = [fileroi.to_dict() for fileroi in dic['files']]

        return json.dumps(dic)


@dataclass
class RoiResponse(CommandResponse):
    command = 'roi'


@dataclass
class SleepRequest(Request):
    command = 'sleep'
    expect_response = False


@dataclass
class LongSleepRequest(Request):
    command = 'long_sleep'
    expect_response = False

class LeanServerMonitor:
    def __init__(
        self,
        lean_server_path: str = f_join('elan', 'bin', 'lean'),
        max_memory_limit: int = 20480,
        log_path: str = "logs",
        finished_callback: callable = None,
        do_logging = False,
        cwd='.'
    ):
        self.cwd = cwd
        self.process = None
        self.finished_callback = finished_callback
        self.thread = None
        self.messages = []
        self.max_memory_limit = max_memory_limit
        self.commands = [lean_server_path, "--server", "-M" , str(self.max_memory_limit)]
        self.seq_num = 0
        self.request = None
        self.response = dict()
        self.process = psutil.Popen(
            self.commands,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=parent_path
        )
        self.run_thread = True
        self.logger = None
        self.thread = threading.Thread(target=self._receive)
        self.thread.start()
        self.log_path = log_path
        self.start_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
        if do_logging:
            self.logger = logging.getLogger('lean_server')
            self.file_handler = logging.FileHandler(f_join(self.log_path, f"{self.start_time}_{self.process.pid}.log"), mode='w')
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.info(f"Starting subprocess with commands: {' '.join(self.commands)}")            

    def _send(self, request : Request):
        self.file_invalidated = False
        self.request = request
        request_json = request.to_json()
        if self.logger:
            self.logger.info('send: ' + request_json)
        self.process.stdin.write(request_json + '\n')
        self.process.stdin.flush()

    def _receive(self):
        for line in iter(self.process.stdout.readline, ""):
            if not self.run_thread:
                return
            if self.logger:
                self.logger.info('receive: ' + line.strip())
            resp = CommandResponse.parse_response(line)
            if isinstance(resp, CurrentTasksResponse):
                self.current_tasks = resp.tasks
                if not resp.is_running and self.file_invalidated:
                    self.task_finished_event.set()
            elif isinstance(resp, AllMessagesResponse):
                self.messages = resp.msgs
            elif isinstance(resp, ErrorResponse):
                self.response[self.request.seq_num] = resp
                self.response_event.set()
            elif isinstance(resp, OkResponse):
                self.response[self.request.seq_num] = resp
                self.response_event.set()
                if self.file_invalidated == False and 'message' in resp.data and resp.data['message'] in ['file invalidated', 'file unchanged']:
                    self.file_invalidated = True
                    self.task_finished_event = threading.Event()
                    if resp.data['message'] == 'file unchanged':
                        self.task_finished_event.set()

    def run_command(self, request : Request):
        request.seq_num = self.seq_num
        self.seq_num += 1
        self.response_event = threading.Event()
        self._send(request)
        self.response_event.wait()
        self.task_finished_event.wait()
        if request.command == 'sync':
            time.sleep(0.3)
        if isinstance(self.response[request.seq_num], OkResponse):
            cmd_response = self.response[request.seq_num].to_command_response(request.command)
        else:
            assert isinstance(self.response[request.seq_num], ErrorResponse)
            raise ChildProcessError(f'Lean server error while executing "{request.command}":\n{self.response[request.seq_num]}')
        
        if self.finished_callback:
            self.finished_callback()

        return cmd_response

    def full_sync(self, filename : str, content=None):
        request = SyncRequest(filename, content)
        cmd_response = self.run_command(request)
        assert isinstance(cmd_response, SyncResponse), 'fatal error'
        return self.messages
    
    def state(self, filename : str, line : int, col : int) -> str:
        request = InfoRequest(filename, line, col)
        request.seq_num = self.seq_num
        cmd_response = self.run_command(request)
        if isinstance(cmd_response, InfoResponse) and cmd_response.record:
            return cmd_response.record or ''
        else:
            return ''

    def stop(self):
        if self.process and self.process.is_running():
            if self.logger:
                self.logger.info("Stopping subprocess.")
            self.process.terminate()
            self.process.wait()
            self.run_thread = False
            self.thread.join()
        if self.logger:
            self.logger.removeHandler(self.file_handler)
            self.file_handler = logging.FileHandler(f_join(self.log_path, f"{self.start_time}_{self.process.pid}.log"), mode='w')
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)

    @property
    def is_running(self):
        if self.process is None:
            return False
        return self.process.is_running()
    
class LeanServerNoInfoError(Exception):
    """
    Raised when the Lean server returns no information.
    """
    pass

class LeanEnv:
    def __init__(
        self,
        request_timeout=600,
        do_logging=False,
        log_path="./logs",
        # lean_file_paths: List[str] = ['_target/deps/mathlib/src', 'src'],
        cwd='.',
    ):
        self.cwd = cwd
        self.request_timeout = request_timeout
        self.log_path = log_path
        # self.lean_file_paths = lean_file_paths
        self.lean_server = None
        self.has_reset = True
        self.reset_options = None
        self.connected = True
        self.do_logging = do_logging

    def get_lean_server_process(self):
        if self.do_logging:
            f_mkdir(self.log_path, "lean_server")
        return LeanServerMonitor(
            log_path=f_join(self.log_path, "lean_server"),
            # lean_file_paths=self.lean_file_paths,
            do_logging=self.do_logging,
            cwd=self.cwd
        )

    def check_process(self):
        if not self.lean_server or not self.lean_server.is_running:
            if self.lean_server:
                del self.lean_server
            self.lean_server = self.get_lean_server_process()
        if self.reset_options:
            return self.lean_server.full_sync(**self.reset_options)

    def reset(
        self,
        options=None,
    ):
        if options is None:
            options = {}

        self.reset_options = options
        
        if self.lean_server:
            self.lean_server.stop()
            del self.lean_server
            self.lean_server = None
            gc.collect()

        returned_data = self.check_process()
        self.has_reset = True
        self.connected = True
        # All the reset in step will be soft
        return returned_data
    
    def step(
        self,
        options: dict,
    ):
        self.reset_options = options
        return self.check_process()
    
    def render(self,
        options: dict):
        return self.lean_server.state(**options)
    
    @staticmethod
    def find_positions(match, s):
        start_pos = match.start()
        end_pos = match.end()
        
        # compute start and end line and column
        start_line = s.count('\n', 0, start_pos) + 1
        start_col = start_pos - s.rfind('\n', 0, start_pos) - 1  # 列数从0开始
        
        # compute end line and column
        end_line = s.count('\n', 0, end_pos) + 1
        end_col = end_pos - s.rfind('\n', 0, end_pos) - 1  # 列数从0开始

        positions = []

        for line in range(start_line, end_line + 1):
            if line == start_line and line == end_line:
                for col in range(start_col, end_col + 1):
                    positions.append((line, col))
            elif line == start_line:
                positions.extend([(line, col) for col in range(start_col, len(s.split('\n')[line - 1]))])
            elif line == end_line:
                positions.extend([(line, col) for col in range(end_col + 1)])
            else:
                positions.extend([(line, col) for col in range(len(s.split('\n')[line - 1]))])
        
        return positions[:-1]
    
    def render_all(self, filename, full_context, again=False, check_span=None):
        full_context = re.sub(r'by\s+sorry\s+', 'by { sorry } ', full_context)
        outputs = None
        search_flag = check_span[0] if check_span is not None else 0
        pattern = re.compile(r'\s*repeat\s*\{\s*sorry\b|by\s+sorry\b|\s*\bsorry\b')
        while search := pattern.search(full_context, search_flag, check_span[1] if check_span is not None else sys.maxsize):
            output = None
            search_flag = search.end()
            positions = self.find_positions(search, full_context)
            for pos in positions:
                if (output := self.render(options={"filename" : filename, "line" : pos[0], "col" : pos[1]})) and output.state:
                    if output.state == 'no goals':
                        if outputs is None:
                            outputs = []
                    elif outputs is None:
                        outputs = [(pos[0], pos[1], output.state)]
                    else:
                        outputs.append((pos[0], pos[1], output.state))
                    break
            if not output or output.state is None: # not output means: is None or is an empty string
                if not again:
                    self.reset(options={"filename": filename, "content": full_context})
                    return self.render_all(filename, full_context, True, check_span=check_span)
                else:
                    raise LeanServerNoInfoError(f"Lean server returns no information at line {pos[0]}, column {pos[1]} in the following content:\n{full_context}")
            assert outputs is not None, full_context
        return outputs, full_context

    def close(self):
        if self.lean_server:
            self.lean_server.stop()

    def verify_lean_file(self, content, filename='', check_span=None):
        """
        Writes the provided content to a working file named f'inference/working_{start_time}.lean'. The function then uses the Lean 3 verifier to validate the content of this working file. It processes the events returned by the verifier and categorizes them into errors, warnings, and informational messages based on their severity. Additionally, the function also retrieves the local proof states for all instances of the keyword `sorry` in the file content. These proof states are attached with their corresponding line and column numbers and returned.
        :param file_content: the content of a .lean file to verify
        """
        events = self.step(options={"filename": filename, 'content': content})
        error =  '\n\n'.join([f"line {e.pos_line}, column {e.pos_col}: \n{e.text}" for e in events if e.severity is Severity.error])
        warning =  '\n\n'.join([f"line {e.pos_line}, column {e.pos_col}: \n{e.text}" for e in events if e.severity is Severity.warning])
        info =  '\n\n'.join([f"line {e.pos_line}, column {e.pos_col}: \n{e.text}" for e in events if e.severity is Severity.information])
        open_states = []
        if check_span is None or check_span[0] < check_span[1]:
            render_result = self.render_all(filename=filename, full_context=content, check_span=check_span) # get the local proof states for all keyword `sorry`
            if isinstance(render_result, LeanServerNoInfoError):
                if error:
                    open_states = []
                else:
                    return render_result
            else:
                open_states, content = render_result
            open_states = [f"line {i[0]}, column {i[1]}: \n{i[2]}" for i in open_states if i[2] is not None] if open_states is not None else [] # proof states attached with line and column
        return {
            'error' : error, 'warning' : warning, 'info' : info, 'open_states' : open_states, 'context' : content
        }