import os
import platform
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

def check_connection(url):
    try:
        urllib.request.urlopen(url)
        return True
    except:
        return False
    
def extract_file(file):
    if file.suffix == '.gz':
        with tarfile.open(file, 'r:gz') as tar:
            tar.extractall(file.parent)
            extracted_file = Path(tar.getnames()[0]).parts[0]
    else:
        with zipfile.ZipFile(file, 'r') as zip:
            zip.extractall(file.parent)
            extracted_file = Path(zip.namelist()[0]).parts[0]
    return file.parent / extracted_file

def get_elan(version='3.0.0', remove_installation=True):
    parent_path = Path(__file__).resolve().parent.parent
    elan_path = parent_path / 'elan'
    if os.path.exists(elan_path):
        return
    if not check_connection(f'https://github.com/leanprover/elan/releases/download/v3.0.0/elan-x86_64-pc-windows-msvc.zip'):
        raise Exception(f'Fail to connect to the URL: https://github.com/leanprover/elan/releases/download/v3.0.0/elan-x86_64-pc-windows-msvc.zip')
    if platform.system() == 'Windows':
        url = f'https://github.com/leanprover/elan/releases/download/v{version}/elan-x86_64-pc-windows-msvc.zip'
    elif platform.system() == 'Linux':
        if platform.machine() == 'x86_64':
            url = f'https://github.com/leanprover/elan/releases/download/v{version}/elan-x86_64-unknown-linux-gnu.tar.gz'
        elif platform.machine() == 'AArch64':
            url = f'https://github.com/leanprover/elan/releases/download/v{version}/elan-aarch64-unknown-linux-gnu.tar.gz'
        else:
            raise ValueError('Unknown platform')
    elif platform.system() == 'Darwin':
        if platform.machine() == 'x86_64':
            url = f'https://github.com/leanprover/elan/releases/download/v{version}/elan-x86_64-apple-darwin.tar.gz'
        elif platform.machine() == 'AArch64':
            url = f'elan-aarch64-apple-darwin.tar.gz'
        else:
            raise ValueError('Unknown platform')
    else:
        raise ValueError('Unknown platform')
    file = Path(__file__).resolve().parent.parent / os.path.basename(url)
    with urllib.request.urlopen(url) as response, open(file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    extracted_file = extract_file(file)
    assert os.system(f'{extracted_file} -y --no-modify-path --default-toolchain none') == 0
    shutil.move(Path.home() / '.elan', elan_path)
    if remove_installation:
        os.remove(file)
        os.remove(extracted_file)

def get_lean(version='3.51.1', remove_installation=True):
    parent_path = Path(__file__).resolve().parent.parent
    if version.startswith('3'):
        toolchain = f'leanprover-community--lean---{version}'
    else:
        toolchain = f'leanprover--lean4---{version}'
    toolchain_path = parent_path / 'elan' / 'toolchains' / toolchain
    bin_path = parent_path / 'elan' / 'bin' / 'elan'
    if os.path.exists(Path(__file__).resolve().parent.parent / 'elan' / 'toolchains' / f'leanprover-community--lean---{version}') or \
        os.path.exists(Path(__file__).resolve().parent.parent / 'elan' / 'toolchains' / f'leanprover--lean4---{version}'):
        assert os.system(f'{bin_path} toolchain link {toolchain} {toolchain_path}') == 0
        assert os.system(f'{bin_path} default {toolchain}') == 0
        return
    if not check_connection(f'https://github.com/leanprover-community/lean/releases/download/v{version}/lean-{version}-windows.zip'):
        raise Exception(f'Fail to connect to the URL: https://github.com/leanprover-community/lean/releases/download/v{version}/lean-{version}-windows.zip')
    if version.startswith('3'):
        if platform.system() == 'Windows':
            url = f'https://github.com/leanprover-community/lean/releases/download/v{version}/lean-{version}-windows.zip'
        elif platform.system() == 'Linux':
            if platform.machine() == 'x86_64':
                url = f'https://github.com/leanprover-community/lean/releases/download/v{version}/lean-{version}-linux.tar.gz'
            elif platform.machine() == 'AArch64':
                url = f'https://github.com/leanprover-community/lean/releases/download/v{version}/lean-{version}-linux_aarch64.tar.gz'
            else:
                raise ValueError('Unknown platform')
        elif platform.system() == 'Darwin':
            url = f'https://github.com/leanprover-community/lean/releases/download/v{version}/lean-{version}-darwin.zip'
        else:
            raise ValueError('Unknown platform')
    else:
        if platform.system() == 'Windows':
            url = f'https://github.com/leanprover/lean4/releases/download/v{version}/lean-{version}-windows.zip'
        elif platform.system() == 'Linux':
            if platform.machine() == 'x86':
                url = f'https://github.com/leanprover/lean4/releases/download/v{version}/lean-{version}-linux.tar.gz'
            elif platform.machine() == 'AArch64':
                url = f'https://github.com/leanprover/lean4/releases/download/v{version}/lean-{version}-linux_aarch64.tar.gz'
        elif platform.system() == 'Darwin':
            url = f'https://github.com/leanprover/lean4/releases/download/v{version}/lean-{version}-darwin.zip'
        else:
            raise ValueError('Unknown platform')
    file = Path(__file__).resolve().parent.parent / os.path.basename(url)
    with urllib.request.urlopen(url) as response, open(file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    extracted_file = extract_file(file)
    shutil.move(extracted_file, toolchain_path)
    if remove_installation:
        os.remove(file)
    assert os.system(f'{bin_path} toolchain link {toolchain} {toolchain_path}') == 0
    assert os.system(f'{bin_path} default {toolchain}') == 0

def get_mathlib_datasets(remove_installation=True):
    if os.path.exists(Path(__file__).resolve().parent.parent / '_target') and \
        os.path.exists(Path(__file__).resolve().parent.parent / 'proving_env_lib.bak') and \
            os.path.exists(Path(__file__).resolve().parent.parent / 'proving_env_lib.dat') and \
                os.path.exists(Path(__file__).resolve().parent.parent / 'proving_env_lib.dir'):
        return
    url = 'https://github.com/xinhjBrant/leaven/releases/download/release/mathlib_dataset.tar.gz'
    file_name = Path(__file__).resolve().parent.parent / os.path.basename(url)
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 1024
        wrote = 0
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(url)) as progress:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                wrote += len(buffer)
                out_file.write(buffer)
                progress.update(len(buffer))
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(Path(__file__).resolve().parent.parent)
    if remove_installation:
        os.remove(file_name)

def download_mathlib(commit="58a272265b5e05f258161260dd2c5d247213cbd3"):
    os.system("git clone git@github.com:leanprover-community/mathlib.git _target/deps/mathlib")
    os.chdir("_target/deps/mathlib")
    os.system(f"git checkout {commit}")
    os.chdir("../../..")

def checking_complete():
    from .lean_server import LeanEnv
    lean_server = LeanEnv(do_logging=True)
    result = lean_server.verify_lean_file(content='''import data.nat.basic\n\n#check nat.succ_eq_one_add''')
    assert not result['error'] and result['info'] == 'line 3, column 0: nat.succ_eq_one_add : ∀ (n : ℕ), n.succ = 1 + n'
    lean_server.close()