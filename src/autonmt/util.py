import subprocess

class SubprocessError(RuntimeError):
    def __init__(self, result, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = result

def run_and_check_result(*args, **kwargs):
    result = subprocess.run(*args, **kwargs)
    if result.returncode != 0:
        raise SubprocessError(result, f'Subprocess returned {result.returncode}')
    return result