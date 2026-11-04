"""Build with the portable CPU flags used by the target environment."""
import runpy
import sys
from pathlib import Path

import setuptools

original_setup = setuptools.setup


def portable_flags(values):
    return [flag for flag in (values or [])
            if not flag.startswith(('-march=', '-mtune='))] + [
                '-march=x86-64', '-mtune=generic']


def setup(*args, **kwargs):
    # Build dependencies come from Base and requirements.txt, not legacy downloads.
    kwargs['setup_requires'] = []
    for extension in kwargs.get('ext_modules') or []:
        flags = extension.extra_compile_args
        extension.extra_compile_args = (
            {key: portable_flags(value) for key, value in flags.items()}
            if isinstance(flags, dict) else portable_flags(flags))
    return original_setup(*args, **kwargs)


setuptools.setup = setup
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
