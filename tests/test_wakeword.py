import subprocess
import sys


def test_wakeword_module_does_not_load_scipy_or_sklearn() -> None:
    # Fresh interpreter: other tests may already have imported these.
    code = (
        "import sys, jarvis.wakeword; "
        "print(sorted(m for m in ('scipy', 'sklearn') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
