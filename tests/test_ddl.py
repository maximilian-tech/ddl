import subprocess
import pytest
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.timeout(100)
def test_mpi_program():
    # Create an absolute path to run_ddl.py based on the current script's directory.
    script_path = os.path.join(SCRIPT_DIR, "run_ddl.py")

    # Start the MPI program.
    cmd = ["mpirun", "-np", "4", "python3", script_path]
    process = subprocess.Popen(cmd, cwd=SCRIPT_DIR)  # set cwd if necessary

    try:
        # Run your tests here.
        process.wait()
        rt = process.returncode
        print(rt)
        assert rt == 0

    except Exception as e:
        # If any exception occurs (including a timeout), terminate the process.
        process.terminate()
        process.wait()
        raise e
