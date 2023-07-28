import subprocess
import pytest


@pytest.mark.timeout(10)
def test_mpi_program():
    # Start the MPI program.
    cmd = ['mpirun', '-np', '4', 'python3', 'run_ddl.py']
    process = subprocess.Popen(cmd)

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


