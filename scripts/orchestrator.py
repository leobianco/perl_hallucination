#!/usr/bin/env python3
"""This is a helper script for hyperparameter searches.

It allows you to launch multiple scripts, and notifies you of their success or failure once they are finished running.

You can also easily do grid searches by creating lists of hyperparameters and launching an experiment for each combination in it.
"""

#!/usr/bin/env python3
import os
import sys

# Add the project root to Python path so you can import from src.utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import orchestrator_run_experiment

if __name__ == "__main__":

    email = "leonardo@rhl.com.br"
    root_dir = project_root

    orchestrator_run_experiment(
        "scripts/hyperparameter_search/reward_model.sh",
        script_args=["npov"],
        notify_email=email,
        working_dir=project_root,
    )
