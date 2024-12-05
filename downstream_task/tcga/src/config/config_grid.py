import os
import time
import json

from sklearn.model_selection import ParameterGrid

class ConfigGridRunner:
    """Converts a configuration file into a parameter grid and executes the corresponding jobs."""

    def __init__(self, config, repo_path):
        self.configs = list(ParameterGrid(config))
        self.repo_path = repo_path

    def run(self):

        for config in self.configs:
            # create results directory
            timestamp = self._get_timestamp()
            save_path = os.path.join(config['save_path'], timestamp)
            config['save_path'] = save_path
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            # add lines of bash script
            script_lines = '\n'.join([
                '#!/bin/bash',
                '#SBATCH --output=/path/to/output/LOGS/%j.out',
                '#SBATCH --error=/path/to/output/LOGS/%j.out',
                '#SBATCH --gres=gpu:1',
                '#SBATCH --mem=60G',
                '',
                'source /path/to/conda/conda/etc/profile.d/conda.sh',
                'conda activate pytcu11',
                '',
                f'python -m bin.train --config_path {os.path.join(save_path, "config.json")}'
            ])

            # save config and bash script
            with open(os.path.join(save_path, 'config.json'), 'w') as conf_file:
                json.dump(config, conf_file, indent=4)
            with open(os.path.join(save_path, 'run.sh'), 'w') as lsf_file:
                lsf_file.writelines(script_lines)

            # run bash script
            os.system(f'sbatch {os.path.join(save_path, "run.sh")}')
            time.sleep(1)

    @staticmethod
    def _get_timestamp():
        now = time.gmtime()
        return '{:02}{:02}-{:02}{:02}{:02}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)