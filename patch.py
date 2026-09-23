import sys
import yaml

with open('configs/environment.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['fire'] = {
    'base_spread_rate': 0.04,
    'base_burning_duration': 40
}

with open('configs/environment.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
