"""Export populated graph memory from a large original checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from econet.checkpoint import load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/absscene_eval.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    model, config, record = load(args.input, config, 'cpu')
    save(args.output, model, config, record.get('step', 0),
         provenance=record.get('provenance', {}), compact=True)


if __name__ == '__main__':
    main()
