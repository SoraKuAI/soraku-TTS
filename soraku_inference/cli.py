"""Command-line inference for an independently downloaded Soraku model directory."""
import argparse
import json

from .runtime import synthesize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','codec-dir','reference-audio','reference-text','text','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--reference-language', choices=('zh','en','ja','auto'), required=True)
    parser.add_argument('--language', choices=('zh','en','ja','auto'), required=True)
    parser.add_argument('--speaker-encoder')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-scale', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=.95)
    parser.add_argument('--max-new-frames', type=int, default=400)
    parser.add_argument('--backend', choices=('native','fast','graph'), default='native')
    print(json.dumps(synthesize(**vars(parser.parse_args())),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
