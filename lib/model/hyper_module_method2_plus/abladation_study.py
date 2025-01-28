import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '../../..'))
from lib.utils.load_parameter import parse_args

args = parse_args()
j2j_adjacency = args.j2j_adjacency
j2e_main = args.j2e_main
j2e_bias = args.j2e_bias
