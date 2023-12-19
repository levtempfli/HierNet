# Status: Complete
# Description: List number of groups in every grouping file in a logdir directory

import argparse
import os
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, help="Logdir with the grouping files", required=True)

def main(args):
    group_files = [f for f in os.listdir(args.logdir) if os.path.isfile(os.path.join(args.logdir, f)) and re.match("grouping-.+-.+\.json",f)]
    for f in group_files:
        with open(os.path.join(args.logdir, f)) as fo:
            g=json.load(fo)
            print(f,":",len(g))



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
