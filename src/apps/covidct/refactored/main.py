import sys, os
import json



def main():
    if len(sys.argv) != 2:
        print("need input filename with json")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        inp = json.load(f)

    o1 = step1()

    o2 = step2()

    o3 = segment()

    o4 =predict()