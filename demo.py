import argparse
import sys

if __name__=='__main__':

    # set_seed(seed=3407)
    # set_seed(seed=42)

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-m', '--model', required=True, default="pesi", action='store_true', help="set model name")

    # args = parser.parse_args()

    # if not args.model:
    #     print("no model name")
    #     exit()
    # else:
    #     model_name = args.model

    # print(model_name)

    model_name = sys.argv[1]

    print(model_name)