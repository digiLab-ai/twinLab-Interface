import sys

import twinlab as tl


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <api_key>")
    exit()

api_key = sys.argv[1]

tl.set_api_key(api_key, verbose=True)
