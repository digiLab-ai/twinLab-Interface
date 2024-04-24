import sys

import twinlab as tl


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <server_url>")
    exit()

server_url = sys.argv[1]

tl.set_server_url(server_url, verbose=True)
