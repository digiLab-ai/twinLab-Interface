import sys
import twinlab as tl

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <emulator_id>")
    exit()
emulator_id = sys.argv[1]

emulator = tl.Emulator(id=emulator_id)
emulator.summarise(verbose=True)
