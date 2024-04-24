import sys
import twinlab as tl

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <emulator_id> <process_id>")
    exit()
emulator_id = sys.argv[1]
process_id = sys.argv[2]
# method = sys.argv[3]

emulator = tl.Emulator(id=emulator_id)
emulator.get_process(process_id=process_id, verbose=True)
