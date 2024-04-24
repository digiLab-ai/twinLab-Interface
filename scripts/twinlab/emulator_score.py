import sys
import twinlab as tl

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <emulator_id> <metric> <combined_score>")
    exit()

emulator_id = sys.argv[1]
metric = sys.argv[2]
combined_score = eval(sys.argv[3])

emulator = tl.Emulator(id=emulator_id)
params = tl.ScoreParams(metric=metric, combined_score=combined_score)
emulator.score(params=params, verbose=True)
