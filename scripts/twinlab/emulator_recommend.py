import sys
import twinlab as tl

if len(sys.argv) != 4:
    print(
        f"Usage: python {sys.argv[0]} <emulator_id> <number_of_points> <acquisition_function>"
    )
    exit()

emulator_id = sys.argv[1]
num_points = int(sys.argv[2])
acq_func = sys.argv[3]

emulator = tl.Emulator(id=emulator_id)
emulator.recommend(num_points, acq_func, verbose=True)
