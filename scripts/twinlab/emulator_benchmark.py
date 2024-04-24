import sys
import twinlab as tl

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <emulator_id> <benchmark_type>")
    exit()

emulator_id = sys.argv[1]
benchmark_type = sys.argv[2]

emulator = tl.Emulator(id=emulator_id)
params = tl.BenchmarkParams(type=benchmark_type)
emulator.benchmark(params=params, verbose=True)
