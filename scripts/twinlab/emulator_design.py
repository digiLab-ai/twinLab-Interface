import sys
import twinlab as tl

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <num_points>")
    exit()

priors = [
    tl.Prior("x1", tl.distributions.Uniform(0, 12)),
    tl.Prior("x2", tl.distributions.Uniform(0, 0.5)),
    tl.Prior("x3", tl.distributions.Uniform(0, 10)),
]
num_points = int(sys.argv[1])

emulator = tl.Emulator(id="test")
emulator.design(priors=priors, num_points=num_points, verbose=True)
