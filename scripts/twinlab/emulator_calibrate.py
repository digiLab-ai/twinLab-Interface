import sys
import twinlab as tl

if len(sys.argv) != 4:
    print(
        f"Usage: python {sys.argv[0]} <emulator_id> <path/to/obs_data.csv> <path/to/obs_std_data.csv>"
    )
    exit()

emulator_id = sys.argv[1]
obs_csv = sys.argv[2]
std_csv = sys.argv[3]

emulator = tl.Emulator(id=emulator_id)
df_obs = tl.load_dataset(obs_csv)
df_std = tl.load_dataset(std_csv)
params = tl.CalibrateParams(
    iterations=100,
    seed=0,
)
emulator.calibrate(df_obs, df_std, params, verbose=True)
