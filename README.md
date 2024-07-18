![twinLab Banner](https://hackmd.io/_uploads/BkQro2CgR.png)

# twinLab - Probabilistic Machine Learning for Engineers

twinLab is a tool for augmenting engineering workflows with Probabilistic Machine Learning. It enables users to quickly and easily build real-time emulators of their simulations, experimental set-ups, or sensor networks. Then they can make predictions, make recommendations, perform optimisations, and calibrate physics parameters from data.

twinLab comes with built-in uncertainty quantification (UQ), which means that even with sparse or noisy data, users can maximise their understanding of the design space and surrogate model with confidence.

For help, or to arrange a trial, please email: [twinlab@digilab.co.uk](mailto:twinlab@digilab.co.uk) or fill in the contact form [here](https://www.digilab.co.uk/contact).

## Getting Started

**Step 1**: Install the Python Interface

```shell
pip install twinlab
```

**Step 2**: Configure your user details

If you don't yet have one, you'll need to request a trial. Please email [twinlab@digilab.co.uk](mailto:twinlab@digilab.co.uk) or fill in the contact form [here](https://www.digilab.co.uk/contact).

Method 1: Use `tl.set_user` and `tl.set_api_key` within a script. Be careful not to publicly expose your API key if sharing files.

```python
import twinlab as tl

tl.set_user("<your_username>")
tl.set_api_key("<your_api_key>")
```

Method 2: Create a `.env` file containing `TWINLAB_USER` and `TWINLAB_API_KEY` in your working directory. You can then `import twinlab as tl` in your Python script / notebook as normal. The API key will be read from `.env` automatically.

```shell
echo "TWINLAB_USER=<your_username>" >> .env
echo "TWINLAB_API_KEY=<your_api_key>" >> .env
```

**Step 3**: Run an Example

Here’s an example script to get you started:

```python
import pandas as pd
import twinlab as tl

# Load an example dataset and upload to twinLab
dataset = tl.Dataset("quickstart")
df = tl.load_example_dataset("quickstart")
dataset.upload(df)

# Train a machine-learning emulator for the data
emulator = tl.Emulator("test-emulator")
emulator.train(dataset, ["x"], ["y"])

# Evaluate the emulator on some unseen data
sample_points = pd.DataFrame({"x": [0.25, 0.5, 0.75]})
predict_mean, predict_std = emulator.predict(sample_points)

# Explore the results
print(predict_mean)
print(predict_std)
```

## Documentation

Find more examples, tutorials, and the full reference guide for our Python Interface in our [documentation](https://twinlab.ai).

## Speak to an Expert

Our Solution Engineers are here to provide technical support and help you maximise the value of twinLab. Please email [twinlab@digilab.co.uk](mailto:twinlab@digilab.co.uk) or fill in the contact form [here](https://www.digilab.co.uk/contact).
