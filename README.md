# goodput-lab

```
pip install -e .
python3 -m goodput_lab.window_policy --download --out runs/window_policy.json
```

`--out` is required. `--download` fetches the Azure Conversation CSV to `src/goodput_lab/data/AzureLLMInferenceTrace_conv_1week.csv` (gitignored). Optional `--window-sec 60`.
