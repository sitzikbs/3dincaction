import wandb
import json
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()

sweep = api.sweep("cgmlab/point_drop_sweep/pk4gfgjx")
sweep_runs = sweep.runs
# if sweep.state == "finished":
results = []
for run_i in sweep_runs:
    summary = run_i.summary
    degredation_rate = summary['degredation_rate']
    config = json.loads(run_i.json_config)
    centroid_noise = config["centroid_noise"]["value"]
    results.append({"centroid_noise": centroid_noise, "degredation_rate": degredation_rate})

results_df = pd.DataFrame(results)
results_df.plot(x="centroid_noise", y="degredation_rate")
plt.show()