import re
import numpy as np
import matplotlib.pyplot as plt

# Load the log data
dataset = 'GeoGLUE'
log_file_path = f'result/training_{dataset.lower()}.log'
with open(log_file_path, 'r') as file:
    log_data = file.readlines()

# Initialize data structure to store results
results = {}
current_run = 0  # Start from 0
current_epoch = None

# Regular expressions for parsing log content
epoch_pattern = re.compile(r"Epoch=(\d+)")
retrieve_loss_pattern = re.compile(r"retrieve loss=([\d\.]+)")
rank_loss_pattern = re.compile(r"rank loss=([\d\.]+)")
ndcg5_pattern = re.compile(r"Dev NDCG @ 5: ([\d\.]+)")

# Process each line in the log file
for line in log_data:
    # Detect the start of a new run based on "ninja: no work to do."
    if "ninja: no work to do." in line:
        current_run += 1
        current_epoch = None  # Reset the epoch for a new run
        continue  # Skip to the next line

    # Extract epoch
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        if current_run not in results:
            results[current_run] = {}
        if current_epoch not in results[current_run]:
            results[current_run][current_epoch] = {"retrieve_loss": None, "rank_loss": None, "dev_ndcg5": None}

    # Extract retrieve loss if we have a valid epoch
    if current_epoch is not None:
        retrieve_loss_match = retrieve_loss_pattern.search(line)
        if retrieve_loss_match:
            retrieve_loss = float(retrieve_loss_match.group(1))
            results[current_run][current_epoch]["retrieve_loss"] = retrieve_loss

        # Extract rank loss
        rank_loss_match = rank_loss_pattern.search(line)
        if rank_loss_match:
            rank_loss = float(rank_loss_match.group(1))
            results[current_run][current_epoch]["rank_loss"] = rank_loss

        # Extract dev NDCG5
        ndcg5_match = ndcg5_pattern.search(line)
        if ndcg5_match:
            dev_ndcg5 = float(ndcg5_match.group(1))
            results[current_run][current_epoch]["dev_ndcg5"] = dev_ndcg5

# Calculate mean and standard deviation across runs for each epoch
max_epoch = max(max(epochs.keys()) for epochs in results.values())
retrieve_loss_mean, retrieve_loss_std = [], []
ndcg5_mean, ndcg5_std = [], []

# Gather per-epoch statistics across runs
for epoch in range(max_epoch + 1):
    # Filter out None values for each metric
    retrieve_loss_vals = [val for val in (results[run][epoch]["retrieve_loss"] for run in results if epoch in results[run]) if val is not None]
    ndcg5_vals = [val for val in (results[run][epoch]["dev_ndcg5"] for run in results if epoch in results[run]) if val is not None]

    # Calculate mean and std
    retrieve_loss_mean.append(np.mean(retrieve_loss_vals))
    retrieve_loss_std.append(np.std(retrieve_loss_vals))
    ndcg5_mean.append(np.mean(ndcg5_vals))
    ndcg5_std.append(np.std(ndcg5_vals))

# Plotting retrieve loss
plt.figure(figsize=(10, 5))
plt.plot(range(max_epoch + 1), retrieve_loss_mean, label='Mean Retrieve Loss')
plt.fill_between(range(max_epoch + 1),
                 np.array(retrieve_loss_mean) - np.array(retrieve_loss_std),
                 np.array(retrieve_loss_mean) + np.array(retrieve_loss_std),
                 color='b', alpha=0.2, label='Retrieve Loss Std Dev')
plt.xlabel("Epoch")
plt.ylabel("Retrieve Loss")
plt.title("Retrieve Loss per Epoch (Averaged across Runs)")
plt.legend()
plt.grid(True)
plt.savefig(f'result/{dataset.lower()}_retrieve_loss.png')

# Plotting dev NDCG5
plt.figure(figsize=(10, 5))
plt.plot(range(max_epoch + 1), ndcg5_mean, label='Mean Dev NDCG5', color='g')
plt.fill_between(range(max_epoch + 1),
                 np.array(ndcg5_mean) - np.array(ndcg5_std),
                 np.array(ndcg5_mean) + np.array(ndcg5_std),
                 color='g', alpha=0.2, label='Dev NDCG5 Std Dev')
plt.xlabel("Epoch")
plt.ylabel("Dev NDCG5")
plt.title("Dev NDCG5 per Epoch (Averaged across Runs)")
plt.legend()
plt.grid(True)
plt.savefig(f'result/{dataset.lower()}_dev_ndcg5.png')

# Plotting dev NDCG5 for each run
plt.figure(figsize=(10, 5))

# Plot dev NDCG5 for each run individually
for run in results:
    ndcg5_vals = [results[run][epoch]["dev_ndcg5"] for epoch in sorted(results[run].keys())]
    plt.plot(sorted(results[run].keys()), ndcg5_vals, label=f'Run {run}')

plt.xlabel("Epoch")
plt.ylabel("Dev NDCG5")
plt.title("Dev NDCG5 per Epoch for Each Run")
plt.legend()
plt.grid(True)  # Show grid for better readability
plt.savefig(f'result/{dataset.lower()}_dev_ndcg5_each_run.png')