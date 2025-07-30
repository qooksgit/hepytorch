from scipy.stats import crystalball
import multiprocessing as mp
import numpy as np
import torch


def normalize_data(data, mean, std):
    return (data - mean) / std

def unnormalize_data(data, mean, std):
    return (data * std) + mean

def normalize_mass(mass):
    return (mass - mass.mean()) / (mass.std() + 1e-6)  # Avoid division by zero

def generate_data(batch_size=5000):
    num_bins = 200
    mass_range = [120, 220]
    bins = torch.linspace(mass_range[0], mass_range[1], num_bins + 1, device=device)
    total_events = 1_000_000

    # \~E Generate Data (Ensure correct type)
    train_obs, train_true = parallel_generate_data(total_events, mass_range, 172.5, 1.3, 48)
    #val_true, val_obs = parallel_generate_data(int(total_events/4), mass_range, 172.5, 1.3, 48)
    test_obs, test_true = parallel_generate_data(int(total_events/4), mass_range, 173.5, 1.8, 48)

    # save train_true, train_obs to a csv file, The file contains two columns: true mass and observed mass with a delimiter of comma
    np.savetxt("train_data.csv", np.column_stack((train_obs, train_true)), delimiter=",", header="observed_mass,true_mass", comments='')
    np.savetxt("test_data.csv", np.column_stack((test_obs, test_true)), delimiter=",", header="observed_mass,true_mass", comments='')

    # concatenate train data and the test data and save to a csv file
    all_obs = np.concatenate((train_obs, test_obs))
    all_true = np.concatenate((train_true, test_true))
    np.savetxt("all_data.csv", np.column_stack((all_obs, all_true)), delimiter=",", header="observed_mass,true_mass", comments='')
    print("train sample stats: ",train_true.mean(), train_obs.mean())
    print("train sample stds: ",train_true.std(), train_obs.std())
    print("test sample means: ",test_true.mean(), test_obs.mean())
    print("test sample stds: ",test_true.std(), test_obs.std())

    ## \~E Compute Normalization Stats
    #train_true_mean = np.mean(train_true)
    #train_true_std = np.std(train_true)

    ## \~E Normalize Data
    #train_true = normalize_data(train_true, train_true_mean, train_true_std)
    #train_obs = normalize_data(train_obs, train_true_mean, train_true_std)

    #val_true = normalize_data(val_true, train_true_mean, train_true_std)
    #val_obs = normalize_data(val_obs, train_true_mean, train_true_std)

    #test_true = normalize_data(test_true, train_true_mean, train_true_std)
    #test_obs = normalize_data(test_obs, train_true_mean, train_true_std)

    ## \~E Convert to PyTorch Tensors
    #train_true = torch.tensor(train_true, dtype=torch.float32, device=device).view(-1, 1)
    #train_obs = torch.tensor(train_obs, dtype=torch.float32, device=device).view(-1, 1)

    #val_true = torch.tensor(val_true, dtype=torch.float32, device=device).view(-1, 1)
    #val_obs = torch.tensor(val_obs, dtype=torch.float32, device=device).view(-1, 1)

    #test_true = torch.tensor(test_true, dtype=torch.float32, device=device).view(-1, 1)
    #test_obs = torch.tensor(test_obs, dtype=torch.float32, device=device).view(-1, 1)

    ## \~E Create Datasets
    #train_dataset = TensorDataset(train_true, train_obs)
    #val_dataset   = TensorDataset(val_true, val_obs)
    #test_dataset  = TensorDataset(test_true, test_obs)

    ## \~E Create DataLoaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #return train_loader, val_loader, test_loader, bins, train_true_mean, train_true_std

    return

def breit_wigner(m, m_t, gamma_t):
    return 1 / ((m - m_t) ** 2 + (gamma_t ** 2 / 4))

def simulate_detector_response(true_data, beta, m, loc, scale, rng):
    observed_masses = np.empty_like(true_data)
    for i, true_mass in enumerate(true_data):
        cb = crystalball(beta, m, loc=true_mass + loc, scale=scale[i])
        observed_masses[i] = cb.rvs(random_state=rng)
        while observed_masses[i] < 120 or observed_masses[i] > 220:
            observed_masses[i] = cb.rvs(random_state=rng)
    return observed_masses

def generate_data_chunk(num_events, mass_range, m_t, gamma_t, seed):
    np.random.seed(seed)
    true_masses = np.random.uniform(mass_range[0], mass_range[1], num_events)
    true_weights = breit_wigner(true_masses, m_t, gamma_t)
    true_data = np.random.choice(true_masses, size=num_events, p=true_weights / true_weights.sum())
    rng = np.random.default_rng(seed)
    scale = np.clip(0.40 * np.sqrt(true_data), 1.0, 8.0)
    observed_data = simulate_detector_response(true_data, beta=1.0, m=1.5, loc=0, scale=scale, rng=rng)
    return true_data, observed_data

def parallel_generate_data(total_events, mass_range, m_t, gamma_t, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()
    chunk_size = total_events // num_processes
    pool = mp.Pool(processes=num_processes)
    seeds = [np.random.randint(0, 10000) for _ in range(num_processes)]
    args = [(chunk_size, mass_range, m_t, gamma_t, seeds[i]) for i in range(num_processes)]
    results = pool.starmap(generate_data_chunk, args)
    pool.close()
    pool.join()
    true_data = np.concatenate([result[0] for result in results])
    observed_data = np.concatenate([result[1] for result in results])
    return observed_data, true_data


# main function to run the data generation
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_data()
    print("Data generation complete.")
