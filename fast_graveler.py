import numpy as np              # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm           # pip install tqdm

number_of_samples = 1000000000
number_of_cores = 6
# No need to edit past this point!!!

def generate_binomial_samples(bi_size: int = 10000) -> tuple:
    '''This is called by each thread to get our odds of rolling 1 out of a 4 sided dice.'''
    distribution = np.random.binomial(n=231, p=0.25, size=bi_size)
    return np.max(distribution), distribution

def quick_graveler(n_samples : int = 10000, n_workers: int = 1) -> tuple:
    '''Takes an integer number of samples (default 10,000) and returns the maximum int found and the distribution'''
    maxOnes = 0                                             # Maximum value found
    bi_size = 100000                                        # How large should the binomial array be?
    bi_distribution = None                                  # Save the array with the largest value in it
    if bi_size > n_samples: bi_size = n_samples             # If there are less samples then 100,000 reduce it
    n_chunks = int(n_samples / bi_size)                     # Number of chunks of work to split across workers
    
    # Use ProcessPoolExecutor to parallelize across multiple processes
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        batch_size = int(n_workers * 4)                    # Run in batches to prevent bluescreen, dont ask me why I know this
        jobs = []                                          # List that holds all jobs in queue
        print(f'{n_samples} samples will be processed in ~{int(n_chunks / batch_size)} batches!')
        
        for i in tqdm(range(0, n_chunks, batch_size),desc='Batch Number:'):
            for _ in range(min(batch_size, n_chunks - i)): # Submit batch of jobs for parallel execution
                jobs.append(executor.submit(generate_binomial_samples, bi_size))
            
            for future in jobs:                            # Process batch of results as they complete
                a, bi = future.result()                    # Unpack the result (max value, distribution)
                if a > maxOnes:
                    maxOnes = a
                    bi_distribution = bi
            jobs.clear()                                   # Clear the processed jobs to free memory

    return maxOnes, bi_distribution

maxOnes, distribution = quick_graveler(number_of_samples, number_of_cores)
print(f'The maximum value found was: {maxOnes}')

# Extra for plotting an example distribution to show I am actually computing all values
value, counts = np.unique(distribution, return_counts=True)
plt.plot(value, counts, label='Binomial distribution')
plt.scatter(maxOnes, 1, label=f'Maximum value: {maxOnes}')
plt.legend()
plt.show()