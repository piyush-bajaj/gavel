# Shared Cloud Cost Computing Policy for Deep Learning Training Jobs in Cloud Clusters

This repository contains the source code implementation of the Research Project "Cluster Scheduling in Data Centers". This code is built on the basis of https://github.com/stanford-futuredata/gavel. As such it is an extension to this code.

## Directory structure

### `scheduler`
Code for scheduler including utils, runtime environment files, could instance prices, policies, driver scripts and simulation experiment scripts.

### `analysis`
This folder contains entirely the new code which is used for analysis of throughputs and cluster utilization. This also hosts the files related to the evaluation of new policy. Specifically, the 3 folders evaluation, throughput and utilization contain the ipynb files showing the results and their comparisons.

### `workloads`
Implementations of target workloads in PyTorch, including changes needed to integrate with the GavelIterator

## New / Updated files for implementation
The following files were added or updated in order to run the new code as per the new algorithm.
1. scheduler/scheduler.py(updated)
2. scheduler/utils.py(updated)
3. scheduler/scripts/sweeps/run_sweep_continuous.py(updated)
4. scheduler/policies/shared_cloud_cost_fairness.py(added)
5. scheduler/prices/aws/logs/us-east-1/prices.json(added)
6. scheduler/prices/azure/logs/us-east-1.csv(added)

### File details
Files 1, 2 and 3 accepts and handle changes for including the new policy in gavel scheduler. 
File 4 contains the main logic for the algorithm to generate allocation based on new metric.
Files 5 and 6 contain the spot prices for the gpu instances on AWS and Azure.

## Algorithm for the code.
Please refer to the figure Algorithm.PNG at https://github.com/piyush-bajaj/gavel/tree/rp-submission

## Setup
Gavel needs Python to run. We ran the experiments on a machine with Ubuntu 22.04.
Python 3.8, which can be installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Required software dependencies can be installed using,

```bash
apt-get -y install cmake g++ gcc libnuma-dev make numactl zlib1g-dev
pip install -r scheduler/requirements.txt
cd scheduler; make
```

## Running Experiments
The below command can be used to run the experiment for different seeds, different number of jobs and different policies, after cd into the scheduler directory.

### static sweep
```bash
python -u scripts/sweeps/run_sweep_static.py -l /home/piyush/rp/work/logs/comparison/static -j 1 -p shared_cost_fairness finish_time_fairness --seeds 5 9 8 -c 5:5:5 -a 1000 -b 2000 -n 5 --available_clouds aws --per_instance_type_prices_dir /home/piyush/rp/work/scheduler/prices --solver SCS
```

### continuous sweep
```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 1000 -e 2000 -l /home/piyush/rp/work/logs/comparison/continuous -j 5 -p shared_cost_fairness finish_time_fairness --seeds 5 9 8 -c 5:5:5 -a 0.0 -b 1.0 -n 5 --available_clouds aws --per_instance_type_prices_dir /home/piyush/rp/work/scheduler/prices --solver SCS
```

#### the following parameters can be specified

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_DIR, --log-dir LOG_DIR
                        Log directory
  -s WINDOW_START, --window-start WINDOW_START
                        Measurement window start (job ID)
  -e WINDOW_END, --window-end WINDOW_END
                        Measurement window end (job ID)
  -j PROCESSES, --processes PROCESSES
                        Number of processes to use in pool (use as many as available if not specified)
  -p POLICIES [POLICIES ...], --policies POLICIES [POLICIES ...]
                        List of policies to sweep
  -c CLUSTER_SPEC [CLUSTER_SPEC ...], --cluster-spec CLUSTER_SPEC [CLUSTER_SPEC ...]
                        Cluster specification in the form of #v100s:#p100s:#k80s
  --seeds SEEDS [SEEDS ...]
                        List of random seeds
  -i INTERVAL, --interval INTERVAL
                        Interval length (in seconds)
  --throughputs-file THROUGHPUTS_FILE
                        Oracle throughputs file
  --solver {ECOS,GUROBI,SCS}
                        CVXPY solver
  -v, --verbose         Verbose
  -a THROUGHPUT_LOWER_BOUND, --throughput-lower-bound THROUGHPUT_LOWER_BOUND
                        Lower bound for throughput interval to sweep
  -b THROUGHPUT_UPPER_BOUND, --throughput-upper-bound THROUGHPUT_UPPER_BOUND
                        Upper bound for throughput interval to sweep
  -n NUM_DATA_POINTS, --num-data-points NUM_DATA_POINTS
                        Number of data points to sweep through
  --available_clouds    List of available clouds choices : [aws, azure, gcp]
  --per_instance_type_prices_dir    Directory location of prices file for aws and azure
