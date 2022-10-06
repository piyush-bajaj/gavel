import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np

from policy import Policy

class IsolatedPolicy(Policy):

    def __init__(self):
        self._name = 'Isolated'

    def get_throughputs(self, throughputs, index, scale_factors,
                        cluster_spec):
        if throughputs is None: return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]
        # calculate the allocation for each each job
        x_isolated = self._get_allocation(
            throughputs, index,
            scale_factors_array,
            cluster_spec)
        # total isolated throughput is the sum of individual throughputs by allocation parts
        # job distributed among the resources, suppose allocation is 0.5 of the resource for 2 jobs case then throughput of both is reduced 
        # by factor of 0.5 and the total isolate throughput will be sum of each of these scaled throughputs 
        isolated_throughputs = np.sum(np.multiply(throughputs, x_isolated),
                                      axis=1).reshape((m, 1))
        return isolated_throughputs

    # calculates the allocation of each job distributed on each worker type for counts of worker type
    # consider you have 2 jobs and 1 resource then you will distribute 0.5 resource for each job. This function returns this 0.5
    def _get_allocation(self, throughputs, index, scale_factors_array,
                        cluster_spec):
        (_, worker_types) = index
        (m, n) = throughputs.shape

        # Split cluster over users (m). By construction,
        # \sum_i (x[i, j] * scale_factor[i]) = num_workers[j].
        # Normalize to ensure \sum_j x[i, j] <= 1 for all i.
        x = np.array([[cluster_spec[worker_type] / m for worker_type in worker_types]
                      for i in range(m)])
        x = x / scale_factors_array
        per_row_sum = np.sum(x, axis=1)
        per_row_sum = np.maximum(per_row_sum, np.ones(per_row_sum.shape))
        x = x / per_row_sum[:, None]

        return x

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        x = self._get_allocation(throughputs, index,
                                 scale_factors_array,
                                 cluster_spec)

        return super().unflatten(x, index)
