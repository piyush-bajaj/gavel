import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from isolated import IsolatedPolicy

class FinishTimeFairnessWithCostPolicy(Policy):

    def __init__(self, solver):
        self._name = 'FinishTimeFairnessCost'
        self._finish_time_fairness_perf_policy = \
            FinishTimeFairnessPolicyWithPerf(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec, instance_costs):
        #this is redundant code, done in perf method               
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        #this is redundant code, done in perf method
        if throughputs is None: return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 # Hardcode worker_type to v100 since all other worker types
                 # have a throughput of 0 for some job.
                 new_unflattened_throughputs[job_id][worker_type] = \
                     unflattened_throughputs[job_id]['v100']

        return self._finish_time_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors,
            unflattened_priority_weights,
            times_since_start,
            num_steps_remaining, cluster_spec, instance_costs)


class FinishTimeFairnessPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'FinishTimeFairness_Perf'
        self._isolated_policy = IsolatedPolicy()
        self._cumulative_isolated_time = {}
        self._isolated_throughputs_prev_iteration = {}
        self._num_steps_remaining_prev_iteration = {}

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec, instance_costs = None):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            self._isolated_throughputs_prev_iteration = {}
            self._num_steps_remaining_prev_iteration = {}
            return None
        (m, n) = throughputs.shape  #m=no. of jobs, n=no. of worker types
        (job_ids, worker_types) = index

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        # TODO: Do something with these priority_weights.
        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        # Create allocation variable, and isolated allocation.
        x = cp.Variable(throughputs.shape)  # optimization solver initilization
        # calculate the summed throughputs as per the allocation, when whole cluster divided among m jobs
        # so isolated throughputs change as the number of jobs change in the cluster
        isolated_throughputs = self._isolated_policy.get_throughputs(
            throughputs, index, scale_factors, cluster_spec, instance_costs)
        expected_time_fractions = []
        for i in range(len(job_ids)):
            if job_ids[i] not in self._cumulative_isolated_time:
                self._cumulative_isolated_time[job_ids[i]] = 0
            if job_ids[i] in self._num_steps_remaining_prev_iteration:
                # increment isolated time by ( steps runs / throughput)
                self._cumulative_isolated_time[job_ids[i]] += (
                    self._num_steps_remaining_prev_iteration[job_ids[i]] -
                    num_steps_remaining[job_ids[i]]) / \
                    self._isolated_throughputs_prev_iteration[job_ids[i]]

            # divide throughput by cost since. This gives the allocation based on cost for throughput
            instance_costs_array = np.ones((1, n))
            if instance_costs is not None:
                for j in range(n):
                    instance_costs_array[0, j] = instance_costs[worker_types[j]]
            # 

            allocation_throughput = cp.sum(cp.multiply(cp.multiply(throughputs[i], cp.inv_pos(instance_costs_array[0])), x[i]))
            # calculate isolated time for the job = isolated time so far + (no. of steps remaining / no. of steps per second)
            # no. of steps per second = throughput
            expected_time_isolated = self._cumulative_isolated_time[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] / isolated_throughputs[i])
            # calculate shared allocation time for the job = time so far for the job + (no. of steps remaining * iteration time current allocation for shared load)
            expected_time_allocation = times_since_start[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] * cp.inv_pos(allocation_throughput))
            # calculation of p = Tsh / Tid , finish_time_fairness metric
            expected_time_fraction = expected_time_allocation / expected_time_isolated
            expected_time_fractions.append(expected_time_fraction)
        if len(expected_time_fractions) == 1:
            # finish_time_fairness for 1 job
            objective = cp.Minimize(expected_time_fractions[0])
        else:
            # finish_time_fairness for all jobs
            objective = cp.Minimize(cp.maximum(*expected_time_fractions))

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        self._num_steps_remaining_prev_iteration = copy.copy(num_steps_remaining)
        self._isolated_throughputs_prev_iteration = {}
        for i in range(m):
            self._isolated_throughputs_prev_iteration[job_ids[i]] = \
                isolated_throughputs[i]

        if x.value is None:
            return self._isolated_policy.get_allocation(
                unflattened_throughputs, scale_factors, cluster_spec)
        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)

