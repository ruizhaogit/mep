import threading

import numpy as np

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

import math

from scipy.stats import rankdata

import json

from sklearn import mixture


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class ReplayBufferEntropy:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, prioritization, env_name):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        self.buffers['e'] = np.empty([self.size, 1])
        self.buffers['p'] = np.empty([self.size, 1]) # priority

        self.prioritization = prioritization
        self.env_name = env_name

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.current_size_test = 0
        self.n_transitions_stored_test = 0

        self.lock = threading.Lock()

        self.clf = 0
        self.pred_min = 0
        self.pred_sum = 0
        self.pred_avg = 0

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size, rank_method, temperature):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size, rank_method, temperature)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            if not key == 'p' and not key == 'e':
                assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def fit_density_model(self):
        ag = self.buffers['ag'][0: self.current_size].copy()
        X_train = ag.reshape(-1, ag.shape[1]*ag.shape[2])
        self.clf = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution", n_components=3)
        self.clf.fit(X_train)
        pred = -self.clf.score_samples(X_train)
        self.pred_min = pred.min()
        pred = pred - self.pred_min
        pred = np.clip(pred, 0, None)
        self.pred_sum = pred.sum()
        pred = pred / self.pred_sum
        self.pred_avg = (1 / pred.shape[0])

        with self.lock:
            self.buffers['e'][:self.current_size] = pred.reshape(-1,1).copy()

    def store_episode(self, episode_batch, rank_method, epoch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        buffers = {}
        for key in episode_batch.keys():
            buffers[key] = episode_batch[key]

        if not isinstance(self.clf, int):
            ag = buffers['ag']
            X = ag.reshape(-1, ag.shape[1]*ag.shape[2])
            pred = -self.clf.score_samples(X)

            pred = pred - self.pred_min
            pred = np.clip(pred, 0, None)
            pred = pred / self.pred_sum

            episode_batch['e'] = pred.reshape(-1,1)

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                if not key == 'p' and not key == 'e' or (key == 'e' and epoch > 0):
                    self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

            entropy_transition_total = self.buffers['e'][:self.current_size]
            if rank_method == 'none':
                rank_method = 'dense'
            
            entropy_rank = rankdata(entropy_transition_total, method=rank_method)
            entropy_rank = entropy_rank - 1
            entropy_rank = entropy_rank.reshape(-1, 1)
            self.buffers['p'][:self.current_size] = entropy_rank.copy()

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, alpha, env_name):
        """Create Prioritized Replay buffer.
        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_shapes, size_in_transitions, T, sample_transitions)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        self.size_in_transitions = size_in_transitions
        while it_capacity < size_in_transitions:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.T = T
        self.buffers['td'] = np.zeros([self.size, self.T])
        self.buffers['e'] = np.zeros([self.size, self.T])
        self.env_name = env_name

    def store_episode(self, episode_batch, dump_buffer):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """

        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        if dump_buffer:

            buffers = {}
            for key in episode_batch.keys():
                buffers[key] = episode_batch[key]
                episode_batch['e'] = np.zeros([buffers['ag'].shape[0], self.T])


        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys(): # ['g', 'info_is_success', 'ag', 'o', 'u']
                if not key == 'td':
                    if dump_buffer:
                        self.buffers[key][idxs] = episode_batch[key]
                    else:
                        if not key == 'e':
                            self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

            for idx in idxs:
                episode_idx = idx
                for t in range(episode_idx*self.T, (episode_idx+1)*self.T):
                    assert (episode_idx+1)*self.T-1 < min(self.n_transitions_stored, self.size_in_transitions)
                    self._it_sum[t] = self._max_priority ** self._alpha
                    self._it_min[t] = self._max_priority ** self._alpha

    def dump_buffer(self, epoch):
        for i in range(self.current_size):
            entry = {"e": self.buffers['e'][i].tolist(), \
                     "td": self.buffers['td'][i].tolist(), \
                     "ag": self.buffers['ag'][i].tolist() }
            with open('buffer_epoch_{0}.txt'.format(epoch), 'a') as file:
                 file.write(json.dumps(entry))  # use `json.loads` to do the reverse
                 file.write("\n")

        print("dump buffer")


    def sample(self, batch_size, beta):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """

        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions, weights, idxs = self.sample_transitions(self, buffers, batch_size, beta)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            if not key == 'td' and not key == 'e':
                assert key in transitions, "key %s missing from transitions" % key

        return (transitions, weights, idxs)


    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities.flatten()):
            assert priority > 0
            assert 0 <= idx < self.n_transitions_stored
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
