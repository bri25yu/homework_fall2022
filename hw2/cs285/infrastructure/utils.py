from typing import Dict, List, Tuple

import copy

import numpy as np
from tqdm import tqdm
import gym


############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1

    # initialize env for the beginning of a new rollout
    ob = env.reset() # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    if render: pbar = tqdm(desc="Sampling trajectory", total=max_path_length)
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob) # HINT: query the policy's get_action function
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = done or (steps >= max_path_length) # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if render: pbar.update()

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        timesteps_this_path = get_pathlength(path)

        paths.append(path)
        timesteps_this_batch += timesteps_this_path        

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1

    paths = []

    for _ in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length, render))

    return paths


############################################
############################################
# Vectorized versions of the previous sample trajectory functions

class SampleTrajectoryVectorizedData:
    TRACKED_OBJECTS = [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminals",
    ]

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

        self.data: Dict[List[List[np.ndarray]]] = {
            name: self.init_vectorized_data_object() for name in self.TRACKED_OBJECTS
        }

    def init_vectorized_data_object(self) -> List[List]:
        num_envs = self.num_envs

        return [[] for _ in range(num_envs)]

    def update_object(self, key: str, updates: np.ndarray) -> None:
        num_envs = self.num_envs
        obj_to_update = self.data[key]
        terminals = self.data["terminals"]

        for i in range(num_envs):
            terminated = terminals[i] and terminals[i][-1]
            if terminated:
                continue

            obj_to_update[i].append(updates[i])

    def to_paths_list(self) -> List[Dict[str, np.ndarray]]:
        num_envs = self.num_envs
        data = self.data
        tracked_objects = self.TRACKED_OBJECTS

        def create_path(args: Tuple[np.ndarray]) -> Dict[str, np.ndarray]:
            return Path(args[0], [], *args[1:])

        paths: List[Dict[str, np.ndarray]] = []
        for i in range(num_envs):
            args = tuple(data[k][i] for k in tracked_objects)
            paths.append(create_path(args))

        return paths


def sample_trajectory_vectorized(
    env: gym.vector.VectorEnv, policy, max_path_length: int
) -> List[Dict[str, np.ndarray]]:
    """
    N_p -> number of parallel gym envs
    T_sample -> length of the path length of a particular sample
    D_o -> observation dim
    D_a -> action dim
    """

    # Initialize env for new rollouts
    # `observations` is of shape (N_p, D_o)
    observations: np.ndarray = env.reset()
    num_envs = observations.shape[0]

    # Initialize our data
    data = SampleTrajectoryVectorizedData(num_envs)

    steps = 0
    while True:
        # Record our observations
        data.update_object("observations", observations)

        # Get the policy actions of shape (N_p, D_a)
        actions = policy.get_action(observations)
        data.update_object("actions", actions)

        # Take the actions
        observations, rewards, terminals, _ = env.step(actions)

        # Record results of taking that action
        steps += 1
        data.update_object("next_observations", observations)
        data.update_object("rewards", rewards)

        # Process terminals
        max_path_length_reached = steps >= max_path_length
        if max_path_length_reached:
            terminals = np.full(num_envs, True, dtype=bool)

        data.update_object("terminals", terminals)

        # If all the envs are terminated, we exit
        if terminals.all():
            break

    paths = data.to_paths_list()
    return paths


def sample_trajectories_vectorized(
    env: gym.vector.VectorEnv, policy, min_timesteps_total: int, max_path_length: int
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_total:
        paths_batch = sample_trajectory_vectorized(env, policy, max_path_length)
        paths.extend(paths_batch)
        timesteps_this_batch += sum(map(get_pathlength, paths_batch))

    return paths, timesteps_this_batch


############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data