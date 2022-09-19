# Transformers for online reinforcement learning

## Related work
Chen et al [1] create the Decision Transformer which takes a sequence of state-action-reward tuples as input and outputs a probability distribution over the next action. They train the Decision Transformer on a predefined replay buffer of state-action-reward tuples taken from an expert. Their loss calculation is performed between the model predicted action and the action of the expert.

Janner et al [2] create the Trajectory Transformer which takes a sequence of states, action, and rewards as input, performs beam search to generate candidate subsequent sequences, and picks the best sequences by maximizing over the final predicted reward value. They train the Trajectory Transformer on a predefined replay buffer of state, action, reward sequences taken from an expert. Their loss calculation is performed between the model predicted next value and the true value in the sampled sequence.

## Proposed methods
The issue with current supervised learning methods is that they typically can only be as good as the set of trajectories that are observed in the replay buffer. In this manner, supervised learning methods are limited to being offline and off policy. It's very easy for a supervised learning model to learn the distribution of states, actions, and rewards, but it's not as easy to inform the model on how to improve itself.

The Decision Transformer solves this problem by imitating the expert. The Trajectory Transformer solves this problem by beam search and maximizing over the final predicted reward value.

We pose a different task to a transformer in an online setting. We leverage a contrastive learning inspired algorithm between the best trajectory seen so far and the current trajectory to try to figure out why the best trajectory is better than the current trajectory.

The reason we use a transformer rather than just a multi-layer perceptron (MLP) is because we want to inform the model on a trajectory-level basis rather than a state-action-reward tuple level basis. It turns out that this is easier to do with transformers rather than MLPs. In addition, transformer architectures have been shown to be powerful and state of the art in other areas. 

The proposed algorithm is as follows:
1. Collect the first trajectory. This will be the incumbent.
2. while not converged:
   1. Collect an additional trajectory. This will be the candidate.
      1. The incumbent and candidate are different because of dropout, noise, etc.
   2. Compute a stepwise loss between the predicted and actual state-action-reward values. Call this the distribution loss.
      1. This is the loss that informs the model on the distribution of states, actions, and rewards.
   3. Compare the final reward values between the incumbent and candidate. Assign the incumbent label to the trajectory with higher reward and the candidate label to the trajectory with lower reward.
   4. Compute a stepwise loss between the incumbent and candidate trajectories, where the incumbent is our "expert" policy. Call this the improvement loss.
   5. Add the distribution and improvement losses together and update the policy.


## Proposed evaluation
We evaluate our methods on tasks from https://github.com/openai/gym and compare against state-of-the-art benchmark performances for each task. Each task has a final test score that is comparable between models and training techniques.


## References
[1] Chen et al. Jul 2021 [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

[2] Janner et al. May 2021 [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://openreview.net/forum?id=wgeK563QgSw)
