import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        qa_values = self.critic.qa_values(observation)
        action = qa_values.argmax(dim=1)

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        return action.squeeze()
