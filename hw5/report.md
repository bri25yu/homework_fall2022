# Q1.1 RND vs Epsilon-greedy exploration

State densities for PointmassEasy environment with Epsilon-greedy
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_env1_random_PointmassEasy-v0_21-11-2022_05-59-34/curr_state_density.png" width="250" height="200" />
</div>

State densities for PointmassEasy environment with RND
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_env1_rnd_PointmassEasy-v0_21-11-2022_05-31-11/curr_state_density.png" width="250" height="200" />
</div>

State densities for PointmassMedium environment with Epsilon-greedy
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_env2_random_PointmassMedium-v0_21-11-2022_06-41-44/curr_state_density.png" width="250" height="200" />
</div>

State densities for PointmassMedium environment with RND
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_env2_rnd_PointmassMedium-v0_21-11-2022_06-25-23/curr_state_density.png" width="250" height="200" />
</div>

<div style="text-align: center">
    <img src="report_resources/q1_1.png" width="250" height="200" />
</div>

The state densities for epsilon-greedy and RND for the easy environment are roughly the same. Both are able to explore the easy environment proficiently. The learning curve for RND learns and converges much earlier and faster than that of epsilon-greedy, signaling that even in the easy environment RND poses an advantage over epsilon-greedy. RND also converges to the optimal solution before exploration has even finished, signaling that it did a very good job of exploring towards the end goal. Contrast this with epsilon-greedy, where the return only starts improving until around 5000 steps after exploration has finished. 

The state density structures for the medium environment are about the same, but the confidence of the correct path is different, as shown by the different in brightness between the two state density graphs. RND has more confidence in the states it has explored towards the goal than epsilon-greedy. 

This is also reflected in the fact that RND is much more stable than epsilon-greedy -- epsilon-greedy has two major regions of oscillation and non-monotonicity while RND has no such regions.

<div style="page-break-after: always;"></div>


# Q1.2 RND vs RND L1 exploration

State densities for PointmassMedium environment with RND
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_env2_rnd_PointmassMedium-v0_21-11-2022_06-25-23/curr_state_density.png" width="250" height="200" />
</div>

State densities for PointmassMedium environment with RND L1
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_alg_med_PointmassMedium-v0_21-11-2022_07-05-49/curr_state_density.png" width="250" height="200" />
</div>

Last exploration trajectory for PointmassMedium environment with RND
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_env2_rnd_PointmassMedium-v0_21-11-2022_06-25-23/expl_last_traj.png" width="250" height="200" />
</div>

Last exploration trajectory for PointmassMedium environment with RND L1
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_alg_med_PointmassMedium-v0_21-11-2022_07-05-49/expl_last_traj.png" width="250" height="200" />
</div>

<div style="text-align: center">
    <img src="report_resources/q1_2.png" width="250" height="200" />
</div>

We modify the RND algorithm slightly. Instead of penalizing model prediction error by the square of the difference, we only penalize with the absolute value of the difference. The overall penalty is less.

This means that the model remembers less about past states visited, so it will bounce around states more often, especially around corners, as shown in the state density plot. This is also exactly what is observed in the last exploration trajectories.

This is a disadvantage earlier in each trajectory, where we observe the model getting stuck in corners. This is an advantage when the environment required to learn is more complex. Thus, the L1 and L2 penalty version of RND represent tradeoffs for modeling capacity and exploitation, respectively. 

This is further demonstrated on the PointmassHard environment, where the L1 RND does an incredibly proficient job at optimizing the trajectory, shown below. 

State densities for PointmassHard environment with RND L1
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_alg_hard_PointmassHard-v0_21-11-2022_07-19-14/curr_state_density.png" width="250" height="200" />
</div>

Last exploration trajectory for PointmassHard environment with RND L1
<div style="text-align: center">
    <img src="run_logs/hw5_expl_q1_alg_hard_PointmassHard-v0_21-11-2022_07-19-14/expl_last_traj.png" width="250" height="200" />
</div>

<div style="page-break-after: always;"></div>
