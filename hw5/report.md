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

The state densities for epsilon-greedy and RND for the easy environment are roughly the same. Both are able to explore the easy environment proficiently. The learning curve for RND learns and converges much earlier and faster than that of epsilon-greedy, signaling that even in the easy environment RND poses an advantage over epsilon-greedy.

The state density structures for the medium environment are about the same, but the confidence of the correct path is different, as shown by the different in brightness between the two state density graphs. RND has more confidence in the states it has explored towards the goal than epsilon-greedy. 

This is also reflected in the fact that RND is much more stable than epsilon-greedy -- epsilon-greedy has two major regions of oscillation and non-monotonicity while RND has no such regions.

<div style="page-break-after: always;"></div>
