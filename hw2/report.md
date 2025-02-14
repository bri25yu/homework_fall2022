### Exercise 5.1
The parameters used were the exact same parameters as the default provided in the instructions.
For run replication, see `scripts/run_5_1.sh`.


![](report_resources/q5_1.jpg)


1. Which value estimator has better performance without advantage-standardization: the trajectory centric one, or the one using reward-to-go?

The reward to go value estimator had better performance than the trajectory centric value estimator. For the small batch size case, neither configuration converged to the optimal 200 reward value, but the trajectory centric value estimator had much lower mins and more frequent dips in performance. For the large batch case, the reward to go estimator converged while the trajectory centric one didn't.


2. Did advantage standardization help?

Advantage standardization helped a lot in the small batch case but not as much in the large batch case. For the small batch case, the reward to go value estimator without advantage standardization didn't converge while the reward to go value estimator with advantage standardization did. For the large batch case, the reward to go value estimators for both the without and with advantage standardization cases converged, but the config with advantage standardization had fewer dips in performance and when it did dip, it dipped less than the config without advantage standardization.


3. Did the batch size make an impact?

Batch size did make an impact, specifically in the case where we use the reward to go value estimator without advantage standardization. The estimator didn't converge in the small batch case but it did converge in the large batch case.


<div style="page-break-after: always;"></div>


### Exercise 5.2
For run replication, see `scripts/run_5_2.sh`.


![](report_resources/q5_2_heatmap.jpg)


![](report_resources/q5_2_learning_curves.jpg)


<div style="page-break-after: always;"></div>


### Exercise 7.3
For run replication, see `scripts/run_7_3.sh`.


![](report_resources/q7_3.jpg)


<div style="page-break-after: always;"></div>


### Exercise 7.4.1
For run replication, see `scripts/run_7_4_1.sh`.

Having a larger learning rate typically improved performance. However, the best performance was when the learning rate matched the a particular batch size, namely with a batch size of 30000 and a learning rate of 2e-2. This is very curious because I would've expected larger batch sizes to do better in all scenarios, but maybe there's not enough noise in 50000 samples and too much noise in 10000 samples.


![](report_resources/q7_4_1_heatmap.jpg)


![](report_resources/q7_4_1_learning_curves.jpg)


<div style="page-break-after: always;"></div>


### Exercise 7.4.2
For run replication, see `scripts/run_7_4_2.sh`.


![](report_resources/q7_4_2.jpg)


<div style="page-break-after: always;"></div>


### Exercise 8.5
For run replication, see `scripts/run_8_5.sh`.


As we increase lambda, our variance decreases and our final average return increases.


![](report_resources/q8_5.jpg)


<div style="page-break-after: always;"></div>

