### Summary
Here I try three continuous control algorithms (DDPG, TD3 and SAC) on solving a flight control problem (i.e., hovering control of a drone with quadrotor engines)

They share the same hyperparameters and similar network structures

- gamma: 0.99
- tau: 0.005 (only 0.5% old model is preserved for each soft update)
- in DDPG and TD3, adversarial noises (std: 0.1) are added to enhance learning robustness. In SAC, this is done  

The main settings of this experiment is

* train_total_steps: 1 million
* test_every_steps: 10 thousands
* memory size: 1 million
*  warmup_size: 10 thousands

The agent is under training for up to 1 million steps (I haven't finished the training because I think ). The best test score reached is 6952 at 150+k steps by SAC, 4374 at 500+k steps by TD3 and -257 at 10+k by DDPG. 


### Result
![Scores](./imgs/SAC_vs_DDPG_vs_TD3_091320.png)


I have to run more experiments to verify my  hypothesis. Right now, I can't seem to draw any convincing conclusion about any algorithms in regards to solve this problem. All algorithms start very well but performance deteriorates after 400-500 k steps. 

### Lesson to learn
Starts easy and print more helpful information to guide tuning. A good question to always ask ourselves is how should I know continuing training will get us to its optimal performance? How hyperparameters should be tuned when continuing training. 

### Reference