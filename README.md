# Maximum Entropy-Regularized Multi-Goal Reinforcement Learning

Here is the code for our ICML-2019 paper "Maximum Entropy-Regularized Multi-Goal Reinforcement Learning". 

The code was developed by Rui Zhao (Siemens AG & Ludwig Maximilian University of Munich). 

For details on Maximum Entropy-based Prioritization (MEP), please read the ICML paper (link: https://icml.cc/Conferences/2019/AcceptedPapersInitial) (forthcoming).  

The code is developed based on OpenAI Baselines (link: https://github.com/openai/baselines).   

## Prerequisites 
The code requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Tested on Ubuntu 16.04
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

To run the code, you need to install OpenAI Gym (link: https://github.com/openai/gym).  
We use the robotics environment in OpenAI Gym, which needs the MuJoCu physics engine (link: http://www.mujoco.org/).   

The experiments were carried out on Amazon Web Services (AWS).  
The EC2 instances we used have 20 CPUs. We use 19 CPUs for training.  
If you are running the experiments on a laptop, please configure a smaller number of CPUs.  
Note that, with less CPUs, the performance will be effected.  

After the installaton of dependicies, you can reproduce the experimental results by running the following commnands:  
```
python baselines/her/experiment/train.py --env_name FetchPickAndPlace-v0 --num_cpu 19 --prioritization none --replay_strategy none --seed 0
python baselines/her/experiment/train.py --env_name FetchPickAndPlace-v0 --num_cpu 19 --prioritization entropy --replay_strategy none --seed 0
python baselines/her/experiment/train.py --env_name FetchPickAndPlace-v0 --num_cpu 19 --prioritization tderror --replay_strategy none --seed 0
python baselines/her/experiment/train.py --env_name HandManipulatePenRotate-v0 --num_cpu 19 --prioritization none --replay_strategy future --seed 0
python baselines/her/experiment/train.py --env_name HandManipulatePenRotate-v0 --num_cpu 19 --prioritization entropy --replay_strategy future --seed 0
python baselines/her/experiment/train.py --env_name HandManipulatePenRotate-v0 --num_cpu 19 --prioritization tderror --replay_strategy future --seed 0
```

To test the learned policies, you can run the command:  
```
python baselines/her/experiment/play.py /path/to/an/experiment/policy_latest.pkl
```

## Citation:

Citation of the ICML paper:

```
@inproceedings{zhao2019maximum,
  title={Maximum Entropy-Regularized Multi-Goal Reinforcement Learning},
  author={Zhao, Rui and Sun, Xudong and Tresp, Volker},
  booktitle={International Conference on Machine Learning},
  year={2019}
}
```

## Licence:

MIT
