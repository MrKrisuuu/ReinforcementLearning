A program that solves the Lunar Landing problem with the Gym API. I compare Deep Q-Learning and Q-Learning and see which one performs better after selecting the appropriate parameters for this problem.

Parameters tested:
- Parameter r (division of the domain into buckets of size r)
- Parameter lr (learning rate, alpha parameter)
- Parameter y (discount for future state, gamma)
- Parameter u (buckets are not every r, but are every r^u)
- Parameter d (the randomness of taking actions every epoch is multiplied before d until it reaches the level of 0.05)
- Parameter n (size of the hidden layer in the neural network)

Below are charts showing how the model copes with the appropriate parameters.
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/94d9fb42-83f6-4817-9bb4-1364f4e7875e)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/db8cc801-79ed-4c6f-bd81-0fabcaf0601b)

