https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/48ec786c-68f9-4148-bf16-dd60d801673e

A program that solves the Lunar Landing problem with the Gym API. I compare Deep Q-Learning and Q-Learning and see which one performs better after selecting the appropriate parameters for this problem.

Parameters tested:
- Parameter r (division of the domain into buckets of size r)
- Parameter lr (learning rate, alpha parameter)
- Parameter y (discount for future state, gamma)
- Parameter u (buckets are not every r, but are every r^u)
- Parameter d (the randomness of taking actions every epoch is multiplied before d until it reaches the level of 0.05)
- Parameter n (size of the hidden layer in the neural network)

Below are charts showing how the model copes with the appropriate parameters.
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/5a2b8644-55d3-4858-8c4f-57adaa53abab)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/86c591e2-0c76-44fc-9add-91020cee6782)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/ae120986-5764-4736-89f0-292f24103d0a)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/3cffae4a-620a-4a8c-9fe3-ddc01d3bfe6f)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/80a01c27-ab51-471b-a5fa-760567f6fb3a)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/842a4998-d095-4c9f-b5b2-23e055f4df96)
![image](https://github.com/MrKrisuuu/ReinforcementLearning/assets/92759002/cc7b5f47-4b46-42ff-a8bc-3582386341a0)

The result of training is shown at the beginning.
