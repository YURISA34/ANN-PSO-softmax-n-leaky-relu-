I use the dataset provided by Fisher (1936) based on the well-known Iris dataset (https://gist.github.com/curran/a08a1080b88344b0c8a7). I use this dataset to demonstrate the multi-classification in the output of data and the expected Output.
The data contains 3 classifications and it will make a great example of how you differentiate the beginner level of classification in the ML (ANN especially).
We use an ANN in 4-4-4-3 with non-bias and bias, relu, leakyrelu, and softmax.
We got 3 examples if you want to see dani (leaky relu), peysen (relu), and lasly ANNnPSO for the use of only softmax and 44 w of the parameter.
We use the fixed parameter provided by my DR which is (w = 0.5; c1 = 1.2; c2 = 1.2;). It is said that
w=0.5 is for the middle side of ML it keeps movement smooth and allows some exploration without chaos, c1 = 1.2 is for the value that attracts the nodes to pbest and c2 = 1.2 is for the rate of how much the nodes/particles influence the nodes/particles of others
These are common defaults from real research papers and practical PSO implementations — especially in small-to-medium scale problems (like your ANN with 52 weights).
They satisfy:
Exploration vs. exploitation balance
Swarm stability (no chaos or stagnation)
Safe convergence rate (neither too fast nor too slow)

Remember these are for beginner entry of ML for ANN model and the algorithm of PSO
The purpose of these is to make data for difference/multi classification in the output and what to expect to optimize the model and algorithm with functions such as Leaky-relu,relu, sigmoid, bias, and softmax.

sincely dani
