I use dataset provided by Fisher (1936) based on the well-known Iris dataset (https://gist.github.com/curran/a08a1080b88344b0c8a7). I use this dataset to demonstrated the multi-calssification in output of data and the expected Output.
The data contain 3 classification and it will make a great example on how you differenciate the begineer level of classification in the ml (ANN expecially).
We use a ANN in 4-4-4-3 with non biase and biase and relu and leakyrelu and softmax.
we got 3 example if you want to see dani (leaky relu), peysen (relu) and lasly ANNnPSO for the Used of only softmax and 44 w of the parameter.
we use the fixed parameter provided by my DR which is (w = 0.5; c1 = 1.2; c2 = 1.2;). It is said that
w=0.5 is for middle side of ML it keeps movement smooth and allows some exploration without chaos, c1 = 1.2 is for the value that attract the nodes to pbest and the c2 = 1.2 is to the rate of how much the nodes/particles influence the nodes/particles of others
These are common defaults from real research papers and practical PSO implementations — especially in small-to-medium scale problems (like your ANN with 52 weights).
They satisfy:
Exploration vs. exploitation balance
Swarm stability (no chaos or stagnation)
Safe convergence rate (neither too fast nor too slow)

Remember these are for begineer entry of ml for ANN model and algorithm of PSO
Purpose of these to make data for difference/multi classification in the output and what to expect to optimize the model and algorithm with function such as Leaky-relu,relu,sigmoid,biase, and softmax
