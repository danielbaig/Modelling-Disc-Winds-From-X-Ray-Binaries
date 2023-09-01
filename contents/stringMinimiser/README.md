# String Minimisation
This section of the project involved investigating a technique to emulate the variable parameter space (VPS) and predict its point distribution without simulating it using PYTHON. The idea is to choose two points in the parameter file parameter space (PFPS) that were simulated and ask the question: What happens to the VPS as we traverse a path in PFPS starting and ending on 'known' VPS point distributions?

## Theory
We can initially guess a straight line in between the two points. Then we can use other simulated points to perturb the string towards their VPS value depending on how good a guess they are (depending on their PFPS distance). As more simulated data points in PFPS are added, the prediction made by the string should improve. The string was modelled by a sequence of nodes and springs and the data points acted as generic potentials.

The potential that was minimised by a Monte Carlo procedure was:
$$V = \sum\limits_{j=0}^m\sum\limits_{i=0}^n \frac{\sigma(\delta_{ij})\sigma(r_{ij})}{\rho_j} + \tilde{k}\sum\limits_{i=0}^{n-1} (|\underline{r}_{i+1} - \underline{r}_i| - l)^2$$

where $n$ is the number of nodes (one fewer springs), $m$ is the number of sampled points in PFPS, $\delta_{ij}$ is the distance in PFPS between the ith node and the jth potential source, $r_{ij}$ is the distance in VPS between the ith node and the jth potential source, $\rho_j$ is the local density of potentials, $\tilde{k}$ is the half spring constant to field energy ratio, $\underline{r}_i$ is the position of the ith node and $l$ is the relaxed length of the springs. Additionally, $\sigma(x)$ is the sigmoid function (translated and stretched) approximated as:

$$\sigma(x) = \frac{2}{1+e^{-x}} - 1 \approx \frac{x}{1+x} &#x202F&#x202F ; &#x202F x\geq0$$
