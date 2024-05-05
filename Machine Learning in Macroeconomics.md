## Machine Learning in Macroeconomics

2024.5.5

### Motivation：

Heterogeneous agents models with nominal frictions and many assets.(Nonlinear,Heterogeneous agents,large state space)

how to give accurate and fast solutions of these models?

Traditional method like Dynamic Programming will face the "curse of dimensionality"

So we want to find ways to keep the   "curse of dimensionality" under control and move to the "feasible" region of the Big-O complexity chart.(Generally, A complexity lower than $O(nlogn)$​​ is acceptable)

Focus on better numerical algorithms: deep learning.(We have some other methods to deal with the "curse of dimensionality" such as FPGAS, functional programming )

### A Basic Example:

Most economics problems need us to solve such a question:
$$
\mathcal{H}(d) = 0, \mathcal{H}:J_{1}\rightarrow J_{2},d:\Omega \rightarrow \mathcal{R}
$$


where $J_{1,2}$ are two functional space.



Consider a stochastic neoclassical growth model:
$$

\left\{ \begin{array}{c}
	\begin{array}{c}
	\max E_0\sum_{t=0}^{\infty}{\beta ^tu\left( c_t \right)}\\
	c_t+k_{t+1}=e^{z_t}{k^{\alpha}}_t+\left( 1-\delta \right) k_t\\
\end{array}\\
	z_t=\rho z_{t-1}+\sigma \varepsilon _t,\varepsilon _t\sim N\left( 0,1 \right)\\
\end{array} \right. 

$$
F.O.C. we have:
$$

u\prime\left( c_t \right) =\beta E_t\left\{ u\prime\left( c_{t+1} \right) \left( 1+\alpha e^{z_{t+1}}{k^{\alpha -1}}_{t+1}-\delta \right) \right\} 

$$
Define :
$$
d=\left\{ \begin{array}{c}
	d^1\left( k_t,z_t \right) =c_t\\
	d^2\left( k_t,z_t \right) =E_t\left\{ u\prime\left( c_{t+1} \right) \left( 1+\alpha e^{z_{t+1}}{k^{\alpha -1}}_{t+1}-\delta \right) \right\}\\
\end{array} \right.
$$
Then:
$$
\mathcal{H}(d) = u\prime\left( d^1\left( k_t,z_t \right) \right) - \beta d^2\left( k_t,z_t \right) = 0
$$
Consider Value Function:
$$

\left\{ \begin{array}{c}
	\begin{array}{c}
	V\left( k_t,z_t \right) =\underset{k_{t+1}}{\max}\left\{ u\left( c_t+\beta E_tV\left( k_{t+1},z_{t+1} \right) \right) \right\}\\
	c_t+k_{t+1}=e^{z_t}{k^{\alpha}}_t+\left( 1-\delta \right) k_t\\
\end{array}\\
	z_t=\rho z_{t-1}+\sigma \varepsilon _t,\varepsilon _t\sim N\left( 0,1 \right)\\
\end{array} \right. 

$$
In this case , $d(k_{t},z_{t})$ is just $V(k_{t},z_{t})$.



The general idea of using Deep Learning to solve these questions is substitute $d(x)$ by $d^{n}(x,\theta)$  than we can learn the function:
$$

d\cong d^n\left( x;\theta \right) =\theta _0+\sum_{m=1}^M{\theta _m\phi \left( z_m \right)},z_m=\sum_{n=0}^N{\theta _{n,m}x_n}

$$
Some core steps:

Loss function: the implied error in the equilibrium/optimality conditions, as loss function.

Optimization: SGD

Simulation: simulated paths of the economy