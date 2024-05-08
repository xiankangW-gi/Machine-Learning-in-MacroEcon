## Deep Equilibrium Nets

Reference：Simon Scheidegger，2023

### 1.Motivation and key ideas

Contemporary dynamic models: heterogeneous & high-dimensional



Have to approximate and interpolate high-dimensional functions on irregular-shaped geometries



Problem: curse of dimensionality

#### Key ideas：

Use the the implied error in the optimality conditions, as loss  function



Learn the equilibrium functions with stochastic gradient descent



Take the (training) data points from a simulated path  → can be generated at virtual zero cost. 、

### 2.Baseline Model（Discrete time OLG model）

Agents live for N periods (N=60 years)  Every t, a representative household in born. There are exogenous aggregate shocks z that follow a Markov chain.

Each period, the agents alive receive a strictly  positive labour endowment which depends  on the age of the agent alone.

#### Household

$$

\left\{ \begin{array}{c}
	\sum_{i=0}^{N-s}{E_t\left[ \beta ^iu\left( c_{t+i}^{s+i} \right) \right]}\left( s\,\,is\,\,the\,\,agents\,\,current\,\,age \right) \,\,  \left( 1 \right)\\
	a_{t}^{s}=k_{t+1}^{s+1}\left( a\,\,is\,\,saving \right) \,\,                                     \left( 2 \right)\\
	a_{t}^{s}\geqslant \underline{a}\,\,                                                          \left( 3 \right)\\
	c_{t}^{s}+a_{t}^{s}\,\,=\,\,r_tk_{t}^{s}\,\,+l_{t}^{s}w_t\,\,                                    \left( 4 \right)\\
\end{array} \right. 

$$

#### Firm and Market

The total factor productivity η (TFP) and the depreciation δ depend  on the exogenous shock z alone($z \in {0,1,2,3}$​)
$$

\pi ^{\eta}=\left[ \begin{array}{l}
	0.905&		0.095\\
	0.095&		0.905\\
\end{array} \right] ,\pi ^{\delta}=\left[ \begin{array}{l}
	0.98&		0.02\\
	0.25&		0.75\\
\end{array} \right] ,

$$

$$

f\left( K,L,z \right) =\eta \left( z \right) K^{\alpha}L^{1-\alpha}+K\left( 1-\delta \left( z \right) \right) 
$$

#### Equilibrium

Maxmize (1) subject to (2), (3), (4) ; Maxmize firm profit ($ 
\left( K_t,L_t \right) \in arg\max f\left( K,L,z \right) -r_tk_t-w_tl_t
$) ,and All market clear:
$$

\left\{ \begin{array}{c}
	L_t=\sum_{s=1}^N{l_{t}^{s}}\\
	K_t=\sum_{s=1}^N{k_{t}^{s}}\\
\end{array} \right. 

$$
 

#### Equilibrium Condition

$$

\left\{ \begin{array}{c}
	u\prime\left( c^s\left( z^t \right) =\beta E_{z_t}\left[ u\prime\left( c^{s+1}\left( z^t,z^{t+1} \right) \right) r\left( z^t,z^{t+1} \right) \right] \right) +\lambda ^s\left( z^t \right)\\
	\lambda ^s\left( z^t \right) \left( a^s\left( z^t \right) -a \right) =0\\
	a^s\left( z^t \right) -a\geqslant 0\\
	\lambda ^s\left( z^t \right) \geqslant 0\\
	w\left( z^t \right) =\left( 1-\alpha \right) \eta \left( z^t \right) K\left( z^t \right) L\left( z^t \right) ^{-\alpha}\\
	r\left( z^t \right) =\alpha \eta \left( z^t \right) K\left( z^t \right) ^{\alpha -1}L\left( z^t \right) ^{1-\alpha}+\left( 1-\delta \left( z^t \right) \right)\\
\end{array} \right. 
$$

### 3.Deep Equilibrium Nets

#### what is a deep neural network

A neural net is characterized by its parameters ρ, Given a parameter vector ρ and input vector x, denote the  neural net as $\mathcal{N}_{\rho}$​ , and some desired function with f.
$$

\left\{ \begin{array}{c}
	\mathcal{N}_{\rho}:\mathbb{R} ^{in}\rightarrow \mathbb{R} ^{out}:x\rightarrow \mathcal{N}_{\rho}\left( x \right)\\
	f:\mathbb{R} ^{in}\rightarrow \mathbb{R} ^{out}:x\rightarrow f\left( x \right)\\
	\left\| \mathcal{N}_{\rho}-f \right\| _{some\,\,norm}\,\,=\,\,0\\
\end{array} \right. 

$$

$$

\left| \begin{array}{c}
	input\,\,:=x\rightarrow \phi ^1\left( {W^1}_{\rho}x+b_{\rho}^{1} \right) \,\,=: hidden1\\
	\rightarrow hidden1\rightarrow \phi ^2\left( {W^2}_{\rho}hidden1+b_{\rho}^{2} \right) \,\,=: hidden2\\
	\rightarrow hidden3\rightarrow \phi ^1\left( {W^3}_{\rho}hidden2+b_{\rho}^{3} \right) =:  ........ output\\
\end{array} \right. 

$$

where $\phi$ is activate function.

#### How to find good parameters ρ?

##### the standard way

1. get “labelled data” $\mathcal{D}:\left\{ \left( x_1,y_1 \right) ,\left( x_2,y_2 \right) ,....,\left( x_{\left| D \right|},y_{\left| D \right|} \right) \right\}$​

2. Define a loss function, for example: $ l_{\rho}\,\,:=\,\,\frac{1}{\left| \mathcal{D} \right|}\sum_{\left( x_i,y_i \right) \in D}{\left( y_i-\mathcal{N}_{\rho}\left( x_i \right) \right)}^2$​

3. SGD:$\rho _{new}^{i}=\rho _{i}^{old}-\alpha ^{step}\frac{\partial l_{\rho ^{old}}}{\partial \rho _{i}^{old}}$

   ***Supervised Learning***

##### the economics way

An “economic” loss function : 
$$

l_{\rho}\,\,:=\,\,\frac{1}{\left| N_{path\,\,length} \right|}\sum_{x_i\,\,on\,\,sim. path}{\left( G\left( x_i,\mathcal{N}_{\rho}\left( x_i \right) \right) \right)}^2

$$
G is chosen such that the true equilibrium policy f(x) is defined by $G(x,f(x)) = 0$  G(.,.): implied error in the optimality conditions.

***Unsupervised Learning*** , just  simulate a path.



### 4.Using DQN solve baseline model

Define economic loss function:
$$

{e^i}_{REE}\,\,:=\,\,\frac{u\prime^{-1}\left( \beta E_{z_j}\left[ r\left( \hat{x}_{j,+} \right) u^{\prime}\left( \hat{c}^{i+1}\left( \hat{x}_{j,+} \right) \right) \right] +\hat{\lambda}^i\left( x_j \right) \right)}{\hat{c}^i\left( x_j \right)}

$$

$$

e_{KKT}^{i}\left( x_j \right) \,\,:=\,\,\hat{\lambda}^i\left( x_j \right) \left( \hat{a}^i\left( x_j \right) -a \right) 

$$

$$
\hat{x}_{j,+}=\left[ \begin{array}{c}
	z_+\\
	0\\
	\hat{a}^{\left[ 1:N-1 \right]}\left( x_j \right)\\
\end{array} \right]
$$

$$
\mathcal{N}_{\rho}:\left\{ 0,1,2,3 \right\} \times R^{60}\rightarrow R^{59\times 2}
$$

Input: $\left[ \begin{array}{c}
  z_t\\
  k_{t1}\\
  ...\\
  k_{t60}\\
\end{array} \right] $     output: $\left[ \begin{array}{c}
  a_{t1}\\
  ...\\
  a_{t59}\\
  \lambda _{t1}\\
  ...\\
  \lambda _{t59}\\
\end{array} \right] $

