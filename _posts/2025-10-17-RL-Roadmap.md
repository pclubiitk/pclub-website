---
layout: post
title: "Reinforcement Learning Roadmap 🦝"
date: 2025-11-01
author: Pclub
website: https://github.com/life-iitk
category: Roadmap
tags:
  - roadmap
  - rl
  - reinforcement learning
  - ml
categories:
  - roadmap
hidden: true
image:
  url: /images/ml-roadmap/rl-roadmap.jpg
---



# RL Guide
Welcome to the Reinforcement Learning (RL) Roadmap with Programming Club IITK! Get ready for a hands on, curiosity driven deep dive into how intelligent agents learn to make decisions. This roadmap is crafted to provide you with both a solid theoretical basis and ample practical experience, tailored for newcomers and enthusiasts eager to build real RL intuition.

## How to Approach This Roadmap:

* **Week 1:** Build your foundation by mastering the **exploration-exploitation dilemma** through various multi-armed bandit problems. This week is dedicated to understanding how an agent can learn the best actions in a static environment, a core concept that underpins all of RL.

* **Week 2:** Transition from stateless bandits to environments with sequential decision-making. You will be introduced to **Markov Decision Processes (MDPs)**, the mathematical framework for RL, and learn how to solve them using foundational methods like Dynamic Programming, Monte Carlo, and Temporal-Difference learning.

* **Week 3:** Next we will study **model-free methods**, where agents learn optimal policies directly from experience without needing a model of the environment. You'll implement and contrast cornerstone algorithms like Q-Learning, SARSA, and Policy Gradients.

* **Week 4:** Scale up your knowledge to tackle complex, high dimensional problems with **Deep Reinforcement Learning**. This week, you will combine neural networks with RL principles to build powerful agents using algorithms like Deep Q-Networks (DQN) and explore methods for continuous action spaces.

No matter which week you’re in, don’t hesitate to experiment! Document your learning, reflect frequently, and seek deeper understanding rather than shortcuts. Happy learning!

## Contents of the Guide

<details>
<summary> <strong>Week 1: The Foundations - Bandits and Basic RL</strong></summary>

* **Day 1:** [Introduction to Reinforcement Learning](#week1-day1)
* **Day 2:** [The Multi-Armed Bandit Problem - Learning from Choices](#week1-day2)
* **Day 3:** [Action-Value Methods & ε-Greedy Exploration](#week1-day3)
* **Day 4:** [Advanced Action Selection: UCB & Optimistic Initialization](#week1-day4)
* **Day 5:** [Gradient Bandit Algorithms](#week1-day5)
* **Day 6:** [Contextual Bandits](#week1-day6)

</details>

<details>
<summary> <strong>Week 2: From Bandits to Full reinforecement  learning</strong></summary>

* **Day 1:** [Markov Decision Processes (MDPs) & Bellman Equations](#week2-day1)
* **Day 2:** [Dynamic Programming](#week2-day2)
* **Day 3:** [Monte Carlo Methods: Learning from Episodes](#week2-day3)
* **Day 4:** [Temporal-Difference Learning](#week2-day4)
* **Day 5:** [Comparing DP, MC, and TD Methods](#week2-day5)
* **Day 6:** [Mini-Project 1: Solving a Gridworld](#week2-day6)

</details>

<details><summary> <strong>Week 3: Model Free Methods</strong></summary>
	
* Day 1: [Q-Learning Deep Dive](#week3-day1)
* Day 2: [SARSA vs Q-Learning](#week3-day2)
* Day 3: [Policy Gradient Methods](#week3-day3)
* Day 4: [Actor-Critic Basics](#week3-day4)
* Day 5: [Hyperparameter Tuning](#week3-day5)
* Day 6: [Linear Function Approximation](#week3-day6)
	
</details>

<details><summary> <strong>Week 4: Deep RL and Advanced Algorithms</strong></summary>

* Day 1: [Deep Q-Networks (DQN)](#week4-day1)
* Day 2: [DQN Improvements](#week4-day2)
* Day 3: [Policy Optimization](#week4-day3)
* Day 4: [Continuous Action Spaces](#week4-day4)
* Day 5: [Exploration Strategies](#week4-day5)
* Day 6: [Mini Project](#week4-day6)
	
</details>





<a id = "week1"></a>
# 🦝 Week 1: RL Basics and Bandits

<a id="week1-day1"></a>
### Day 1: Introduction to Reinforcement Learning
The idea that we learn by interacting with the world around us feels obvious. Think about an infant exploring its surroundings, reaching out, touching things, making sounds. No one is explicitly teaching every action, yet the child learns cause and effect. Some actions bring smiles and attention, while others bring discomfort or no response. Through these interactions, the child figures out what works to achieve certain goals.

This pattern continues throughout life. When we learn to ride a bike, cook a new recipe, or hold a conversation, we rely on trial, error, and feedback. We act, observe the outcome, and adjust. The better we get at understanding how our actions influence what happens next, the more effectively we can achieve our goals.

**Reinforcement Learning** is built on this same idea—learning through interaction and feedback. Instead of being told the right answer in advance, the learner discovers it by trying different things, seeing the results, and improving over time.

**Why formalize this process?**
To study this kind of learning systematically, we need a clear way to describe what’s happening. When we act and learn from feedback, there’s always something we’re interacting with, choices we make, and outcomes that guide our decisions. In reinforcement learning, these ideas are formalized into concepts like the environment, actions, rewards, and strategies. To understand them better, let’s look at an example.

### Example: A Delivery Robot
Imagine a small delivery robot that needs to deliver packages in a city. It doesn’t know the roads well, so it has to figure out how to reach destinations efficiently.
* **Environment:** The city streets, traffic signals, and weather—all the external factors the robot interacts with.
* **State:** The robot’s current situation—its location, battery level, and maybe the time of day.
* **Action:** What the robot decides to do next—go straight, turn left, take a shortcut, or stop to recharge.
* **Reward:** The feedback it gets—delivering faster earns a positive reward, running out of battery or getting delayed gives a negative one.
* **Policy:** The robot’s strategy—a mapping of situations to actions, like “if battery is low, go to charging station.”
* **Exploration vs. Exploitation:** Should it stick to known safe routes (exploitation) or try a new shortcut that might be faster—or riskier (exploration)?

**Working of the Robot:**
Every decision the robot makes shapes what it learns. Each choice, whether it takes a shortcut or sticks to a main road, brings consequences that reinforce or discourage that behavior. Through this constant cycle of action and feedback, the robot builds an understanding of which strategies work best.

It doesn't just aim for one quick delivery; its goal is to maximize success in the long run, that is, delivering packages faster and safer while avoiding failures like dead batteries or traffic delays. To achieve this, it must balance two competing needs: exploring new possibilities to discover better routes and exploiting what it already knows works well.

What’s remarkable is that no one hands the robot a map or the correct answers. It learns by doing, by interacting with its environment, observing the outcomes, and adjusting its strategy based on experience. This ability to discover effective behavior through trial and error, driven by rewards and penalties, is the essence of reinforcement learning.



Now that you understand the core idea of reinforcement learning, the best way to deepen your understanding is to see it in action and start with simple problems before tackling complex ones.

**Watch RL in Action**
Before diving into technical details, it helps to see reinforcement learning in the real world. RL isn’t just theory; it powers some of the most exciting breakthroughs, from bots that master games to robots navigating complex environments. To explore these applications, check out the [AI Warehouse YouTube channel](https://www.youtube.com/@aiwarehouse/videos).

**Bonus for cat lovers:** Want something fun? Watch this [short video](https://www.youtube.com/watch?v=KiHdKynXDtw) that explains RL through cats; it’s simple, intuitive, and adorable.

**Build a Strong Foundation**
Once you’ve seen RL in action, it’s time to understand the core ideas step by step. Start with this [StatQuest video](https://www.youtube.com/watch?v=Z-T0iJEXiwM); it’s one of the most beginner-friendly introductions, perfect for building intuition.

After that, reinforce your understanding by reading this [GeeksforGeeks article](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/?utm_source=chatgpt.com). It summarizes the key concepts in simple, clear language—great for a quick refresher after the video.

Finally, if you want a solid reference to guide you throughout your learning journey, dive into *Chapter 1* of [Reinforcement Learning: An Introduction by Sutton & Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf?utm_source=chatgpt.com). This book is considered the gold standard for RL and will provide the depth you need once you’ve grasped the basics.

***

<a id="week1-day2"></a>
### Day 2: The Multi-Armed Bandit Problem - Learning from Choices 

Imagine you walk into a casino with 10 slot machines lined up. Each one hides an unknown probability of giving you a payout. You have only 100 coins. Every pull of a lever costs one coin.

#### What's your objective?
Win as much as possible before you run out of coins.

But here’s the dilemma:
* You don’t know which machine pays best.
* Some may be bad, some average, maybe one is a jackpot.
* Every time you pull a lever, you learn something, but at the cost of not trying another machine.

This simple setup captures one of the core challenges of decision-making under uncertainty:
*Should you exploit what seems good or explore the unknown for something better?*

**Why this Matters?**
This is the **multi-armed bandit problem**, the simplest form of reinforcement learning.
There are no states or transitions, just choices with uncertain rewards. Yet, solving it well requires intelligent strategies to manage the **exploration–exploitation trade-off**.

**Resources:**
Here's a cool video:
[Multi-Armed Bandits: A Cartoon Introduction](https://www.youtube.com/watch?v=bkw6hWvh_3k&list=PLmofwKVg1Zm3i9r7A9XGg1KGikg-W9HeU)
(This video also introduces a few strategies; don’t worry, we’ll explore them in detail later.)

For a deeper read (just the intuition and examples, no need to worry about math yet):
[Sutton & Barto, Reinforcement Learning: An Introduction, Chapter 2.1 – The Multi-Armed Bandit Problem](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

***

<a id="week1-day3"></a>
### Day 3: Action-Value Methods & ε-Greedy Exploration
Imagine you’re playing a new game, and you’re not sure which moves score the most points. How do you figure it out? 
For that, we’ll explore methods for estimating action values and using them to choose actions intelligently.

**Key Idea:**
For each action $a$:
$$
Q(a) \approx \mathbb{E}[R \mid A = a]
$$

This means:
* $Q(a)$ → Our estimate of the value of action $a$.
* $R$ → The reward we receive after taking an action.
* $A$ → The action taken.

So $Q(a)$ is basically our guess of the average reward if we keep taking action $a$.

**If you’re new to expectation or probability, check this first:**
[Interactive Guide: Seeing Theory – Basic Probability](https://seeing-theory.brown.edu/basic-probability/index.html#section2)

#### Estimating Action Values
The simplest estimate is the **sample average**:
$$
Q_n(a) = \frac{\sum_{i=1}^{N_n(a)} R_i}{N_n(a)}
$$
Where:
* $Q_n(a)$: The estimated value of action $a$ after $n$ steps.
* $R_i$: The reward received on the $i$-th time action $a$ was chosen.
* $N_n(a)$: The number of times action $a$ has been selected up to step $n$. 

**Read:** [Sutton & Barto, Section 2.2](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

#### Acting on estimates: Greedy vs. ε-Greedy 
* **Greedy:** Always choose the option with the highest $Q(a)$.
* **ε-Greedy:**
  * With probability $1−ε$: pick the best action (exploit).
  * With probability $ε$: pick a random action (explore).
				
**Read:** [Sutton & Barto, Section 2.3](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)				

#### Incremental Update
Recomputing averages from scratch is costly. Use an incremental formula:  

$$
Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)
$$

For **nonstationary problems** (where reward probabilities change over time), use a constant step size:  

$$
Q_{n+1} = Q_n + \alpha(R_n - Q_n), \; 0 < \alpha \le 1
$$

**General RL pattern:** $$
\text{New Estimate} \leftarrow \text{Old Estimate} + \text{Step Size} \times (\text{Target} - \text{Old Estimate})
$$

**Read:** [Sutton & Barto, Section 2.4](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

**Implementation:**
[Notebook](https://colab.research.google.com/drive/1ajO-ihEcnCcZdNIN6zaJIKOoo2wCId0B?usp=sharing) with all the above implemented.

***

<a id="week1-day4"></a>
### Day 4: Advanced Action Selection

The exploration in ε-greedy is "blind." It wastes time trying obviously suboptimal arms and can't prioritize actions that are promising but uncertain. Today, we'll explore smarter ways to explore.

#### Optimistic Initial Values
By setting the initial action values $Q_1(a)$ to a number much higher than any possible reward, we can force a greedy agent to explore. Why? Because every real reward it receives will be "disappointing" compared to its optimistic initial belief, causing it to try every other action at least once before converging. This is a simple but powerful trick for encouraging initial exploration.

However, this strategy is not well-suited for non-stationary settings because the drive to explore is temporary.

**Read:** [Sutton & Barto, Section 2.5](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

#### Upper Confidence Bound (UCB)
UCB addresses the shortcomings of ε-greedy by exploring intelligently.It chooses actions based on both their estimated value and the uncertainty in that estimate.
$$
A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$
Here:
* $Q_t(a)$: Current estimate of the action's value.
* $N_t(a)$: How many times action $a$ has been chosen.
* $t$: The total number of steps so far.
* $c$: A parameter that controls the level of exploration.

The square root term is an "uncertainty bonus." It's large for actions that have been tried infrequently and shrinks as actions are tried more often. This makes UCB's exploration strategic, not random.

**Read:** [Sutton & Barto, Section 2.6](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

***

<a id="week1-day5"></a>
### Day 5: Gradient Bandit Algorithms

So far, we've focused on methods that estimate the *value* of actions. But what if we could learn a *preference* for each action directly? This shifts us from action-value methods to **policy-based methods**, where we learn a parameterized policy directly.

**Key Idea:**
Instead of learning action values, we learn a numerical preference $H_t(a)$ for each action.These preferences are converted into action probabilities using a softmax distribution:
$$
\pi_t(a) = \text{Pr}\{A_t=a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k}e^{H_t(b)}}
$$

**Updating Preferences with Stochastic Gradient Ascent:**
We update these preferences using the reward signal. The update rules are:
* **For the action taken, $A_t$**:
    $H_{t+1}(A_t) \doteq H_t(A_t) + \alpha(R_t - \overline{R}_t)(1 - \pi_t(A_t))$
* **For all other actions $a \neq A_t$**:
    $H_{t+1}(a) \doteq H_t(a) - \alpha(R_t - \overline{R}_t)\pi_t(a)$

Here, $\overline{R}_t$ is the **average of all rewards** up to time $t$, which serves as a **baseline**.
* If the reward $R_t$ is **higher than the baseline**, the preference for taking $A_t$ is increased.
* If the reward is **lower than the baseline**, the preference for taking $A_t$ is decreased.

**Read:** [Sutton & Barto, Section 2.7](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

***

<a id="week1-day6"></a>
### Day 6: Contextual Bandits

We've only considered **nonassociative tasks**, where we try to find the single best action overall. But what if the best action depends on the situation?This is the **contextual bandit** problem, a crucial step toward the full reinforcement learning problem.

**Key Idea:**
The goal in a contextual bandit task is to learn a **policy** that maps situations (contexts) to the actions that are best in those situations.

Imagine a bank of slot machines, but their colors change randomly. You learn that "if the machine is red, pull arm 2" and "if the machine is blue, pull arm 1." This is what we are doing here:
* **Search:** We still use trial-and-error to find the best actions.
* **Association:** We associate these actions with the specific contexts in which they are best.

This is different from the full RL problem because here, an action only affects the immediate reward, not the next context you see. We'll tackle that final piece next week!

**Read:** [Sutton & Barto, Section 2.8](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

***

<a id="week2"></a>
# 🦝 Week 2: From Bandits to Full Reinforcement Learning

This week, we move beyond single-step decisions and into the world of sequential prob=ems, where an action affects not only the immediate reward but also all future situations. We'll formalize this problem using Markov Decision Processes and explore the three fundamental approaches to solving it: Dynamic Programming, Monte Carlo, and Temporal-Difference Learning.

<a id="week2-day1"></a>
### Day 1: MDPs & Bellman Equations

We're now moving from single-step bandits to **sequential decision problems**, where an action has long-term consequences. To do this, we need a formal framework.

Today, we'll introduce the core concepts that define this framework: the **agent-environment interface**, the **goal** of maximizing a cumulative **return**, and the **Markov property** that governs the environment's dynamics. These ideas come together to form the **Markov Decision Process (MDP)**.

**For a full background on these foundational concepts, please read sections 3.1-3.4 in the Sutton & Barto text.**
* **Read:** [Sutton & Barto, Sections 3.1-3.4](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

---

#### 1. The Markov Property
The key assumption that makes sequential problems solvable is the **Markov property**.

* **The Idea:** "The future is independent of the past given the present."
* **What it means:** The current state $S_t$ holds all the necessary information to predict what happens next. A system that follows this rule is called a **Markov Process**.

**Read:** [Sutton & Barto, Section 3.5](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
**Watch for better understanding:** [Markov Property](https://www.youtube.com/watch?v=1meaW5GxUbY)

---

#### 2. Markov Decision Processes (MDPs)
An MDP is the mathematical framework for reinforcement learning. It's a Markov Process with the addition of **actions** (choices) and **rewards** (goals).

An MDP is formally defined by a tuple containing:
* $\mathcal{S}$: A set of states.
* $\mathcal{A}$: A set of actions.
* $p(s', r | s, a)$: The probability of transitioning to state $s'$ with reward $r$, from state $s$ and action $a$.
* $\gamma$: The discount factor, which determines the value of future rewards.

**Read:** [Sutton & Barto, Section 3.6](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

---

#### 3. Value Functions & The Bellman Equation
To solve an MDP, we need to figure out how "good" each state is. We do this by learning **value functions**, which estimate the expected future return.

* **State-Value Function ($v_\pi(s)$):** Expected return from state $s$ while following policy $\pi$.
* **Action-Value Function ($q_\pi(s, a)$):** Expected return from taking action $a$ in state $s$, then following policy $\pi$.

These value functions follow a recursive relationship known as the **Bellman Expectation Equation**:
$$
v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]
$$
In simple terms, the value of where you are is the expected immediate reward you get, plus the discounted value of where you're likely to end up next. This equation is the foundation for almost all RL algorithms.

**Resources:**
* **Watch:** [UCL/DeepMind - RL Lecture 2: Markov Decision Processes](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
* **Read:** [Sutton & Barto, Section 3.7](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
* **Read:** [GeeksforGeeks: MDPs](https://www.geeksforgeeks.org/machine-learning/markov-decision-process/)
* **Read:** [GeeksforGeeks: Bellman Equation](https://www.geeksforgeeks.org/machine-learning/bellman-equation/)

***

<a id="week2-day2"></a>
### Day 2: Dynamic Programming

Yesterday we formalized our problem as a Markov Decision Process (MDP). Today, we'll learn how to *solve* it. **Dynamic Programming (DP)** is a collection of algorithms that can compute the optimal policy, given a perfect model of the environment.

The core idea of DP is to turn the Bellman equations we learned about into update rules to progressively find the optimal value functions.

---

#### The Core Idea: Generalized Policy Iteration (GPI)

Almost all reinforcement learning algorithms, including DP, follow a general pattern called **Generalized Policy Iteration (GPI)**. It's a dance between two competing processes:

1.  **Policy Evaluation:** We take our current policy and calculate its value function. (How good is this strategy?)
2.  **Policy Improvement:** We take our current value function and update the policy to be greedy with respect to it. (Given what I know, what is a better strategy?)

These two processes are repeated, interacting until they converge to a single joint solution: the optimal policy and the optimal value function, where neither can be improved further .

---

#### The Main Algorithms

There are two classic DP algorithms that implement GPI:

* **Policy Iteration:** This algorithm performs the full GPI dance. It alternates between completing a full **Policy Evaluation** step and then performing a **Policy Improvement** step. This process is repeated until the policy is stable .

* **Value Iteration:** This is a more streamlined approach. Instead of waiting for the value function to fully converge, Value Iteration performs just *one* backup for each state before improving the policy. It combines the two steps of GPI by directly using the Bellman Optimality Equation as its update rule .

---

#### Why DP Matters & Its Limitations

DP provides the theoretical foundation for reinforcement learning and is guaranteed to find the optimal solution. However, it has two major drawbacks:
1.  It requires a **perfect model** of the environment's dynamics, which is often unavailable.
2. It involves sweeping through every single state, which is impractical for problems with very large state spaces (the "curse of dimensionality") .

#### Resources
* **Watch:** For a great visual and conceptual overview of Dynamic Programming, watch this [video by Google DeepMind](https://www.youtube.com/watch?v=Nd1-UUMVfz4).
* **Read:** For those following the book, read Chapter 3 for MDPs and Chapter 4 for DP from [here](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf).

***

<a id="week2-day3"></a>
### Day 3: Monte Carlo Methods

Yesterday, we saw that Dynamic Programming can find the optimal policy if we have a perfect model of the environment. But what if we don't have that model? This is where **model-free** reinforcement learning begins.

**Monte Carlo (MC) methods** are our first approach to learning without a model. The idea is simple: we learn the value of states by running many complete episodes and simply averaging the returns we get . To figure out the average time it takes to get home, just drive home many times and calculate the average. That's the essence of Monte Carlo.

---

#### Monte Carlo Prediction

The first step is learning to predict the value function, $v_\pi(s)$, for a given policy.

* **The Method:**
    1.  Play through a complete episode of experience following policy $\pi$.
    2.  For each state $S$ visited, calculate the **return** $G_t$ that followed that visit.
    3. Update the value estimate for that state, $V(S)$, by averaging it with the new return . 

* **First-Visit vs. Every-Visit:** **First-visit MC** averages returns from only the *first* time a state is visited in an episode, while **every-visit MC** averages returns from *every* visit . 

---

#### Monte Carlo Control

Our real goal is to find the optimal policy. To do this in a model-free world, we must learn **action-values**, $q_\pi(s, a)$, because they tell us how good an action is without needing a model to look ahead . We follow the same GPI pattern as before:

1.  **Policy Evaluation:** Generate an episode and use the returns to update our estimates of the action-value function, $Q(s, a)$.
2.  **Policy Improvement:** Improve the policy by making it more greedy with respect to our updated action-value estimates (e.g., using an $\epsilon$-greedy strategy).

This process, called **on-policy** Monte Carlo control, repeats, with the policy and action-value estimates gradually improving each other.

---

#### Key Characteristics of Monte Carlo

* **Model-Free:** It learns from raw experience.
* **Unbiased:** It learns from actual, complete returns.
* **High Variance:** The return for a full episode can vary significantly, which can make learning noisy.
* **Episodic Only:** Standard MC methods can only update at the end of an episode.
* **No Bootstrapping:** It does not update value estimates based on other estimates; it waits for the true outcome .

#### Resources
* **Watch:** [Monte Carlo Method for Learning State Value Functions in Python](https://www.youtube.com/watch?v=xGMjVX59aVE)
* **Read (Summary):** [Monte Carlo Policy Evaluation – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/monte-carlo-policy-evaluation/)
* **Read (Deep Dive):** [Sutton & Barto, Chapter 5 - Monte Carlo Methods](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

***

<a id="week2-day4"></a>
### Day 4: Temporal-Difference Learning

So far, we've seen two extremes: DP (needs a model) and MC (must wait for an episode to end). **Temporal-Difference (TD) Learning** combines the best of both.

TD learning is a **model-free** method, like MC, that learns from raw experience. But, like DP, it updates its estimates based on other learned estimates. This elegant idea of learning a "guess from a guess" is called **bootstrapping** and is central to modern reinforcement learning .

---

#### TD Prediction and the TD Error

Instead of waiting for the final return $G_t$, TD updates its estimate $V(S_t)$ toward a **TD Target**.

* **The TD Target:** This is an estimated return formed after one step: the immediate reward plus the *estimated value* of the next state: $R_{t+1} + \gamma V(S_{t+1})$ . 

* **The TD Error ($\delta_t$):** The learning signal in TD is the difference between the TD target and our current estimate.
    $$
    \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
    $$
    The **TD(0)** update rule uses this error to nudge the value of the current state:
    $$
    V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t
    $$

This allows the agent to learn from every single step.

---

#### TD Control: Sarsa vs. Q-Learning

When we use TD for control, we learn action-values ($Q(s,a)$). This leads to two of the most famous algorithms in RL.

* **Sarsa (On-Policy):** Sarsa learns the value of the policy the agent is *currently following*. Its name comes from the quintuple of experience it uses: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$. It's "on-policy" because its update target depends on the action $A_{t+1}$ that the policy *actually* chooses next . 

* **Q-Learning (Off-Policy):** Q-Learning is an **off-policy** algorithm. It learns the value of the *optimal* policy, regardless of what exploratory actions the agent takes. Its update target uses the *best possible action* from the next state, represented by the $\max$ operator .
    $$
    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
    $$

---

#### Key Characteristics of TD Learning

* **Model-Free:** Learns directly from experience.
* **Online and Incremental:** Learns from incomplete episodes, after every step.
* **Efficient:** By reducing variance, TD methods often learn faster than MC methods.
* **Bootstrapping:** It uses its own estimates to update its estimates, which is the source of its efficiency.

#### Resources
* **Watch (Concept):** [Temporal Difference learning by Mutual Information](https://www.youtube.com/watch?v=AJiG3ykOxmY)
* **Read (Sarsa):** [SARSA Algorithm: article by UpGrad](https://www.upgrad.com/tutorials/ai-ml/machine-learning-tutorial/sarsa-in-machine-learning/)
* **Read (Deep Dive):** [Sutton & Barto, Chapter 6 - Temporal-Difference Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

***

<a id="week2-day5"></a>
### Day 5: Comparing DP, MC, and TD Methods

Over the last few days, we've introduced the three foundational pillars for solving MDPs. Today, we'll consolidate our understanding by directly comparing them.

---

#### The Two Key Dimensions of Difference

The methods can be understood by looking at how they perform backups along two key dimensions: the **depth** and the **width** of the backup.

##### 1. Backup Depth: Bootstrapping vs. Sampling Full Returns

This dimension describes how far into the future an algorithm looks to get its update target.

* **Shallow Backups (Bootstrapping):** **DP** and **TD** methods **bootstrap**. They create an update target using the reward from just one step, plus the *estimated value* of the next state. They are learning a "guess from a guess."

* **Deep Backups (Full Returns):** **MC** methods do **not bootstrap**. They use the entire sequence of rewards from a completed episode—the actual, unbiased return—as their update target.

##### 2. Backup Width: Full vs. Sample Backups

This dimension describes whether an algorithm considers all possibilities or just one.

* **Full Backups (Model-Based):** **DP** methods use **full backups**. They average over every possible next state and reward according to a model of the environment ($p(s', r | s, a)$). This is why DP requires a model .

* **Sample Backups (Model-Free):** **MC** and **TD** methods use **sample backups**. They learn from a single, observed trajectory of experience and don't need a model .

---

#### The Unified View: A Summary Table

This table summarizes the core trade-offs:

| Method                  | **Requires a Model?** | **Bootstraps?** | **Key Advantage** | **Key Disadvantage** |
| :---------------------- | :-------------------: | :-------------: | :---------------------------------------------- | :------------------------------------- |
| **Dynamic Programming** |          Yes          |       Yes       | Guaranteed to find the optimal policy. | Requires a model; computationally expensive. |
| **Monte Carlo** |          No           |       No        | Unbiased; conceptually simple. | High variance; must wait for the episode to end. |
| **Temporal-Difference** |          No           |       Yes       | Low variance; learns online (step-by-step).      | Biased (learns from an estimate).       |

Essentially, TD learning is the sweet spot that combines the model-free sampling of Monte Carlo with the step-by-step bootstrapping of Dynamic Programming.

---

#### Practical Takeaways

* Use **Dynamic Programming** for planning when you have a perfect model and a manageable state space.
* Use **Monte Carlo** for episodic tasks where you can easily simulate many full episodes.
* Use **Temporal-Difference** as your default for most problems. It can learn online and is generally the most efficient.

#### Resources

* **Watch:** [UCL/DeepMind - RL Lecture 4 - Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=4) (This video provides a great comparison of MC and TD).
* **Read:** Review the summary sections of Chapters 4, 5, and 6 in the Sutton & Barto text.

***

<a id="week2-day6"></a>
### Day 6: Mini-Project: Escape the Gridworld!

Use this day to apply the concepts you've learned so far in a hands-on project.

**Project:**
Design and solve a simple gridworld environment. The goal is to train an agent to find the shortest path from a start state to a goal state, avoiding obstacles.

* **Task:** Design a simple grid (e.g., 5x5) with a start, a goal, and obstacle states.
* **Implementation:** Code a simple agent that can solve this environment using one of the model-free methods you've learned (e.g., Q-Learning or Sarsa). Watch how the agent's path improves as it learns.
* **Key Learning:** This project helps you grasp how value functions are updated in a structured environment and how a policy evolves from random to optimal. This is a crucial step before tackling more complex problems.










<a id="week3"></a>
# 🦝 Week 3: Model-Free Methods

<a id = "week3-day1"></a>
### Day 1: Q-Learning Deep Dive
Q-learning is a fundamental algorithm in model-free reinforcement learning (RL). Unlike model-based methods, it does not require knowledge of the environment’s dynamics. Instead, the agent learns purely through trial-and-error interactions with the environment.

The central idea is the use of Q-values, denoted as Q(s,a), which estimate the value of taking action 𝑎 in state 𝑠. These estimates are updated repeatedly as the agent explores, gradually improving its understanding of which actions lead to higher rewards.

Q-learning uses an off-policy temporal-difference (TD) update rule, meaning it can learn the optimal action-value function even if the agent’s behavior policy is not optimal. Over time, Q-values converge to their true values, enabling the agent to derive the optimal policy.

![q_learning_algorithm](https://hackmd.io/_uploads/S14uYsacle.jpg)

Resources:
* First half of this video: https://www.youtube.com/watch?v=VnpRp7ZglfA&ab_channel=Gonkee.
* [Learning Q-learning using a GridWorld Example](https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9).
* [Chapter 6 Section 6.5](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) from Sutton and Barto.


<!-- Q-Learning algorithm in detail
Learn about exploration vs. exploitation and epsilon-greedy strategies -->

<a id = "week3-day2"></a>
### Day 2: SARSA vs Q-Learning
Although SARSA and Q-learning are both temporal-difference (TD) learning methods used in reinforcement learning, they differ in how they update the action-value function. Both algorithms aim to estimate the optimal policy, but the way they handle the next action distinguishes them:

**Q-learning (off-policy):**

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$

Here, the update rule uses the maximum future action-value regardless of the agent’s current policy. This makes Q-learning an off-policy method, since it learns the optimal greedy policy while potentially following a different exploratory policy during training.

**SARSA (on-policy):**

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right] $$


In contrast, SARSA updates its value based on the actual action chosen by the current policy in the next state. This makes it an on-policy method, where learning is tied to the agent’s own behavior, including exploration.

To get a deeper understanding of their behavior, similarities, differences, and applications, go through the following resources:

* [Differences between Q-learning and Sarsa](https://www.geeksforgeeks.org/artificial-intelligence/differences-between-q-learning-and-sarsa/).
*  [Q Learning vs SARSA article](https://medium.com/@priya61197/q-learning-vs-sarsa-b9e433dec930).
*  Watch this [section](https://www.youtube.com/watch?v=VnpRp7ZglfA&t=2182s) of the video for knowing the mathematical differences explicitely.
<!-- Compare on-policy (SARSA) and off-policy (Q-Learning) methods
Understand their similarities and differences     -->

<a id = "week3-day3"></a>
### Day 3: Policy Gradient Methods
Till now we've studied how a RL model chooses how good an action is and chooses the best. Now, instead of estimating these values, it'll directly learn how to act.

Policy gradient methods directly optimize the policy instead of estimating value functionss. The REINFORCE algorithm is the simplest such method. It stars with a random policy, then runs several episodes using this policy and records the respective states, actions and rewards. Then for each action taken, it updates the policy function using an update rule. This process is repeated to reach the optimal policy.

Go through the following resources to learn about policy gradient methods:
* Watch this [section](https://www.youtube.com/watch?v=VnpRp7ZglfA&t=3525s) of the video.
* Watch this [video](https://www.youtube.com/watch?v=e20EY4tFC_Q&ab_channel=MutualInformation) for an in-depth study.
* Implement the REINFORCE algorithm following this [article](https://medium.com/@sofeikov/reinforce-algorithm-reinforcement-learning-from-scratch-in-pytorch-41fcccafa107).
<!-- Introduction to policy gradients
Study the REINFORCE algorithm -->

<a id = "week3-day4"></a>
### Day 4: Actor-Critic Basics
Actor–critic methods are TD methods that have a separate memory structure to explicitly represent the policy independent of the value function. The policy structure is known as the actor, because it is used to select actions, and the estimated value function is known as the critic, because it criticizes the actions
made by the actor.

Resources: 
* [Actor-Critic Algorithm in Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)
* [Chapter 11](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) from Sutton and Barto.
<!-- Learn the basics of actor-critic methods
Understand how policy and value functions are combined -->

<a id = "week3-day5"></a>
### Day 5: Hyperparameter Tuning
Hyperparameters are external parameters set before the training begins. Learning rate α, discount factor γ, exploration rate 𝜖, number of episodes, etc, are some of the common hyperparameters in RL models.

Often, RL models face convergence issues, that is it fails to learn. One of the major reasons for this failed convergence is often a bad set of hyperparameters as mentioned above. Check out this [video](https://www.youtube.com/watch?v=haINijwCDkk&ab_channel=NathanLambert) summarizing the importance of hyperparameter tuning in RL paper. 

There are three major optimization methods used in tuning RL hyperparameters. Those are: 
* **Grid Search**: Selecting the best suited hyperparameters from a pre-defined set.
* **Random Search**: As the name suggests, selecting the best set of hyperparameters by iterating through random such sets.
* **Bayesian Optimization**: This is the *smart* way of chosing hyperparameters. Here we start with a random set and then proceed to guide the hyperparameters to the opimal values by each iteration.

Watch this [video](https://www.youtube.com/watch?v=nMubTWJGgiU&ab_channel=DigiKey) to learn how to implement hyperparameter optimization in RL models using Meta's Ax.

<a id = "week3-day6"></a>
### Day 6: Linear Function Approximation
Before diving into Deep Q-Networks (DQNs) in the upcoming Week-4, it’s important to understand Linear Function Approximation (LFA). This is one of the simplest yet powerful approaches to approximate value functions in Reinforcement Learning.
In RL, the state space can often be enormous or continuous. Take chess for example, where the number of possible states is estimated to be around 10<sup>47</sup>. Storing a value for each state is clearly impossible. This explosion of possibilities is termed as the **Curse of Dimensionality** – as the number of state variables grows, the total number of states grows exponentially.

#### Feature Vector Representation of States
To make this possible, we represent each state using a **feature vector**. A state $s$ is mapped into a vector:  

$$
\phi(s) = [\phi_1(s), \phi_2(s), ..., \phi_n(s)]^T
$$

where each component $\phi_i(s)$ captures some aspect of the state. This feature-based representation reduces the complexity of the state space and makes learning more tractable.  

Once states are represented by feature vectors, we can approximate the value function as a linear combination of these features. Mathematically, this is written as:  

$$
\hat{v}(s; \theta) = \theta^T \phi(s)
$$

Here, $\theta$ is the weight vector (the parameters we learn), $\phi(s)$ is the feature vector of the state, and $\hat{v}(s; \theta)$ is the estimated value of state $s$. The aim is to learn weights $\theta$ that best approximate the true values of states.  

#### Stochastic Gradient Descent (SGD) Update
Using the SGD update rule, we minimize the mean squared error between the predicted value and the target:  
$$
J(\theta) = \frac{1}{2} \big( v(s) - \hat{v}(s; \theta) \big)^2
$$

and update the weights as:
$$
\theta \leftarrow \theta + \alpha \big( v(s) - \hat{v}(s; \theta) \big) \phi(s)
$$
where $\alpha$ is the learning rate and $v(s)$ is the target (either the actual return or a bootstrapped value estimate).
For those unfamiliar with the SGD update rule, you can refer to the SGD update rule section in the [ML Roadmap](https://pclub.in/roadmap/2024/06/06/ml-roadmap/).

What makes linear approximation powerful is how easily it integrates with reinforcement learning algorithms. In TD(0), for instance, the target is a bootstrapped estimate $R + \gamma \hat{v}(s'; \theta)$. In SARSA and Q-learning, we can extend the idea to action-value functions, approximating them linearly as $\hat{q}(s,a;\theta) = \theta^T \phi(s,a)$

You can refer to the following resources for a deeper dive into LFA:
- [Understanding Q-learning with LFA](https://danieltakeshi.github.io/2016/10/31/going-deeper-into-reinforcement-learning-understanding-q-learning-and-linear-function-approximation/)
- [LFA with application](https://gibberblot.github.io/rl-notes/single-agent/function-approximation.html)
- [Repo for implementing SARSA with LFA](https://github.com/LucasAlegre/linear-rl)


<a id = "week4"></a>
# 🦝 Week 4: Deep RL & Advanced Algorithms

We'll be integrating Q-learning with deep neural networks this week. So, for obvious reasons, you'll need to know about neural networks and some other basics of machine learning. So, for those who are new to these terms, here are a few resources that will be helpful to get started:

- [Neural Networks and Backpropogation(Week-6: Days 1,2)](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week6-Day1).
- [Optimization Algorithms(Week-7: Days 1,2)](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week7-Day1,2)

<a id = "week4-day1"></a>
### Day 1: Deep Q-Networks (DQN)
Deep Q-Networks revolutionized reinforcement learning by combining Q-learning with deep neural networks, enabling agents to handle high-dimensional state spaces like raw pixel inputs. DQN uses a convolutional neural network that takes raw state observations as input and outputs Q-values for each possible action.

![1*emv9eFMbGODD4gnITjfwcQ](https://hackmd.io/_uploads/r1ahnJ2Lle.png)


To brush up your neural network concepts, refer to [Week6](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week6) of the ML roadmap.

**Resources:**
* [Deep Q-Learning/Deep Q-Network (DQN) Explained](https://www.youtube.com/watch?v=EUrWGTCGzlA) - This tutorial contains step by step explanation, code walkthru, and demo of how Deep Q-Learning (DQL) works
* [Hugging Face Deep RL Course: Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm) - Comprehensive course module explaining DQN architecture and training process
* [deep-Q-networks](https://github.com/cyoon1729/deep-Q-networks) - A repo which has all DQN algos implemented

**Key Learning Points:**
* Neural network approximation of Q-values
* Experience replay for stable training
* Target networks to reduce correlation
* Applications in Atari games and robotics


<a id = "week4-day2"></a>
### Day 2: DQN Improvements
Deep Q-Networks (DQN) represented a breakthrough in reinforcement learning, but several fundamental problems emerged that led to the development of improved variants.

**Key Problems:**
* **Overestimation** 
  * DQN gets too optimistic about how good actions are
  * It uses the same "brain" to pick actions and judge how good they are
  * This creates a feedback loop where bad estimates get worse over time
  * The agent picks actions that look amazing but actually aren't

  **Solution: Double DQN**
Uses two separate "brains" - one picks the action, the other judges it. This prevents the overoptimistic feedback loop.
* **Sample Inefficiency**
  * DQN learns from random past experiences, treating boring and exciting ones equally
  * Important "moments" get ignored while routine stuff gets repeated
  * Like studying for an exam by randomly picking pages instead of focusing on hard topics
  * Takes way too long to learn because it wastes time on useless examples
  
  **Solution: Prioritized Experience Replay**
  Smart studying! Focuses more on important experiences and less on  routine ones.
  
* **Architecture Problems**
  * DQN tries to learn two different things at once with one network
  * It must figure out "how good is this situation?" AND "which action is best?"
  * This makes learning slower and more confusing
  
  **Solution: Dueling DQN**
  Splits the network into two parts - one learns "how good is this situation?" and another learns "which action is relatively better?" Then combines them smartly.
  
**Resources:**
* [Double DQN (DDQN) Explained & Implemented](https://www.youtube.com/watch?v=FKOQTdcKkN4) - Enhance the DQN code with Double DQN (DDQN).
* [Dueling DQN (Dueling Architecture) Explained & Implemented](https://www.youtube.com/watch?v=3ILECq5qxSk&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=1)- Enhance the DQN module with Dueling DQN (aka Dueling Architecture).
* [Prioritized experience replay](https://www.youtube.com/watch?v=t_KWBRO1ZRU) - The Prioritized experience paper explained
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298v1) - Comprehensive study combining multiple DQN improvements including Double DQN, Dueling DQN, and Prioritized Experience Replay

The Implementations of DDQN, Dueling DQN and PER can be found [here](https://github.com/cyoon1729/deep-Q-networks).

<a id = "week4-day3"></a>
### Day 3: Policy Optimization
While DQN improvements—Double DQN, Dueling DQN, and Prioritized Experience Replay significantly enhanced the stability and efficiency of value based learning, they still operate within fundamental constraints that limit their real-world applicability. These methods remain bound to discrete action spaces and continue learning policies indirectly through value function approximation.

**Key Limitations of Value-Based Methods:**
* **Discrete action spaces only** - cannot handle continuous control
* **Indirect policy learning** - must derive policy from Q-values
* **Memory intensive** - storing Q-values for all state-action pairs
* **Limited exploration** - epsilon-greedy can be inefficient

**Policy-Based vs Value-Based Approaches**
* **Value-Based (DQN) Approach:**
*State → Neural Network → Q-values for each action → Choose max Q-value action
Example: Q(s, left) = 0.8, Q(s, right) = 0.6 → Choose left*
* **Policy-Based Approach:**
*State → Neural Network → Action probabilities or direct actions
Example: π(left|s) = 0.7, π(right|s) = 0.3 → Sample from distribution*

**On-Policy vs Off-Policy Methods**
* **On-Policy Methods (TRPO, PPO):**
  * Learn from data generated by the current policy
  * Update the same policy that's collecting data
  * More stable but sample inefficient
  * Better for continuous control tasks
* **Off-Policy Methods (DDPG, TD3):**
  * Learn from data generated by any policy (including old versions)
  * Can reuse old experience data
  * More sample efficient but potentially unstable
  * Good for environments where data collection is expensive


**Trust Region Policy Optimization (TRPO)**
TRPO prevents significant performance drops by keeping policy updates within a trust region using KL-divergence constraints

**Mathematical Foundation:**
* **Objective:** Maximize expected return while constraining policy change
* **Constraint:** KL-divergence between old and new policies ≤ δ
* **Update Rule:** Uses conjugate gradient and line search for optimization

**Resources:**
* [TRPO Paper Explained](https://www.youtube.com/watch?v=xUOQMvR0T6Q) - Mathematical derivation walkthrough
* [Trust Region Methods in RL](https://spinningup.openai.com/en/latest/algorithms/trpo.html) - OpenAI Spinning Up guide

<!-- ![Model-based-TRPO-framework.ppm](https://hackmd.io/_uploads/Sk6Fix38eg.png) -->

**Proximal Policy Optimization (PPO)**
PPO is an on-policy, policy gradient method that uses a clipped surrogate objective function to improve training stability by limiting policy changes at each step.

**Key Innovation:** Clipped probability ratio prevents large policy updates

*Clipped Objective = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
where r(θ) = π_new(a|s) / π_old(a|s)*

**Advantages over TRPO:**
* **Simpler implementation** - no conjugate gradient required
* **Faster training** - fewer computational steps per update
* **Better sample efficiency** - more stable learning

**Resources:**
* [PPO Explained Simply](https://www.youtube.com/watch?v=5VHLd9eCZ-w) - Whiteboard walkthrough with intuition
* [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - OpenAI Spinning Up guide
* [PPO vs TRPO Comparison](https://arxiv.org/pdf/1707.06347) - Original PPO paper

<!-- ![Proximal-Policy-Optimization-PPO-architecture.ppm](https://hackmd.io/_uploads/BJlfjxhUll.png) -->

** Practical Implementation Tips**
1. **Start with PPO** - easier to implement and debug
2. **Hyperparameter sensitivity** - both methods require careful tuning
3. **Environment normalization** - crucial for stable training
4. **Advantage estimation** - use GAE (Generalized Advantage Estimation)

Implementations of both PPO and TRPO can be found [here](https://github.com/Khrylx/PyTorch-RL)

<a id = "week4-day4"></a>
### Day 4: Continuous Action Spaces
Policies like PPO and TRPO work when you have a limited number of clear choices (like "turn left," "turn right," "go straight").But in real life, many problems need smooth, precise control.

DDPG and TD3 specifically tackle environments where actions are continuous vectors (like robot joint angles, steering wheel positions, or throttle controls) rather than discrete choices.

These algorithms can learn to output precise numerical values. Instead of choosing between 4 discrete actions, these algorithms can choose any number between -1 and +1 (or any range you need).

**Deep Deterministic Policy Gradient (DDPG)**
DDPG combines ideas from DQN and policy gradient methods using an actor-critic architecture. The actor learns a deterministic policy mapping states to actions, while the critic evaluates actions by estimating Q-values

**Twin Delayed DDPG (TD3)**
TD3 addresses DDPG's brittleness with three critical improvements:
* Clipped Double-Q Learning: Uses two Q-functions and takes the minimum to reduce overestimation
* Delayed Policy Updates: Updates policy less frequently than Q-functions
* Target Policy Smoothing: Adds noise to target actions for robustness

**DDPG vs. TD3**

| Feature | DDPG | TD3  |
| ------- | ---- | ---- |
| **Value Estimation**  | Single critic; prone to overestimation     |Two critics; uses minimum value to reduce bias      |
| **Update Schedule**        | Actor and critic update together    |   Actor updates are delayed and less frequent   |
| **Stability**        |  Brittle and sensitive to hyperparameters    |  Significantly more stable and robust   |



**Resources:**
* [Deep Deterministic Policy Gradient (DDPG) in reinforcement learning explained with codes](https://www.youtube.com/watch?v=n7ZtPS_a4TI)
* [TD3: Overcoming Overestimation in Deep Reinforcement Learning](https://medium.com/@kdk199604/td3-overcoming-overestimation-in-deep-reinforcement-learning-c52d1cc9d69a) - Medium article
* [OpenAI Spinning Up: Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html) - Detailed breakdown of TD3's three key improvements over DDPG
* [Twin-Delayed DDPG (TD3) with Bipedal Walker](https://github.com/ChienTeLee/td3_bipedal_walker) -  Practical implementation of TD3

<a id = "week4-day5"></a>
### Day 5: Exploration Strategies
So far, we've built agents that can learn optimal actions. But how does an agent discover these actions in the first place? This is the core of the [exploration vs. exploitation dilemma](https://www.youtube.com/watch?v=Fhe9zKmTyBY&t=7s). Should the agent exploit its current knowledge to get high rewards, or should it explore new, untried actions that might lead to even better rewards in the long run?

Agents like DDPG and TD3 are great at executing a known strategy, but without a good exploration plan, they can get stuck in a rut, repeating the first successful actions they find without ever discovering superior [alternatives](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/). This section covers the fundamental strategies agents use to explore their environment effectively.


**Classic Exploration Strategies (for Discrete Actions):**
These methods are foundational and work best in environments with a limited set of distinct actions (like "left," "right," "up," "down").
* **ε-Greedy (Epsilon-Greedy)**
  With probability ε, the agent chooses a random action (explores). With probability 1-ε, it chooses the best-known action (exploits).
  **Reading:**
  * [Optimizing Decision-Making in AI: Fine-Tuning the Epsilon-Greedy Algorithm for Enhanced Performance](https://medium.com/operations-research-bit/optimizing-decision-making-in-ai-fine-tuning-the-epsilon-greedy-algorithm-for-enhanced-performance-ea61e86d6f1d)
  * [ Epsilon greedy algorithm](https://www.youtube.com/watch?v=EjYEsbg95x0)
* **Softmax (Boltzmann) Exploration**
  Actions are chosen based on a probability distribution derived from their Q-values. Higher-value actions have a higher probability of being selected, but no action has a zero probability. A "temperature" parameter controls the randomness: high temperatures lead to more random choices (more exploration), while low temperatures make the agent greedier.  
  **Reading:**
  * [Boltzmann Exploration Done Right](https://proceedings.neurips.cc/paper_files/paper/2017/file/b299ad862b6f12cb57679f0538eca514-Paper.pdf)
  * [What is softmax action selection in RL?](https://zilliz.com/ai-faq/what-is-softmax-action-selection-in-rl)
  * [An Alternative Softmax Operator for Reinforcement Learning](https://www.youtube.com/watch?v=wVBQGKMX974)
* **Upper Confidence Bound (UCB)**
  This strategy chooses actions based on both their estimated reward and the uncertainty of that estimate. It adds a "bonus" to actions that haven't been tried often, making them more attractive. The core formula balances the average reward with an exploration term that shrinks as an action is selected more frequently.
  **Reading:**
  * [Know all About Upper Confidence Bound Algorithm in Reinforcement Learning](https://www.turing.com/kb/guide-on-upper-confidence-bound-algorithm-in-reinforced-learning)
  * [Upper Confidence Bound UCB Algorithm](https://www.youtube.com/watch?v=s6UHInwoqb0)
 
 **Performance Insights:**
Research shows that softmax consistently performs best across different maze environments, while ε-greedy often underperforms compared to other strategies. UCB and pursuit strategies show competitive performance with proper parameter tuning.
 
 **Exploration in Continuous Action Spaces:**
 For algorithms like DDPG and TD3, which operate in continuous action spaces (e.g., controlling a steering angle), exploration is handled differently. Since you can't just pick a "random" action from an infinite set, exploration is typically achieved by adding noise to the policy's output.
 * **Action Noise:** The most common approach is to add random noise directly to the action selected by the actor. This noise is often drawn from a simple Gaussian distribution or a more complex process like the Ornstein-Uhlenbeck process, which generates temporally correlated noise suitable for physical control tasks.
  **Reading:**
    * [Action Noise in Off-Policy Deep Reinforcement Learning: Impact on Exploration and Performance](https://arxiv.org/abs/2206.03787)
 * **Parameter Noise:** An alternative is to add noise directly to the weights of the policy network. This can lead to more consistent and structured exploration than simply perturbing the final action.
  **Reading:**
    * [Better exploration with parameter noise](https://openai.com/index/better-exploration-with-parameter-noise/)
	* [Parameter Space Noise for Exploration](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/nips/param-noise-final%20-%20Matthias%20Plappert.pdf)
	
**Advanced Exploration**
A more advanced concept is to create "curious" agents that are intrinsically motivated to explore. Instead of relying only on external rewards from the environment, the agent receives an intrinsic reward for visiting new or unpredictable states.
* **Entropy-Based Exploration:** Adds a bonus to the reward function based on the entropy of the policy. This encourages the agent to maintain a more random (higher-entropy) policy, preventing it from converging too quickly to a suboptimal solution.
* **Curiosity-Driven Exploration:** The agent builds a model of the environment's dynamics. It is then rewarded for taking actions that lead to states its model cannot predict well. In other words, the agent is rewarded for being "surprised".
  **Reading:**
    * [Reasoning with Exploration: An Entropy Perspective](https://huggingface.co/papers/2506.14758)
    * [ELEMENT: Episodic and Lifelong Exploration via
Maximum Entropy](https://arxiv.org/pdf/2412.03800)
    * [Exploration Strategies in Deep Reinforcement Learning](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) - Article explaining all the exploration strategies.
	

<!-- **Key Exploration Methods:**
* **Entropy-based:** Uses policy entropy to encourage diverse action selection
* **Upper Confidence Bound (UCB):** Balances mean reward estimates with uncertainty measures
* **Softmax:** Uses probabilistic action selection based on Q-value estimates
* **ε-greedy:** Simple approach alternating between greedy and random actions -->

 **Further Reading:**
* [Comparing Exploration Strategies for Q-learning in
Random Stochastic Mazes](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/Exploration_QLearning.pdf)
* [Entropy-based tuning approach for Q-learning in an unstructured environment](https://www.sciencedirect.com/science/article/pii/S0921889025000107)




<a id = "week4-day6"></a>
### Day 6: Mini Project
Here are advanced project suggestions for Week 3:

* **DQN Atari Implementation**
	* Implement vanilla DQN on a simple Atari game
	* Add Double DQN improvements
	* Compare performance metrics

* **Continuous Control with DDPG**
	* Use OpenAI Gym's continuous control environments
	* Implement basic DDPG
	* Upgrade to TD3 and compare results

* **Exploration Strategy Comparison**
	* Implement different exploration strategies in a gridworld
	* Compare learning curves and final performance
	* Analyze exploration-exploitation tradeoffs

**Implementation Resources:**
* [TensorFlow Agents](https://www.tensorflow.org/agents/tutorials/0_intro_rl) - Provides ready-to-use implementations of all covered algorithms
* [OpenAI Gym](https://gymnasium.farama.org) - Standard environment suite for testing RL algorithms
* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) - High-quality implementations of modern RL algorithms

**Contributors**
- Harsh Malgatte \| +91 7385840159
- Shreyansh Rastogi \| +91 7459094300
