---
layout: post
title: "QubitFlow: Quantum Computing and Machine Learning"
date: 2025-05-26 19:30:00 +0530
author: Advaith GS, Himanshu Sharma 
category: Project
tags:
- summer25
- project
categories:
- project
hidden: true
summary:
- Introduction to Quantum Computing and building some Quantum Machine Learning Algorithms like QNNs, Quantum Regression and Classification, QSVCs, QCNNs.
image:
  url: "https://g.foolcdn.com/editorial/images/818851/rigetti-computing-versus-d-wave-quantum-stocks-2025.jpg"
---

# About the Project
Introduction to Quantum Computing and building some Quantum Machine Learning Algorithms like QNNs, Quantum Regression and Classification, QSVCs, QCNNs.

Interesting Channel: [MLO](https://www.youtube.com/@physicsowen/playlists)
Has playlists on Quantum Physics, IBM Qiskit, QAOA, VQEs, Some Paper Discussions, QCOpt, QOSF, QML Theory, Pennylane, Cirq, Tensorflow Quantum, RL,Intro to QC and some Conference Videos

# Resources
## Week 0 | Into the Quantum World
1. Linear Algebra Refresher:  Videos 1-3, 9-11, 13-15 of 3B1B [Lin Alg Playlist](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
2. [The Language of Quantum mechanics](https://www.youtube.com/watch?v=payp7simhBM&ab_channel=ParthG)
3. [This video shall provide an intuitive introduction to QM](https://www.youtube.com/watch?v=MzRCDLre1b4&ab_channel=3Blue1Brown)
4. [What are wave functions](https://www.youtube.com/watch?v=w9Kyz5y_TPw&ab_channel=ParthG)
5. [What does a wave function represent](https://www.youtube.com/watch?v=i7Du1J6t8qo&list=PLOlz9q28K2e4Yn2ZqbYI__dYqw5nQ9DST&index=12&ab_channel=ParthG)

## Week 1 | Fundamentals of Quantum Computing
1. Lectures  1 to 3 of [this playlist](https://www.youtube.com/watch?v=NZD9APb7ZtY&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=2&ab_channel=Qiskit)
2. [Reference notes for the lectures](https://raw.githubusercontent.com/qiskit-community/intro-to-quantum-computing-and-quantum-hardware/refs/heads/master/lectures/introqcqh-lecture-notes-1.pdf)

## Relevant Sections of Nielsen Chuang
Link to the book can be found [here](https://github.com/Himanshu2909/EE798V-2025-Spring/blob/main/Resources/Nielsen-Chuang.pdf)
1. Start off with Stern-Gerlach (1.5.1). Provides a good understanding of some non-intuitive parts of QM
2. Cover the results of 2.1-Linear Algebra. Proofs not important for the project but having an understanding of the results will really help
3. Post that, you can read the first 3 sub sections of 2.2. Anything beyond that would be a it too complex for now.
4. Then, cover 4.1 and 4.2.

## Introduction to Python
1. [Intro to Python](https://www.youtube.com/watch?v=K5KVEU3aaeQ&ab_channel=ProgrammingwithMosh)
2. [Exception Handling](docs.python.org/3/tutorial/errors.html#exceptions)
3. [Anaconda](docs.python.org/3/tutorial/errors.html#exceptions)
4. [Path and environment variables for Python and Anaconda in Windows](www.youtube.com/watch?v=HyxR0QTTJJs&ab_channel=Start-TechAcademy)
5. [Interactive Python Notebooks](https://www.youtube.com/watch?v=5pf0_bpNbkw&t=208s&ab_channel=RobMulla) 
6. [Venv](https://www.youtube.com/watch?v=APOPm01BVrk&ab_channel=CoreySchafer)
7. [Managing Packages with venv](docs.python.org/3/tutorial/venv.html)
8. [Python virtualenv](https://virtualenv.pypa.io/en/latest/)
9. [PyEnv](https://github.com/pyenv/pyenv) for Python Version Management
10. [Git](https://learngitbranching.js.org/?locale=en_US)

## Week 2 | Hands on Qiskit

### Circuit Solving
Solving circuits involves bring given a logical quantum circuit diagram with initial states and a set of gates applied to those states in a particulat order and you deciphering the state of the system after application of each state. For mastering circuit solving, you need to understand how time evolution of states occurs, how to convert states and gates into matrix representations and apply gates to various states. [This](https://www.youtube.com/watch?v=tsbCSkvHhMo&ab_channel=freeCodeCamp.org) video along with resources from Week 1 equip you with all tools required for Solving Quantum Circuits. For a more comprehensive tutorial, go through this [document](https://www.cs.cmu.edu/~odonnell/quantum15/lecture01.pdf) and this [blog](https://www.thp.uni-koeln.de/trebst/PracticalCourse/quantum_circuits.html).

### Migration to Qiskit 2.O
Migration Guides:
1. [https://quantum.cloud.ibm.com/docs/en/migration-guides/classic-iqp-to-cloud-iqp](https://quantum.cloud.ibm.com/docs/en/migration-guides/classic-iqp-to-cloud-iqp)
2. [https://quantum.cloud.ibm.com/docs/en/migration-guides/qiskit-2.0](https://quantum.cloud.ibm.com/docs/en/migration-guides/qiskit-2.0)

For accessing IBM Backends:
1. Sign into [https://cloud.ibm.com/](https://cloud.ibm.com/)
2. Sign into [https://quantum.cloud.ibm.com](https://quantum.cloud.ibm.com)
3. Use the IBM backends the way they are described in the migration guides

### Introduction to Qiskit
1. [Qiskit Hello World Notebook](https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/Coding_With_Qiskit/ep3_Hello_World.ipynb)
2. [Quantum Gates in Qiskit Notebook](https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/Coding_With_Qiskit/ep4_Gates.ipynb)
3. [Qiskit Circuits](https://docs.quantum.ibm.com/api/qiskit/circuit)
4. [Quantum Teleportation](https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/Coding_With_Qiskit/ep5_Quantum_Teleportation.ipynb)
5. [Implementing a basic VQE using Qiskit](https://github.com/qiskit-community/ibm-quantum-challenge-2024/blob/main/content/lab_1/lab-1.ipynb)

### Classical and Quantum Complexity Classes
Additional Resources:
1. [Classical Complexity](https://groups.uni-paderborn.de/fg-qi/courses/UPB_QCOMPLEXITY/2020/notes/Lecture%201%20-%20Complexity%20theory%20review.pdf)
2. [Quantum Complexity](https://cs.uwaterloo.ca/~watrous/Papers/QuantumComputationalComplexity.pdf)
3. [Classical v/s Quantum Complexity](https://arxiv.org/pdf/2312.14075v3)

### Assignment 1
Implement a Quantum Teleportation Circuit in Python using Qiskit.


## Week 3 | Simulations and Transpilers

### Transpilers
1. [Qiskit Transpilers](https://docs.quantum.ibm.com/api/qiskit/transpiler): Skim through the subpages,but read the overview and info about stages, pass managers in detail
2. [Qiskit Transpilers II](https://qiskit.qotlabs.org/guides/transpile): this has a pretty similar wording but a few more illustrative examples that might be helpful
3. [Video 1](https://www.youtube.com/watch?v=720cN0WUo4M&pp=ygUScWlza2l0IHRyYW5zcGlsZXJz): An intro to the intuition behind AI Transpilation, its training and tuning
4. [Video 2](https://www.youtube.com/watch?v=MvX5OUK-tbE): A pretty good intro on Transpilers, motiovation, and their structure in Qiskit
5. [Circuit Converters](https://docs.quantum.ibm.com/api/qiskit/converters)
6. [Transpiler Targets](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.Target)
7. [Transpiler Stages](https://docs.quantum.ibm.com/transpile/transpiler-stages)

### AI-powered Transpilers
1. [Optimize quantum circuits with AI-powered transpiler passes](https://www.ibm.com/quantum/blog/ai-transpiler-passes)
2. [Seminar: Transpile your circuits with AI](https://www.youtube.com/watch?v=-k3P9E0UTY8)

### Quantum Simulations

#### Backends
1. [Qiskit Backends](https://medium.com/qiskit/qiskit-backends-what-they-are-and-how-to-work-with-them-fb66b3bd0463)
2. [IBM Backends Docs](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/ibm-backend)
3. [Transpile against custom backends](https://quantum.cloud.ibm.com/docs/en/guides/custom-backend)

#### Simulators: Documentation and Theory
1. [BasicProvider and BasicSimulator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/providers_basic_provider)
2. [Fake Provider and Generic Backends](https://quantum.cloud.ibm.com/docs/en/api/qiskit/providers_fake_provider)
3. [Qiskit AER](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html)
4. [AER Provider](https://qiskit.github.io/qiskit-aer/apidocs/aer_provider.html#)
   1. [AER Simulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
   2. [QASM Simulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.QasmSimulator.html)
   3. [State Vector Simulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.StatevectorSimulator.html)
   4. [Unitary Simulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.UnitarySimulator.html)
   5. [AER Error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerError.html)
5. [Device backend noise model simulations](https://qiskit.github.io/qiskit-aer/tutorials/2_device_noise_simulation.html)
6. [The Extended Stabilizer Simulator](https://qiskit.github.io/qiskit-aer/tutorials/6_extended_stabilizer_tutorial.html)
7. [Matrix product state simulation method](https://qiskit.github.io/qiskit-aer/tutorials/7_matrix_product_state_method.html)
8. [Parallel GPU Quantum Circuit Simulations on Qiskit Aer ](https://www.youtube.com/watch?v=T-a__rCfKTE&ab_channel=NERSC)


#### Simulators: Implementations
1. [Running PUBs on qiskit 1.x using Aer Simulator and IBM Quantum Computers](https://www.youtube.com/watch?v=3KjHMli2jnc)
2. [Qiskit Aer's AerSimulator](https://www.youtube.com/watch?v=YqHbmOARKM4&ab_channel=DiegoEmilioSerrano)


### Assignment 2
Implement (on IBM-Q) digital simulations of the time-evolution of the two- and three-spin Heisenberg model, explaining what are the main differences in the two implementations.

### Assignment 3
Complete the following lab from IBM Quantum Challenge 2025: [Lab on Transpilers](https://colab.research.google.com/drive/10EKqVhYnmEhu1aODN9di8_SIyHqndaQP?usp=sharing)

### Assigment 4
(Continuation of Assignment 2)
1. Using the QASM simulation, compute estimates of various observables of the system as a function of time, and in particular of the dynamical correlation functions.
2. Run the simulation with the QASM simulator, including noise, and study how rapidly the quality of the solution deteriorates as a function of the time interval that is being simulated. This is done by computing the fidelity of the solution.
3. Discuss the origin of errors and, in particular, study the relative importance of the finite time step error (called Suzuki-Trotter error) and of the errors originating from the quantum hardware itself.

### Assignment 5
Complete the following lab from IBM Quantum Challenge 2025: [Lab on AI Transpilers](https://colab.research.google.com/drive/1R7snjOibRUDpVX-jh4cfzsA9ih4EkLsg?usp=sharing)

## Week 4 | Quantum Primitives and Knitting

### Qiskit Primitives
1. [Primitves Introduction Notebook](colab.research.google.com/drive/1ERpEjoMrR_gSae8bv7CMLCvrM10wbEl0?usp=sharing#scrollTo=K_syeGQaO0hn)
2. [Primitives Documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/primitives)
3. [Primitves Guide](https://docs.quantum.ibm.com/guides/primitives)
4. [Nice Video on Primitives]( https://www.youtube.com/watch?v=OuYz02clnx4&ab_channel=Qiskit)

### Quantum Knitting
1. [Need of Quantum Knitting](www.ibm.com/quantum/blog/circuit-knitting-with-classical-communication)
2. [Circuit Cutting](https://youtu.be/0LCj434k8VA?si=TlQOMUBu4stCNQZz)
3. Additional Read: [Circuit knitting with classical communication](https://arxiv.org/pdf/2205.00016)


### Assignment 6
Complete the following lab from IBM Quantum Challenge 2025: [Lab on Quantum Knitting](https://colab.research.google.com/drive/1eoqEy9oFlx3S9mIKroFpS44CiMNtcTa0?usp=sharing)



## Week 5 | Introductory Quantum Algorithms 

### Deutsch-Jozsa Algorithm
1. Quantum Parallelism: [1.4.2](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A271%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D) of Nielsen Chuang
2. Deutsch Algorithm: [1.4.3](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A282%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D) of Nielsen Chuang
3. Deutsch Jozsa Algorithm: [1.4.4](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A289%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D) of Nielsen Chuang
4. Nice [Video](https://www.youtube.com/watch?v=7MdEHsRZxvo) with visualizations

### Assignment 7
Implement Deutsch Jozsa Algorithm in Qiskit

### Grovers Algorithm
1. [3B1B Video on Grovers Algorithm](https://www.youtube.com/watch?v=RQWpF2Gb-gU&t=1912s) for intuitive and visual understanding
2. [Overview of Grovers Algorithm](https://www.youtube.com/watch?v=EoH3JeqA55A)
3. [Deep Dive and Query Complexity Analysis](https://www.youtube.com/watch?v=hnpjC8WQVrQ&t=1711s&ab_channel=Qiskit)
4. Theory from Nielsen Chuang: [6: Quantum search algorithms](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A1305%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D)
5. [Grover’s Algorithm and Amplitude Amplification (Implementation)](https://qiskit-community.github.io/qiskit-algorithms/tutorials/06_grover.html)
6. [Grover’s algorithm examples (Implementation)](https://qiskit-community.github.io/qiskit-algorithms/tutorials/07_grover_examples.html)

### Assignment 8
Implement Grover's Algo in qiskit.

### QFT and Shor's Algorithm
1. Videos from Qiskit:
   1. [Understanding Quantum Fourier Transform, Quantum Phase Estimation - Part 1](https://www.youtube.com/watch?v=mAHC1dWKNYE&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=9)
   2. [Understanding Quantum Fourier Transform, Quantum Phase Estimation - Part 2](https://www.youtube.com/watch?v=pq2jkfJlLmY&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=9)
   3. [Understanding Quantum Fourier Transform, Quantum Phase Estimation - Part 3](https://www.youtube.com/watch?v=5kcoaanYyZw&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=10)
   4. [From Factoring to Period-Finding, Writing the Quantum Program - Part 1](https://www.youtube.com/watch?v=YpcT8u2a2jc&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=11)
   5. [From Factoring to Period-Finding, Writing the Quantum Program - Part 2](https://www.youtube.com/watch?v=dscRoTBPeso&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=12)
   6. [From Factoring to Period-Finding, Writing the Quantum Program - Part 3](https://www.youtube.com/watch?v=IFmkzWF-S2k&list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR&index=13)
2. Theory from Nielsen Chuang:
   1. [5.1: The quantum Fourier transform](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A1150%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D)
   2. [5.2: Phase estimation](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A1207%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D)
   3. [5.3: Applications: order-finding and factoring](https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf#%5B%7B%22num%22%3A1232%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D)


### Assignment 9
Implement Quantum Fourier Transform, Inverse Quantum Fourier Transform, Quantum Phase Estimation and Shor's algorithm using Qiskit.

### Revision
Once you're done with the first 5 weeks, go through [this video](https://www.youtube.com/watch?v=tsbCSkvHhMo&ab_channel=freeCodeCamp.org), which summarizes all the theoretical concepts discussed in the project till now. 

## Introduction to Machine Learning
1. Introduction to Machine Learning
   1. Nice Introductory [Video](https://www.youtube.com/watch?v=ukzFI9rgwfU)
   2. Week 2, Day 4 of [Pclub ML Roadmap](https://pclub.in/roadmap/2024/06/06/ml-roadmap) 
2. K-Nearest Neighbours: Week 2, Day 4 of [Pclub ML Roadmap](https://pclub.in/roadmap/2024/06/06/ml-roadmap)
3. [Linear Regression and Gradient Descent](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week2-Day5)
4. [Logistic Regression](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week4-Day1)
5. [Support Vector Machines](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week4-Day6)
6. [Kernel Methods](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week4-Day7)
7. [Neural Networks](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week5)

## Week 6 | Introduction to Quantum Machine Learning

### Quantum Gradients

First three resources are extremely important and enough to understand basics of Quantum Gradient:
1. [Estimating Gradients and Higher-Order Derivatives on Quantum Hardware](https://www.youtube.com/watch?v=oM-WTddjNqA&t=135s&ab_channel=Qiskit)
2. [Qiskit Algorithms Gradient Framework](https://qiskit-community.github.io/qiskit-algorithms/tutorials/12_gradients_framework.html)
3. [Gradients Docs](https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.gradients.html)

For a better understanding, you can also go through these resources: 
4. [Review of Quantum Gradient Descent](https://physlab.org/wp-content/uploads/2023/04/QuantumGD_24100266.pdf)
5. [Quantum natural gradient](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient)
6. [Quantum gradients with backpropagation](https://pennylane.ai/qml/demos/tutorial_backprop)

(5) and (6) use the pennylane library, which has not been introduced to the mentees. So, they can just go through the theory (which is more than enough for understanding the theory of Quantum Gradients), or if interested go through the code (syntax is pretty intuitive).

Additional Resources (Advanced):
1. [Quantum Gates and Gradients](https://www.youtube.com/watch?v=cobp2Sf5f3o)
2. [Paper: Estimating the gradient and higher-order derivatives on quantum hardware](https://arxiv.org/pdf/2008.06517)
3. [Paper: Gradients of parameterized quantum gates using the parameter-shift rule and gate decomposition](https://arxiv.org/pdf/1905.13311)
4. [Paper: Quantum Natural Gradient](https://arxiv.org/pdf/1909.02108)

**Note:** All the contents in the resources are extracted from Papers given in Additional Resources 2, 3 and 4. Reading them would just provide you mathematical clarity of how and why things are done in a specific manner.

### Paper Presentation
1. [Classification with QNNs on Near Term Processors](https://arxiv.org/pdf/1802.06002)
2. [Continuous Variable QNNs](https://arxiv.org/pdf/1806.06871)
3. [Quantum K-Nearest Neighbour](https://arxiv.org/pdf/1409.3097)
4. [Quantum K-Means](https://arxiv.org/pdf/1307.0411)

### Variational Quantum Algorithms
1. [Theoretical Introduction to Variational Quantum Eigensolvers](https://www.youtube.com/watch?v=TUFovZsBcW4&ab_channel=Qiskit)
2. [Intro to Algorithms and VQE implementation using Qiskit Algorithms](https://qiskit-community.github.io/qiskit-algorithms/tutorials/01_algorithms_introduction.html)
3. [Advanced VQE Options](https://qiskit-community.github.io/qiskit-algorithms/tutorials/02_vqe_advanced_options.html)
4. [VQE with Qiskit Aer Primitives](https://qiskit-community.github.io/qiskit-algorithms/tutorials/03_vqe_simulation_with_noise.html)
5. Variational Quantum Deflation (VQD) Algorithm
   1. [Reference Paper](https://arxiv.org/pdf/1805.08138)
   2. [API Reference](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.VQD.html)
   3. [Implementation using Qiskit Algorithms](https://qiskit-community.github.io/qiskit-algorithms/tutorials/04_vqd.html)
6. [Testing a Variational Quantum Classifier on a real backend](https://github.com/qiskit-community/ibm-quantum-challenge-2024/blob/main/content/lab_4/lab-4.ipynb)

Additional Reading: [Variational quantum algorithms](https://arxiv.org/pdf/2012.09265)

### Assignment 10
1. Implement a VQE, try out Advanced VQE Options and Variational Quantum Deflation Algorithms (references provided in the resources). 
2. VQC
   1. Implement a Variational Quantum Classifier as given in [Lab 4](https://github.com/qiskit-community/ibm-quantum-challenge-2024/blob/main/content/lab_4/lab-4.ipynb)
   2. For the `birds_dataset.csv`, implement a classical classification algorithm
   3. Compare and contrast the training speed and accuracy, inference speed and accuracy and other metrics for (2.1) and (2.2)  

## Week 7 | Quantum Machine Learning Algorithms
### Quantum Neural Networks
1. [Quantum Neural Networks](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html)
2. [Neural Network Classifier & Regressor](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html)
3. [Training a Quantum Model on a Real Dataset](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html)
4. [Saving, Loading Qiskit Machine Learning Models and Continuous Training](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/09_saving_and_loading_models.html)
5. [Effective Dimension of Qiskit Neural Networks](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/10_effective_dimension.html)

Additional Theoretical Reading: [Quantum Neural Networks](https://arxiv.org/pdf/2205.08154)

### Assignment 11
Implement the notebooks for:
1. QNNs
2. NN Classifier and Regressor
3. Training a Quantum Model on a Real Dataset

In all these 3 implementations:
1. Interrupt training and save the model
2. Load the model, continue training till the end 
3. Calculate the effective dimensions of the algorithms

Project Mentors:
- Advaith GS | advaithgs23@iitk.ac.in | +91 88009 25876
- Himanshu Sharma | himans23@iitk.ac.in | +91 999963 3455