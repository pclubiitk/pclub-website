---
layout: post
title: "Rolling the Dice"
date: 2026-10-11
author: Aditya Sai, Jithin J Joshi
tags:
- Algorithms
- Competitive Programming
- Randomized Algorithms
- Number Theory
categories:
- events
image:
  url: /images/rolling-the-dice/title_image.jpeg
---

## A Game of Chance?

Imagine I hand you a black box containing **1 million numbers**. 
I tell you that **500,001** of these numbers are exactly the same (let's call this number $X$), and the rest are random noise.

**Your mission:** Find $X$.

You could write a loop. You check the first number, then the second, then the third... counting frequencies as you go. It works, but it's slow, tedious, and predictable. You *have* to process the data to be sure.

Now, what if I told you to just **close your eyes, reach into the box, and grab a number**?

* What are the odds you are holding $X$? **> 50%.**
* What if you do it 10 times? The odds of *never* seeing $X$ drop to less than **0.1%**.

This is the essence of **Randomized Algorithms**. Instead of fighting through data with brute force, we use randomness to cut corners. By allowing our code to make "guesses" using a source of random bits, we often find solutions that are simpler, faster, and surprisingly robust compared to their deterministic counterparts. 

In this blog, we're going to solve problems that seem computationally heavy, just by getting lucky-on purpose.

## 1. Classification

In computer science, randomized algorithms are named after two famous gambling capitals. The difference lies entirely in what you are gambling with: your time or your accuracy. Randomized algorithms are classified into:

### 1. Las Vegas Algorithms
A Las Vegas algorithm is a randomized algorithm that always gives the correct answer, but its running time varies randomly. The goal is to design an algorithm that is efficient with high probability.

**Examples:** Randomised Quicksort, Quickselect.

### 2. Monte Carlo Algorithms
A Monte Carlo algorithm is a randomized algorithm that may sometimes give a wrong answer. For such an algorithm to be useful, the probability of a wrong answer should be small.

**Examples:** Miller-Rabin Primality Test, Rabin-Karp Algorithm. 

## 2. The Miller-Rabin Primality Test: A Monte Carlo algorithm

Suppose we need to find out if a number is prime. This has massive applications in areas like Cryptography (RSA encryption), generating large hash keys, and solving complex number theory problems. 

Now, if the number is of the order of $10^9$, we can comfortably use the standard $O(\sqrt{N})$ trial division method. It takes a few milliseconds. But what if the number is of the order of $10^{18}$ (the limit of a 64-bit integer)? If you try the standard $O(\sqrt{N})$ approach, your loop will run 1 billion times. In a typical Competitive Programming environment (where the time limit is usually 1 second), this guarantees a Time Limit Exceeded (TLE). You need something much faster.

This is where Miller-Rabin test comes in:

### The 'lie detector' for numbers

We suspect a number $N$ might be prime, so we ask it a series of mathematical questions. 
- If $N$ answers even one question wrong, it is definitely composite (a liar!).
- If $N$ answers all questions correctly, it is very likely prime.

This is a Monte Carlo algorithm, since there's a chance that the questions asked may not be sufficient to conclude. 

### The two rules of interrogation

We use two rules that every prime number satisfies in order to determine primality:

#### 1. Fermat's Little Theorem

If $p$ is a prime number, then for any integer $a$ (where $1 < a < p$):$$a^{p-1} \equiv 1 \pmod p$$

Usage: We pick a random number $a$ and calculate $a^{N-1} \pmod N$, and if the result is not 1, then $N$ is definitely composite.

#### 2. Square Root Check

In modular arithmetic with prime number $p$, the number $1$ has only two square roots: $1$ and $-1$ (which is written as $p-1$).

Usage: If we find a number $x$ such that $x^2 \equiv 1 \pmod N$, but $x$ was neither $1$ nor $N-1$, then $N$ has "too many" square roots of unity. It is definitely composite.

### Algorithm

Let's test an odd number $N$.
1. Since $N$ is odd, $N-1$ is even. We factor out all the 2s from $N-1$ until we are left with an odd number $d$.$$N - 1 = d \times 2^s$$
2. Choose a random number $a$ between $[2, N-2]$. 
3. Calculate $x = a^d \pmod N$.
4. Pass Condition: If $x == 1$ or $x == N-1$, then it continues. 
5. If not, we keep squaring $x$ ($x = x^2 \pmod N$) repeatedly. We do this $s-1$ times.
6. If at any point $x$ becomes $N-1$, then $N$ survives.
7. If we finish the loop and never saw $N-1$, then $N$ is Composite.

### Deterministic Variant

Miller showed that it is possible to make the algorithm deterministic by only checking all bases  $\le O((\ln N)^2)$ . Bach later gave a concrete bound, it is only necessary to test all bases  $a \le 2 \ln(N)^2$ .

Mathematicians have showed that for testing 64 bit integer it is enough to check the first 12 prime bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, and 37.

### Implementation

Here is an implementation of the Miller-Rabin Primality test in C++: [(reference)](https://cp-algorithms.com/algebra/primality_tests.html#miller-rabin-primality-test)

```cpp
using u64 = uint64_t;
using u128 = __uint128_t;

u64 binpower(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    while (e) {
        if (e & 1)
            result = (u128)result * base % mod;
        base = (u128)base * base % mod;
        e >>= 1;
    }
    return result;
}

bool check_composite(u64 n, u64 a, u64 d, int s) {
    u64 x = binpower(a, d, n);
    if (x == 1 || x == n - 1)
        return false;
    for (int r = 1; r < s; r++) {
        x = (u128)x * x % n;
        if (x == n - 1)
            return false;
    }
    return true;
};

bool MillerRabin(u64 n, int iter=5) { 
    // returns true if n is probably prime, 
    //else returns false.
    if (n < 4)
        return n == 2 || n == 3;

    int s = 0;
    u64 d = n - 1;
    while ((d & 1) == 0) {
        d >>= 1;
        s++;
    }

    for (int i = 0; i < iter; i++) {
        int a = 2 + rand() % (n - 3);
        if (check_composite(n, a, d, s))
            return false;
    }
    return true;
}
```


## 3. The Ctrl+F Algorithm: Rabin-Karp (Monte Carlo)

Most of us use "Ctrl+F" daily. But have you ever wondered how the computer finds the word "needle" inside a "haystack" of 10 million characters instantly?

The naive approach is to slide the pattern over the text one position at a time and compare every character. If the text length is $N$ and the pattern length is $M$, this takes $O(N \cdot M)$ time. The algorithm is forced to check every single character at every position, grinding performance to a halt.

**The Randomized Insight**:
Comparing strings is slow ($O(M)$). Comparing integers is fast ($O(1)$).
What if we could turn the pattern into a unique integer (a "fingerprint") and just look for that integer in the text?

This is the **Rabin-Karp** algorithm.

### The Rolling Hash

The core logic relies on a Polynomial Rolling Hash.
We treat a string $S$ as a number in base $B$ (usually a prime roughly slightly larger than the alphabet size, like 31 or 53).

$$H(S) = (S[0] \cdot B^{M-1} + S[1] \cdot B^{M-2} + \dots + S[M-1] \cdot B^0) \pmod P$$Where $P$ is a large prime (e.g., $10^9 + 7$).

Calculating the hash of a substring takes $O(M)$. If we recalculate it for every window, we are back to $O(N \cdot M)$. However, when we slide the window one step to the right, only two things change: The leading character is removed, and a trailing character is added. We can update the hash in $O(1)$ time:$$H_{new} = \left( (H_{old} - S[remove] \cdot B^{M-1}) \cdot B + S[add] \right) \pmod P$$

### The Collision Problem & Double Hashing

A single modulo $P \approx 10^9$ allows for $10^9$ unique fingerprints. By the Birthday Paradox, if you have $\approx \sqrt{10^9} \approx 30,000$ substrings, you are likely to encounter a collision.
**The Solution** : Use two different sets of (Base, Modulo).
- Hash 1: Base $31$, Mod $10^9 + 7$
- Hash 2: Base $37$, Mod $10^9 + 9$

We only declare a match if both Hash 1 and Hash 2 match. The probability of a collision drops to $\approx \frac{1}{10^{18}}$. You are more likely to win the lottery twice in a row than to fail this check.

### Implementation

Here is an implementation of Rabin-Karp algorithm in C++: [(reference)](https://cp-algorithms.com/string/rabin-karp.html)


```cpp
vector<int> rabin_karp(string const& s, string const& t) {
    const int p = 31; 
    const int m = 1e9 + 9;
    int S = s.size(), T = t.size();

    vector<long long> p_pow(max(S, T)); 
    p_pow[0] = 1; 
    for (int i = 1; i < (int)p_pow.size(); i++) 
        p_pow[i] = (p_pow[i-1] * p) % m;

    vector<long long> h(T + 1, 0); 
    for (int i = 0; i < T; i++)
        h[i+1] = (h[i] + (t[i] - 'a' + 1) * p_pow[i]) % m; 
    long long h_s = 0; 
    for (int i = 0; i < S; i++) 
        h_s = (h_s + (s[i] - 'a' + 1) * p_pow[i]) % m; 

    vector<int> occurrences;
    for (int i = 0; i + S - 1 < T; i++) {
        long long cur_h = (h[i+S] + m - h[i]) % m;
        if (cur_h == h_s * p_pow[i] % m)
            occurrences.push_back(i);
    }
    return occurrences;
}
```

## An example: [Ghd](https://codeforces.com/problemset/problem/364/D)
A deterministic approach to find the GCD of all subsets of size $n/2$ is combinatorially impossible ($C(n, n/2)$ is massive). Standard GCD algorithms find a divisor for *all* numbers, but here we only need a divisor for *half* of them. This relaxation makes the problem harder because we don't know *which* half contains the answer.

### The Randomized Insight (Monte Carlo)
This problem is a classic candidate for a **Monte Carlo algorithm**.

Let the answer be $G$. By definition, $G$ divides at least $\frac{n}{2}$ numbers in the array.
If we pick one number $x$ uniformly at random from the array $a$, what is the probability that $x$ is divisible by $G$?

$$P(x \text{ is a multiple of } G) \ge \frac{n/2}{n} = \frac{1}{2}$$

This means if we pick a random number, there is a **50% chance** that the optimal answer $G$ is a divisor of this chosen number.

If we repeat this random sampling $k$ times:
* The probability that we **never** pick a number divisible by $G$ is $\le (\frac{1}{2})^k$.
* For $k = 10$, the failure probability is $\frac{1}{1024} \approx 0.09\%$.
* For $k = 12$, it is essentially negligible.

Thus, we can iterate a fixed number of times, assume the chosen number contains the Ghd as a divisor, and verify the best candidate.

### Algorithm Breakdown

#### 1. Random Selection
In each iteration, pick a random index `idx` and let `val = a[idx]`. We assume the Ghd is one of the divisors of `val`.

#### 2. Divisor Enumeration
Find all divisors of `val` in $O(\sqrt{val})$. Let this set be $D$.
Since $val \le 10^{12}$, the maximum number of divisors is relatively small (even for highly composite numbers near $10^{12}$, $|D| \approx 6720$, but usually much smaller).

#### 3. GCD Computation
We need to count how many numbers in the array are divisible by each divisor $d \in D$.
Directly checking `a[i] % d == 0` for all $i$ and all $d$ is too slow ($O(N \cdot |D|)$).

Instead, we optimize:
* Calculate $g_i = \gcd(val, a[i])$ for all $i$ from $0$ to $n-1$.
* Note that $g_i$ must be a divisor of `val`, so $g_i \in D$.
* Maintain a frequency array `div_counts` mapped to indices of $D$.
* If $\gcd(val, a[i]) = x$, increment the count for $x$.

#### 4. Aggregating Counts (The "Superset" Sum)
The `div_counts` array currently tells us how many numbers have a GCD of **exactly** $x$.
However, if a number has a GCD of $20$, it is also divisible by $10$, $5$, $4$, $2$, and $1$.
We need to propagate the counts from multiples to divisors.

We iterate through the sorted divisors. For every divisor $D[i]$, and every larger divisor $D[j]$ (where $j > i$):
* If $D[j]$ is a multiple of $D[i]$ (i.e., `divs[j] % divs[i] == 0`), then every number that yielded a GCD of $D[j]$ is **also** divisible by $D[i]$.
* We add `div_counts[j]` to `div_counts[i]`.

This runs in $O(\vert D \vert^2)$. Since $\vert D \vert$ is usually small, this is very fast.

#### 5. Update Answer
After aggregation, if `div_counts[i] * 2 >= n`, then $D[i]$ is a valid candidate for Ghd. We maximize our result.

### Time Complexity
The complexity per iteration is:
$$O(N \cdot \log(\text{MaxVal}) + |D|^2)$$
* $N \cdot \log(\text{MaxVal})$ comes from calculating GCD with every element.
* $\vert D \vert^2$ comes from the nested loop to aggregate divisor counts.

With $K$ iterations (where $K \approx 10$), the total time is well within the limits for $N=10^6$.

---

## C++ Solution
```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

ll calc_gcd(ll a, ll b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}

// Function to get all divisors of a number
vector<ll> get_divisors(ll n) {
    vector<ll> divs;
    for (ll i = 1; i * i <= n; ++i) {
        if (n % i == 0) {
            divs.push_back(i);
            if (i * i != n) {
                divs.push_back(n / i);
            }
        }
    }
    sort(divs.begin(), divs.end());
    return divs;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    cin >> n;

    vector<ll> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

    ll max_ghd = 1; 
    int iterations = 0;
    while (iterations < 12) { 
        iterations++;
        
        // Pick a random index uniformly
        uniform_int_distribution<int> dist(0, n - 1);
        int rand_idx = dist(rng);
        ll current_val = a[rand_idx];

        if (current_val <= max_ghd) continue;

        vector<ll> divs = get_divisors(current_val);

        vector<int> div_counts(divs.size(), 0);

        for (int i = 0; i < n; ++i) {
            ll g = calc_gcd(current_val, a[i]);

            auto it = lower_bound(divs.begin(), divs.end(), g);
            if (it != divs.end() && *it == g) {
                int index = distance(divs.begin(), it);
                div_counts[index]++;
            }
        }

        for (int i = 0; i < divs.size(); ++i) {
            ll d = divs[i];

            if (d <= max_ghd) continue;

            int count = div_counts[i];

            for (int j = i + 1; j < divs.size(); ++j) {
                if (divs[j] % d == 0) {
                    count += div_counts[j];
                }
            }

            if (count * 2 >= n) {
                max_ghd = max(max_ghd, d);
            }
        }
    }

    cout << max_ghd << endl;
    return 0;
}
```

## 4. An example of Guaranteeing Accuracy: QuickSelect

A classic algorithm to find the $k$-th smallest element in an unsorted array: The "QuickSelect" algorithm ***(you may be familiar with this if you've done DSA earlier!)***

QuickSelect is similar to QuickSort, but with a crucial difference: we only recurse into **one** side of the partition. ([reference](https://www.geeksforgeeks.org/dsa/quickselect-algorithm/))

1.  Pick a random **pivot** $p$.
2.  Partition the array into elements $< p$ and $> p$.
3.  If $p$ is the $k$-th element, stop.
4.  Otherwise, recurse only into the subarray that contains the $k$-th target.

### The Mathematical Guarantee
Why is this $O(N)$ and not $O(N \log N)$?

```cpp
// Standard partition logic
int partition(vector<int>& arr, int left, int right) {
    int pivotIndex = left + rand() % (right - left + 1);
    int pivotValue = arr[pivotIndex];
    swap(arr[pivotIndex], arr[right]); 
    
    int storeIndex = left;
    for(int i = left; i < right; ++i) {
        if(arr[i] < pivotValue) {
            swap(arr[i], arr[storeIndex]);
            storeIndex++;
        }
    }
    swap(arr[storeIndex], arr[right]);
    return storeIndex;
}

int quickSelect(vector<int>& arr, int left, int right, int k) {
    if (left == right) return arr[left];

    int pivotIndex = partition(arr, left, right);

    if (k == pivotIndex) {
        return arr[k]; // Found it! Guaranteed correct.
    } else if (k < pivotIndex) {
        return quickSelect(arr, left, pivotIndex - 1, k);
    } else {
        return quickSelect(arr, pivotIndex + 1, right, k);
    }
}
```

Let $T(n)$ be the random variable representing the number of comparisons required for an array of size $n$.

#### 1. The "Good" Pivot Definition
Let's define a **"good pivot"** as one that lands in the middle **50%** of the sorted order (between the 25th and 75th percentiles).
* If we pick a good pivot, the size of the remaining subarray is at most $\frac{3}{4}n$.
* Since we pick uniformly at random, the probability of picking a good pivot is:
    $$P(\text{good}) = \frac{0.75n - 0.25n}{n} = \frac{1}{2}$$

#### 2. Expected Cost
Since the probability of success (finding a good pivot) is $1/2$, this follows a Geometric Distribution with expected trials $= 2$.
On average, we do $2 \times n$ work to reduce the problem size from $n$ to $\frac{3}{4}n$.

We can model the expected runtime $E[T(n)]$ as a geometric series:
$$
E[T(n)] \le 2n + 2\left(\frac{3}{4}n\right) + 2\left(\frac{3}{4}\right)^2 n + \dots
$$

Factor out $2n$:
$$
E[T(n)] \le 2n \sum_{i=0}^{\infty} \left(\frac{3}{4}\right)^i
$$

The sum of the geometric series $\sum r^i$ where $r < 1$ is $\frac{1}{1-r}$. Here $r = 3/4$:
$$
\frac{1}{1 - 3/4} = \frac{1}{1/4} = 4
$$

Substituting back:
$$
E[T(n)] \le 2n \times 4 = 8n
$$

#### Conclusion
Thus, **$E[T(n)] = O(N)$**.
While the worst-case scenario (picking the smallest element every time) is $O(N^2)$, the randomization makes this event vanishingly rare, giving us a practical linear-time solution.

## 5. Finding a Quadratic Non-Residue- A classic Las Vegas problem

Given an odd prime $p$, we want to find an integer $a$ such that $a$ is a **Quadratic Non-Residue** modulo $p$.

Mathematically, we need an $a$ where the congruence:
$$x^2 \equiv a \pmod p$$
has **no solution**.

### The Algorithm
1.  Pick a random integer $a$ uniformly from $[1, p-1]$.
2.  Check if $a$ is a non-residue using **Euler's Criterion** (explained below). For reference, you may also have a look at [this](https://en.wikipedia.org/wiki/Euler%27s_criterion).
3.  If it is, return $a$.
4.  If not, repeat step 1.

This is a **Las Vegas** algorithm because: [reference](https://eprint.iacr.org/2012/470.pdf)
* **Accuracy:** It never returns a wrong answer (Euler's criterion is deterministic).
* **Runtime:** It might take 1 iteration, or it might take 100 (random variable).

---

### The Math Behind It

#### Euler's Criterion
How do we check if $a$ is a non-residue efficiently? We use the **Legendre Symbol** $\left(\frac{a}{p}\right)$. Euler's criterion states:

$$
\left(\frac{a}{p}\right) \equiv a^{\frac{p-1}{2}} \pmod p
$$

* If $a^{\frac{p-1}{2}} \equiv 1 \pmod p$, then $a$ is a quadratic residue (a perfect square).
* If $a^{\frac{p-1}{2}} \equiv -1 \pmod p$, then $a$ is a non-residue.
* If $a^{\frac{p-1}{2}} \equiv 0 \pmod p$, then $a$ is a multiple of $p$.

This check takes $O(\log p)$ time using Binary Exponentiation (Modular Exponentiation).

#### Probability Analysis 
The efficiency depends on the distribution of residues and non-residues in the multiplicative group $\mathbb{Z}_p^*$.

**Theorem:** For any odd prime $p$, exactly half of the integers in $\{1, \dots, p-1\}$ are quadratic residues, and the other half are non-residues.

**Proof Sketch:**
Let $g$ be a primitive root modulo $p$. The elements of $\mathbb{Z}_p^*$ can be written as $g^1, g^2, \dots, g^{p-1}$.
* An element $g^k$ is a quadratic residue if and only if $k$ is even.
* Since $p$ is odd, $p-1$ is even.
* In the range $[1, p-1]$, there are exactly $\frac{p-1}{2}$ even powers (residues) and $\frac{p-1}{2}$ odd powers (non-residues).

**Conclusion:** If we pick $a$ at random, the probability of success (finding a non-residue) is:
$$P(\text{success}) = \frac{(p-1)/2}{p-1} = \frac{1}{2}$$

#### Expected Runtime 
Since each trial is an independent Bernoulli trial with $p = 0.5$, the number of trials $T$ follows a [**Geometric Distribution**](https://en.wikipedia.org/wiki/Geometric_distribution). 

That means we are essentially performing a series of coin flips where "Heads" is a failure (residue) and "Tails" is a success (non-residue)! The distribution models the number of flips required to get the **first success**.

The expected number of iterations is:
$$E[T] = \frac{1}{p} = \frac{1}{0.5} = 2$$

This proves that although the algorithm *could* theoretically loop forever (if we are incredibly unlucky), it is **expected** to finish in just 2 iterations.

### Implementation
This is often used as a subroutine in problems involving modular square roots.

```cpp
using ll = long long;

ll power(ll base, ll exp, ll mod) {
    ll res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (__int128)res * base % mod;
        base = (__int128)base * base % mod;
        exp /= 2;
    }
    return res;
}

ll find_non_residue(ll p) {
    static mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<ll> dist(1, p - 1);

    while (true) {
        ll a = dist(rng);
        
        // Euler's Criterion Check
        if (power(a, (p - 1) / 2, p) == p - 1) {
            return a;
        }
        // If not, we automatically loop again 
    }
}
```

## 6. The Factorization Gamble: Pollard's Rho

While the Quadratic Non-Residue algorithm searches for a property, **Pollard's Rho** searches for a *COLLISION*. It is one of the most elegant applications of probability theory to Number Theory. [Reference](https://www.geeksforgeeks.org/dsa/pollards-rho-algorithm-prime-factorization/)

Classifying it requires nuance:
* If you run it for a fixed number of steps and stop, it's **Monte Carlo** *(might fail to find a factor)*.
* If you randomize the parameters and restart until a factor is found, it becomes **Las Vegas** *(guaranteed correct factor with random runtime)*.

### The Mathematical Problem
Given a composite integer $n$, find a non-trivial factor $d$ (where $1 < d < n$).

Trial division takes $O(\sqrt{n})$. For $n \approx 10^{18}$, this is impossible ($\approx 10^9$ operations).
Pollard's Rho aims for $O(n^{1/4})$, which is drastically faster ($\approx 10^{4.5}$ or $\approx 32,000$ operations).

### The Algorithm
1.  Define a [*pseudorandom sequence*](https://www.geeksforgeeks.org/dsa/pseudo-random-number-generator-prng/) using a polynomial, typically:
    $$x_{i+1} = (x_i^2 + c) \pmod n$$
2.  We want to find two indices $i \neq j$ such that $x_i \equiv x_j \pmod p$, where $p$ is an unknown small prime factor of $n$.
3.  If $x_i \equiv x_j \pmod p$, then $p$ divides the difference $|x_i - x_j|$.
4.  Since $p$ also divides $n$, we can find $p$ by computing:
    $$g = \gcd(|x_i - x_j|, n)$$
5.  If $1 < g < n$, we found our factor!

To detect this efficiently without storing the sequence, we use [**Floyd's Cycle-Finding Algorithm**](https://www.geeksforgeeks.org/dsa/floyds-cycle-finding-algorithm/) ("Tortoise and Hare").

---

### How does it come about?

#### 1. The Birthday Paradox
Why is this fast?
Consider the sequence modulo $p$ (the smallest prime factor of $n$). The sequence behaves like a random walk on the values $\{0, 1, \dots, p-1\}$.

The **Birthday Paradox** states that if you draw random integers from a set of size $N$, you expect a collision (repetition) after drawing approximately $\sqrt{N}$ numbers.

This also finds [applications in cryptography](https://www.geeksforgeeks.org/ethical-hacking/birthday-attack-in-cryptography/).

* Here, our "set" is integers modulo $p$.
* We expect a collision modulo $p$ after $\sqrt{p}$ steps.
* Since $p$ is a factor, $p \le \sqrt{n}$.
* Therefore, the expected runtime is roughly $\sqrt{\sqrt{n}} = n^{1/4}$.

#### 2. The "Rho" Shape ($\rho$)
Since the sequence $x_{i+1} = f(x_i)$ is deterministic given a start point, once a value repeats, the sequence enters a cycle.
* The initial segment is the "tail".
* The cycle is the "loop".
* Together, they look like the Greek letter **Rho ($\rho$)**.


Occasionally, the algorithm might find $g = n$ (failure) or get stuck in a cycle that doesn't reveal a factor.

To make it a true **Las Vegas** algorithm: If it fails, we simply pick a different constant $c$ (the random seed of our polynomial) and **restart**.

---

### Implementation

```cpp
#include <bits/stdc++.h> 
using namespace std;
using ll = long long;

ll mul_mod(ll a, ll b, ll m) {
    return (__int128)a * b % m;
}

ll f(ll x, ll c, ll n) {
    return (mul_mod(x, x, n) + c) % n;
}

// Las Vegas implementation of Pollard's Rho
ll pollard_rho(ll n) {
    if (n % 2 == 0) return 2;
    
    // 1. Initialize random parameters
    ll c = 1; 
    
    // Keep trying different 'c' until factor is found
    while (true) {
        ll x = 2; 
        ll y = 2; 
        ll d = 1; 
        
        while (d == 1) {
            x = f(x, c, n);
            y = f(y, c, n);
            y = f(y, c, n);
            d = gcd(abs(x - y), n);
        
        }

        if (d != n) {
            return d;
        }
        
        // Failure case: d == n means we found the number itself (bad luck).
        c++;
    }
}
```

## Conclusion

Randomized algorithms fundamentally reshape our approach to problem-solving by exchanging the rigidity of deterministic guarantees for the pragmatic power of probability. Whether employing Monte Carlo methods to approximate solutions in vast search spaces or leveraging Las Vegas algorithms to ensure accuracy through variable runtime, these techniques allow us to bypass the prohibitive costs of worst-case scenarios. 

From cracking number-theoretic barriers in cryptography to optimizing competitive programming solutions, embracing randomness proves that sometimes the most efficient path to certainty is a calculated roll of the dice :)

## Sources and Additional Reading

1. [Randomized Algorithms- Errichto](https://codeforces.com/blog/entry/71097)
2. [Hashing- When you want chaos](https://www.sidhantbansal.com/2025/Hashing-when-you-want-chaos/)
3. [Primality testing](https://www.cs.cmu.edu/~harchol/Probability/chapters/chpt23.pdf)
4. [When do randomized algorithms perform best?](https://medium.com/@sachi05july/when-do-randomized-algorithms-perform-best-d7eae1ca806a)

#### Authors: Aditya Sai, Jithin J Joshi