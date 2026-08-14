---
layout: post
title: "Procedural Generation: Algorithms for Infinite Worlds"
date: 2026-08-11 
author: Austin Shijo & Sujal Satish Motagi
tags:
- graphics
- procedural-generation
- terrain-generation
- perlin-noise
- fractal-brownian-motion
- splines
- signed-distance-fields

categories:
- events
image:
  url: /images/proc-gen-cover.png
---

# Procedural Generation: Algorithms for Infinite Worlds

Have you ever wondered how games like Minecraft, Terraria and others with seemingly infinite worlds manage to achieve this? That's clearly too much work for artists to manually do. So we have algorithms which can generate these worlds randomly.
Procedural Generation is often romanticised as a "magical seed" that creates an infinite universe.  When we say "procedural generation", we primarily refer to the class of algorithms that synthesize content via mathematical functions, specifically Noise, Splines, and Signed Distance Fields (SDFs), rather than manual artist authorship.

However, in practice, it is largely seen as a black box of noise functions that frequently result in unplayable or repetitive chaos. That is, even if a Perlin Noise function can provide a heightmap, we often cannot understand *why* it produced a specific artifact, nor can we easily tune it without breaking the entire world generation pipeline.

ProcGen isn't just "more random", it is about **Constraint** and **Local Control**. In this blog, we will discuss the architecture of modern procedural pipelines, moving from the foundational mathematics of Fractal Brownian Motion to the systems-level implementation of Parallel Reduction on GPUs, and finally to the spline-based shaping that powers modern voxel engines.

Some of the aforementioned terms might sound intimidating (wait until we get to "Parallel Reduction via Warp Intrinsics" 🙃), but they are just optimized algorithms represented with complex notation. In this blog, we will break down these mathematically intensive techniques in an intuitive manner.

## Chapter 1 | The Status Quo: Fractal Brownian Motion & Derivatives

If you are already well versed with Perlin Noise, Gradient Vectors, and basic FBM, you can skip to Chapter 2.

### 1.1 | Beyond White Noise

The most naive approach to generation is White Noise (pure randomness). If we map a function $f(x) \to \text{random}(0, 1)$, we get static:

<div style="text-align: center;">
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f6/White-noise-mv255-240x180.png">
</div>

There is no continuity. To create a "world," we need **coherence**. This is typically solved by Gradient Noise (Perlin/Simplex), where we interpolate between random gradients at grid points.

However, a single layer of noise looks like "blobs." 
<div style="text-align: center;">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Perlin_noise_example.png/500px-Perlin_noise_example.png" width=230>
</div>

To achieve the jagged, natural look of terrain, we employ **Fractal Brownian Motion (FBM)**. This is the summation of multiple "octaves" of noise, where each successive layer adds finer detail.
$$f(x) = \sum_{i=0}^{n} A \cdot G^i \cdot \text{noise}(F \cdot L^i \cdot x)$$

Where:

* $G$ is Gain (usually $0.5$, reducing amplitude each step).
* $L$ is Lacunarity (usually $2.0$, increasing frequency each step).

Each term is one "layer" of noise which is a variation of the previous layer that is compressed by a factor of $G$ in amplitude and by a factor of $L$ in the lateral (x, y) plane and also rotated by the rotation matrix $F$.
Adding successive layers brings roughness to the surface.
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/SJLSBL_mbx.png" width=350>
</div>

The surface at the bottom is the initially generated noise, while the one shown above it is the next layer which is to be added. The resulting surface is still smooth but with more roughness to it:
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/Byv1ILOXZg.png" width=350>
</div>


Standard FBM aligns grid artifacts because every octave shares the same cardinal axes. By rotating the domain (x, z) by a specific angle (using a rotation matrix based on Pythagorean triples like 3-4-5) between each octave, we eliminate these "digital looking" alignments. This technique is called **Domain Rotation**.


### 1.2 | Analytical Derivatives for Lighting
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/Byi2UIu7-g.png" width=350>
</div>
Once we have a shape, we need to light it. The naive approach is to sample the height at $x+1$ and $z+1$ to calculate a normal vector (Finite Differences). This is expensive (requires extra noise lookups) and inaccurate.
A rigorous approach is to compute the **Analytical Derivative** of the noise function itself. Since noise is just a polynomial interpolant (usually quintic: $6t^5 - 15t^4 + 10t^3$), its derivative is continuous and computable in the same pass as the value.
By using the analytical gradient $\nabla f(x, z)$, we can compute the surface normal $\vec{n}$ instantly:
$$\vec{n} = \text{normalize}(-f_x, 1, -f_z)$$
This allows for lighting that respects the mathematical curvature of the terrain, enabling interactions like "Self-Shadowing" and "Ambient Occlusion" purely through math.


## Chapter 2 | The Architecture: Splines & Multi-Noise

While FBM gives us "roughness," it doesn't give us "structure." How do we decide where an Ocean ends and a Mountain begins?

### 2.1 | The "Spline" Revolution (Minecraft 1.18)
In older generation techniques, height was often a linear function of noise: $\text{Height} = \text{Noise} \times 64 + 64$.
This creates a dependency hell. If you want taller mountains, you multiply by 128, but now your oceans are 128 blocks deep, hitting bedrock.

Modern *Minecraft* solved this using **Splines**. Instead of using noise as the output, noise is the *input* to a Spline function.
They calculate independent noise values for parameters:
1.  **Continentalness** ($C$): Far inland vs. Coast.
2.  **Erosion** ($E$): Flat vs. Jagged.
3.  **Peaks & Valleys** ($PV$): Local variation.

The terrain height $H$ is then defined as a Piecewise Spline function $S(C, E, PV)$.
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/rkouKUumZx.png" width=350>
</div>

This offers **Local Control**. We can define a control point on the spline such that:
* If $C < -0.2$, $H = \text{Sea Level}$ (Ocean).
* If $C > 0.5$, $H = \text{High}$ (Mountain).
* But, if $E$ is high (High Erosion), we can flatten the mountain into a plateau *without affecting the ocean*.

This effectively decouples the "What" (Mountain/Ocean) from the "How" (Noise Mathematics).
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/rk8K9U_XZe.png" width=350>
</div>


### 2.2 | 3D Noise & Density Functions
To generate overhangs and caves, we must move beyond 2D heightmaps ($y = f(x, z)$) to 3D Density Fields ($d = f(x, y, z)$).
The world is defined as an isosurface where Density $d = 0$.
* $d > 0 \implies$ Stone.
* $d < 0 \implies$ Air.

However, calculating 3D Perlin noise is computationally expensive ($\mathcal{O}(N^3)$). To optimize this, we rely on a **Squashing Factor**. We bias the density based on the Y-coordinate.
$$d_{final}(x, y, z) = \text{Noise}(x, y, z) - \text{bias}(y)$$
As $y$ increases (going up into the sky), the bias becomes large, forcing the density negative (Air). As $y$ decreases, the bias forces density positive (Stone).
Overhangs appear naturally where the 3D noise locally overcomes the vertical bias.
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/r1Xpa8_XWe.png" width=350>
</div>


### 2.3 | Noise Ridging (Spaghetti Caves)
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/ryu8xwuQWx.png" width=600>
</div>

A specific technique for creating tunnel-like structures is **Noise Ridging**. Instead of looking for $Noise > 0$, we look for noise values close to zero.
$$d_{cave} = \text{Thickness} - | \text{Noise}(x, y, z) |$$
If the noise is $0.0$, the absolute value is $0$, and density is positive (Air/Cave). As the noise moves away from $0$ (either positive or negative), the absolute value increases, turning the density negative (Stone). This creates long, winding "Spaghetti" tunnels along the zero-contours of the noise field.

## Chapter 3 | Taming the Chaos: Parallel Reduction on GPU

So far we have discussed generating shapes. But procedural generation also involves placing objects (trees, particles, fractals). Unbounded math tends to explode. If we use an Iterated Function System (IFS) to generate a fractal, points can drift to infinity. We need to constrain them to a Bounding Box (AABB) to render them efficiently.

### 3.1 | The Constraints Problem
Finding the bounding box of 4 million particles is trivial on a CPU:
```
// O(N) Sequential - Too slow (~9ms)
for(auto p : particles) {
    min_x = min(min_x, p.x);
    max_x = max(max_x, p.x);
}
```
On a GPU, we have thousands of threads. We cannot have a single variable `min_x` that everyone writes to (Race Condition), and using `atomicMin` on a single global address would serialize the entire GPU, destroying performance.

### 3.2 | Parallel Reduction Algorithm
The solution is a tree-based reduction approach, a staple of GPGPU programming.
We treat the buffer as a massive array and reduce it in steps.
1.  **Thread Level:** Each thread loads a value from VRAM to Local Register.
2.  **Warp Level (Intra-group):** Threads within a "Warp" (usually 32 threads) can communicate via **Shared Memory** or **Warp Shuffle Intrinsics**.
    * Thread 0 compares `val[0]` and `val[16]`.
    * Thread 1 compares `val[1]` and `val[17]`.
    * ... (Active threads halve each step).
3.  **Group Level:** The result of each Warp is written to Shared Memory, where the first Warp reduces *those* results.
4.  **Global Level:** The result of the Workgroup is written to global memory. We then dispatch a *second* kernel to reduce the results of the workgroups.

This reduces the complexity from linear time to logarithmic depth $\mathcal{O}(\log N)$, allowing us to calculate bounds for millions of procedural entities in $< 0.5\text{ms}$. This speed allows us to reshape and constrain the procedural generation in real-time, preventing "unplayable" or "invisible" content.

## Chapter 4 | The Polish: Atmospheric Mathematics

Finally, a procedural world is not just geometry; it is light and atmosphere.

### 4.1 | Signed Distance Field (SDF) Fog
<div style="text-align: center;">
<img src="https://hackmd.io/_uploads/Bk6fQwdXZx.png" width=600>
</div>


In traditional graphics, fog is a uniform distance calculation. In procedural art, we can use the derivative of density.
**Exponential Fog** follows the Beer-Lambert law. We can approximate light transport through the atmosphere by integrating density along the view ray.
$$T = e^{-\int_0^d \rho(t) dt}$$
By controlling the decay constants for Red, Green, and Blue channels independently ($b_r, b_g, b_b$), we create artistic atmospheres where distant mountains fade into a specific blue, while mid-ground objects retain a warm hue, simulating scattering without expensive physics simulations.

### 4.2 | Soft Shadows via Raymarching
<div style="text-align: center;">
<img src="https://iquilezles.org/articles/rmshadows/gfx03.png" width=600>
</div>

If our terrain is defined by an SDF, we get shadows for "free." When marching a ray from a surface point towards the light, we measure how close the ray comes to intersecting the terrain.
$$k = \min \left( \frac{d(p)}{t} \right)$$
Where $d(p)$ is the distance to the terrain and $t$ is the distance traveled along the shadow ray. This value $k$ approximates the "penumbra", how much light is blocked. This allows for soft, realistic shadows that harden as the object gets closer to the caster, purely derived from the mathematical definition of the shape.

## Chapter 5 | Conclusion

Procedural Generation is evolving from simple randomness to sophisticated systems of **Controlled Chaos**.
* We use **FBM** and **Analytical Derivatives** to create natural, detailed surfaces.
* We use **Splines** to architect the macro-structure, giving us local control over biomes.
* We use **Parallel Reduction** on GPUs to enforce bounds and constraints on millions of entities.
* We use **SDFs** to simulate light transport and atmosphere.

The "magic" of infinite worlds is actually just a very long chain of composable functions, rigorously optimized and artistically tuned.
