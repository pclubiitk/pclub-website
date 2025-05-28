---
layout: post
title: "Game Hacking: Exploiting the Pwnie Island Game"
date: 2025-05-26 19:30:00 +0530
author: Aayush Anand, Aryan Mahesh Gupta, Harshit Tomar, Rishi Divya Kirti
category: Project
tags:
- summer25
- project
categories:
- project
hidden: true
summary:
- The project progresses from basics of memory exploitation to advanced kernel level anti-cheat bypasses. We will be exploiting the Pwnie Island game (a deliberately vulnerable MMORPG).

image:
  url: "https://i.ytimg.com/vi/RDZnlcnmPUA/hq720.jpg?sqp=-oaymwEXCK4FEIIDSFryq4qpAwkIARUAAIhCGAE=&rs=AOn4CLANoeYWna49fV7Gv7zC9lWWtyzUug"
---


# About the Project
The project progresses from basics of memory exploitation to advanced kernel level anti-cheat bypasses. For hands-on, we will be exploiting the Pwnie Island game (a deliberately vulnerable MMORPG).

# Resources

## Meet 1
- **Setting up the Pwnie Adventure 3 Game** : 
    - Client : https://www.pwnadventure.com/
    - Sever setup guide : https://hackmd.io/@codeIMperfect/rJQVjowbxl and https://docs.google.com/document/d/1u8OUS_gWtqCxrzFDz3qHHwZgAyFWtIfH1iakPPo5MvI/edit?usp=sharing
- **Compilation** : https://www.youtube.com/watch?v=ksJ9bdSX5Yo
- **Linking** : CSAPP(Computer Systems: A Programmer's Perspective Book) Ch-7 [Don't need to read all of it, just read stuff relevant to today's discussion]

## Meet 2
- **PLT** and **GOT**:
    - https://docs.thecodeguardian.dev/operating-systems/linux-operating-system/understanding-plt-and-got
- **GDB**:
    - https://ctf101.org/reverse-engineering/what-is-gdb/
    - Try these **4 challenges**:
        - https://play.picoctf.org/practice/challenge/395?category=3&page=1&search=GDB
        - https://play.picoctf.org/practice/challenge/396?category=3&page=1&search=GDB
        - https://play.picoctf.org/practice/challenge/397?category=3&page=1&search=GDB
        - https://play.picoctf.org/practice/challenge/398?category=3&page=1&search=GDB

## Meet 3
**Topics**: 
- **C++** : Aggregate Initialization, Structs and Classes, Inheritance, Polymorphism, Constructors, Virtual Functions, VTables
- Reversing, Assembly

**Resources**:
For all the **C++ stuff** covered in last 2 sessions, read following chapters from learncpp.com :
- Introduction to Classes, constructors etc. (chapter 14)
- Operator Overloading (chapter 21)
- Inheritance (chapter 24)
- Virtual Functions, VTables, vptr (chapter 25)

**Do attempt exercises from above chapters** to get a better hold of topics.

**Reversing 101** : https://hackmd.io/@rdksupe/B1bo2ixzlg


If you are interested you can also watch these videos (not relevant directly) just interesting : 
- https://youtu.be/rlM9JGx81xk?si=1MIx0aiNnYA5R-uD 
- https://youtu.be/suABtb8_2Zk?si=hDHIgwhl8Mx0NTrQ

To practice reversing, practicing challenges from following websites:
- [crackme](https://crackmes.one/)
- picoctf
- microcorruption