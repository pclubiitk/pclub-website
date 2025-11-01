---
layout: post
title: "Systems Roadmap"
date: 2025-11-01
author: Austin Shijo
category: Roadmap
tags:
  - roadmap
  - systems
categories:
  - roadmap
hidden: true
image:
  url: /images/systems-roadmap/systems.png
---

# Systems Guide

## Table of Contents

#### [General Skills](#id-GeneralSkills)
     [Basics of C++](#id-BasicsofC)  
     [Basics of C](#id-BasicsofC-1)  
     [Tooling for C/C++](#id-ToolingforCC)  
     [Object Oriented Programming](#id-ObjectOrientedProgramming)  
     [STL](#id-STL)
#### [Concurrency](#id-Concurrency)
#### [Compilers and Interpreters](#id-CompilersandInterpreters)
     [Lexing](#id-Lexing)  
     [Parsing](#id-Parsing)  
     [Variables, Conditions and More](#id-VariablesConditionsandMore)  
     [Compilers and Code Generation](#id-CompilersandCodeGeneration)  
     [The Road Ahead](#id-TheRoadAhead)  
#### [KernelDev](#id-KernelDev)
#### [Network Programming](#id-NetworkProgramming)
     [Network Models and Architecture](#id-NetworkModelsandArchitecture)  
     [OSI (Open Systems Interconnection) Model](#id-OSIOpenSystemsInterconnectionModel)  
     [TCP/IP Model](#id-TCPIPModel)  
     [Protocols (by layer)](#id-Protocolsbylayer)  
     [Advanced Network Programming Concepts](#id-AdvancedNetworkProgrammingConcepts)  
     [Blocking and Non-Blocking I/O](#id-BlockingandNon-BlockingIO)  
     [Concurrency Models](#id-ConcurrencyModels)  
     [Performance Optimization](#id-PerformanceOptimization)  
     [Socket Programming](#id-SocketProgramming)  
     [Some Practical Assignments](#id-SomePracticalAssignments)  
     [Further Resources](#id-FurtherResources)  


<div id='id-GeneralSkills'></div>
## General Skills

<div id='id-BasicsofC'></div>
### Basics of C++
- [learncpp.com](<https://www.learncpp.com/>): A good tutorial for beginners
- **C++ Primer, 5th Edition**: For beginners who prefer working with a book. Dives more in-depth than learncpp.com. 
- **A Tour of C++ by Bjarne Stroustrup**: For those with experience in another programming language, needing a quick start with C++. 


It is not essential to finish the entirety of the resources mentioned above, however for a smooth experience following the rest of this guide, we highly recommend doing the sections corresponding to General Skills from them. 
Most topics in Systems require a good deal of programming experience in these languages beforehand so get well acquainted with them.

<div id='id-BasicsofC-1'></div>
### Basics of C
- **[Beej's tutorial](<https://beej.us/guide/bgc/>)**: Most comprehensive tutorial introducing all concepts.
- **[Beej's library reference](<https://beej.us/guide/bgclr/>)**: Library reference, covering all c stdlib functions.

The above two books should introduce a beginner to all features of the language and functions in stdlib, providing examples, common pitfalls, and references to [c standard](<https://web.archive.org/web/20230402172459/https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf>).


<div id='id-ToolingforCC'></div>
### Tooling for C/C++
Important things you should learn aside from the language itself, which are really important when you are dealing with large projects, are: 

- Build systems: [makefiletutorial.com](<https://makefiletutorial.com/>). Makefiles make it harder to manage multiple libraries, so instead, a preferred, albeit equally tedious way to manage libraries is to use [cmake.org](<https://cmake.org/cmake/help/book/mastering-cmake/cmake/Help/guide/tutorial/index.html>)
- Debugging is an important skill in all of programming. For C/C++ a popular debugger is GDB. [Here's](https://www.youtube.com/watch?v=bWH-nL7v5F4) a quick tutorial that should get you started. And here's a popular [cheatsheet](<https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf>). You need not know every single command but you should be familiar with commonly used ones at least.

*************************

Make sure to interlace studying theory and tutorials with making some simple projects on your own. It's ok, and in fact recommended, to aim for projects where you don't know all the concepts required to complete them.

<div id='id-ObjectOrientedProgramming'></div>
### Object Oriented Programming
This is relevant if you're using C++. The book and website mentioned above for the basics of C++ should also cover this. Apart from this, make projects in C++ and those will inevitably need OOP concepts.

<div id='id-STL'></div>

### STL
STL is an important tool for C++ programmers as it implements a lot of useful data structures and algorithms.

Again, if you're following the book or website then this will probably be covered. Otherwise, to get familiar with STL you can watch this [Complete C++ STL in 1 Video](https://www.youtube.com/watch?v=RRVYpIET_RU) or read [The Complete Practical Guide to C++ STL(Standard Template Library)](https://abhiarrathore.medium.com/the-magic-of-c-stl-standard-template-library-e910f43379ea).
Rather than having an extensive knowledge it's important to be familiar with the available data structures and how to apply them since you'll need those quite often in software development.

<div id='id-Concurrency'></div>
## Concurrency
In any software, seemingly multiple tasks happen at the same time. Dealing with multiple tasks at once is concurrency. It is slightly different from parallelism, which, in a sense, is "True Concurrency" since it uses (and thus requires) multiple cores of a CPU to achieve concurrent systems, executing tasks in parallel on different cores.

Initially, a general idea of how concurrent systems work is enough. You should go through the talk mentioned, but before it, to get an idea of what a _thread_ is, what a _process_ is, etc. For this, properly go through multiple answers AND comments (always a good practice with Stack Exchange) in the following thread ;) on stackexchange [What is the difference between a process and a thread?](https://stackoverflow.com/questions/200469/what-is-the-difference-between-a-process-and-a-thread). For a decently complete, C++ specific implementation of concurrency, and concurrent systems, refer to Chapter 13 of A Tour of C++, titled Concurrency.  We *highly recommend* the following talk to get an idea of when to use concurrency, [CppCon 2017: Ansel Sermersheim “Multithreading is the answer. What is the question? ](https://www.youtube.com/watch?v=GNw3RXr-VJk&t=2614s&ab_channel=CppCon).


<div id='id-CompilersandInterpreters'></div>
## Compilers and Interpreters

This section deals with how to build compilers and interpreters. By the end of it you should be able to create your own programming language and maybe even contribute to an open source compiler/interpreter of your favourite language.

<div id='id-Lexing'></div>
### Lexing

The input to a compiler is the source code, which is simply a raw list of characters. This is too low level to make sense of so we first extract **tokens** from a raw string - this process is called **lexing/lexical analysis**.

- [This chapter](https://craftinginterpreters.com/scanning.html) of the Crafting Interpreters book focuses on building a lexer for the Lox programming language implemented in the book.
- [LLVM's Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl01.html) on writing a lexer.
- [This tutorial](https://lisperator.net/pltut/parser/) on writing a parser involves making a lexer.
- [Format Grammar](https://suchanek.name/work/teaching/topics/formal-grammars/): A formal grammar is a way to describe the syntax (grammar) of a language. Involves fair bit of discrete mathematics.

<div id='id-Parsing'></div>
### Parsing

Once you have a list of tokens the next step is to organize them in a more meaningful manner than a simple list. Usually this means making an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree).

- [This chapter](https://craftinginterpreters.com/parsing-expressions.html) and the [next chapter](https://craftinginterpreters.com/evaluating-expressions.html) from the book "Crafting Interpreters" explain it well.
This book is a great resource that I will be referring to again later. Note that this chapter includes code that builds on some previous code in the book and it also deals with error handling. These can be safely ignored for now.
- The same tutorial referenced above. As I said its main focus is writing a parser: https://lisperator.net/pltut/parser/

Now that you know how to make a lexer and a parser, try to apply your knowledge to a simple project.  
Try to make a mathematical expression parser which takes an input string such as `3 + sin(0)*3` for example and prints the answer.  

**Note:** This can also be done without following the usual pipeline needed for a programming language compiler. In particular, you don't necessarily need to make an AST.  
Look into the [Shunting Yard algorithm](https://en.wikipedia.org/wiki/Shunting_yard_algorithm) and [Reverse Polish notation](https://en.wikipedia.org/wiki/Reverse_Polish_notation). But since our goal is to make a compiler/interpreter, try doing it that way.

<div id='id-VariablesConditionsandMore'></div>

### Variables, Conditions and More
Now we'll move on to the more interesting stuff. But first, to get an idea of what the architecture of a compiler looks like and how you may go about designing one, read the top answer [here](https://softwareengineering.stackexchange.com/questions/165543/how-to-write-a-very-basic-compiler)

- [Crafting Interpreters](https://craftinginterpreters.com/): A great introductory book. It teaches most of the fundamental concepts that we've covered and eventually develops an interpreter for a toy programming language.


Follow this book to make a relatively complete (although not yet usable for production) programming language.  
The first part (tree walk interpreter) is written in Java but I recommend trying this out in any object-oriented language of your choice.

Note that the book makes an interpreter - first a tree walk interpreter and later a bytecode VM.
The working of the bytecode VM is very similar to a compiler, the difference being that a compiler generates assembly/machine code for a specific CPU architecture. Bytecode, on the other hand is a made-up instruction set and the VM emulates a chip running this instruction set.

- [This playlist](https://youtube.com/playlist?list=PLZQftyCk7_SdoVexSmwy_tBgs7P0b97yD&si=BbjADZf9qqQ0CdG9) on YouTube is about making an interpreter. 

<div id='id-CompilersandCodeGeneration'></div>
### Compilers and Code Generation
An interpreter goes through source code line by line and executes it on the fly. On the other hand a compiler converts the source code into native machine code, which can then be run as an executable. An interpreter may also take an intermediate approach, such as the bytecode VM in Crafting Interpreters.

Interpreters and compilers are quite similar and use a lot of the same techniques. Once you've learned how to make an interpreter you can attempt to make a compiler for one specific CPU architecture. You will need to go through a reference manual for assembly and CPU instruction sets.

- [This playlist](https://youtube.com/playlist?list=PLUDlas_Zy_qC7c5tCgTMYq2idyyT241qs&si=eMB_o1m52ohjr7jl) goes through the development of a compiler for a simple language through assembly code generation.
- Chapter 2 of [The dragon book](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools) goes through making a mini-compiler (Yes in just a chapter, that's how comprehensive this book is). It is a classic book on compiler design, but it is a fairly advanced writeup and is to be mainly used as a reference.

If you just want to dabble into the waters of generating assembly code, then a simple project could be to make a compiler for an [esolang](https://en.wikipedia.org/wiki/Esoteric_programming_language) like [Brainfuck](https://en.wikipedia.org/wiki/Brainfuck). BF is a famous esolang with a very simple instruction set, yet it is [Turing complete](https://stackoverflow.com/questions/7284/what-is-turing-complete).  
Since it has very simple instructions, writing an interpreter would be trivial. But it could be a good exercise to write a compiler for it that converts these instructions into assembly or even machine code.

<div id='id-TheRoadAhead'></div>
### The Road Ahead

By this point you probably have a good handle on the techniques used in compiler and interpreter development. Now you can go wild and explore new techniques on your own.  
I will not make this section like a roadmap but instead put out some random ideas and resources.

- Compiler architectures like LLVM will often handle the backend (code generation and optimisation) for you in a practical setting since this is a very hard task and generating machine code for several different platforms on your own is virtually impossible.  
So learn how to use tools like LLVM. https://llvm.org/docs/tutorial/
- Play around with open source compilers like GCC or your favourite language's compiler/interpreter. Maybe you could try to contribute a few bug fixes or simply locally tweak some stuff and see what changes.
- JIT Compilation - This is a hybrid of interpreting and compilation where native machine code is generated on the fly and executed. For example, JavaScript is usually JIT compiled.  
This technique allows for optimisations on parts of code that are being executed more often.
Once again, you could start out by attempting to do JIT compilation on a simple esolang like Brainfuck, like in [this video](https://youtu.be/mbFY3Rwv7XM?si=PDP3m5YC7wRw9Uvl).
- Compiler optimisations. This is another advanced topic you may explore. A lot of code written by the user can be optimised during compilation.  
For example, if you `x++` twice in a row you could replace that with a single `x += 2`.
The dragon book has a chapter or two on this.
Here's a paper on some compiler optimisations: https://www.clear.rice.edu/comp512/Lectures/Papers/1971-allen-catalog.pdf

<div id='id-KernelDev'></div>
## KernelDev
<!--  (add as further reading) -->
 
[OSDev Wiki](https://wiki.osdev.org/Expanded_Main_Page) - A great resource for learning operating system development from scratch. 
Make sure to read the [Introduction](https://wiki.osdev.org/Introduction) and [Beginner Mistakes](https://wiki.osdev.org/Beginner_Mistakes) sections.
Learn about the environment, CPU, kernels, storage devices, memory management, booting, and the "Tools" section. Read the in-page links.

Another good resource is [Linux Kernel Development](https://www.cse.iitd.ac.in/~rijurekha/col788_2023/linux_kernel_development.pdf). I would suggest going through starting 3-4 pages of every chapter of this book - this will give you a high-level understanding of how the Linux kernel is structured and maintained. You can of course go deeper and read more in-depth.

Extended learning & Roadmaps
[Linux Roadmap](https://roadmap.sh/linux) - While it’s not focused directly on writing kernel code, Linux Roadmap is an important foundation for anyone aiming to contribute to or build a kernel.
[Linux Kernel Developer Roadmap](https://github.com/Krimson-Squad/LinuxKernel_DevRoadmap.git) - You can learn up to module 3. Rest of it has yet to be updated.



<div id='id-NetworkProgramming'></div>
## Network Programming

At its core, network programming deals with **communication between processes over a network** using protocols like TCP/IP and UDP. Examples can include sending a message, fetching a webpage, transferring files, syncing game state and more.


<div id='id-NetworkModelsandArchitecture'></div>
### Network Models and Architecture

<div id='id-OSIOpenSystemsInterconnectionModel'></div>
#### OSI (Open Systems Interconnection) Model
This one is a classic. It might seem academic a bit at first glance but, this [blog](https://www.splunk.com/en_us/blog/learn/osi-model.html) does a pretty good job at explaining the OSI model.

<div id='id-TCPIPModel'></div>
#### TCP/IP Model
While most modern networks use the [TCP/IP stack](https://www.splunk.com/en_us/blog/learn/tcp-ip.html), OSI model remains a foundational tool for learning and discussing network architecture.



Guide: If you are just getting started, [Beej's Guide to Networking Concepts](https://beej.us/guide/bgnet0/) is an absolute gem.
This amazing guide should get you started with a basic understanding of networking concepts, don't get overwhelmed by this though, you can cover this at your own pace :)

<div id='id-Protocolsbylayer'></div>
### Protocols (by layer)

We will now get a high-level overview on what these are and where and how they are used.
1. Application Layer: 
    - **HTTP/1.x, [HTTP/2](https://tools.ietf.org/html/rfc7540)**: This is basically how your web browsers talk to websites.
    - **FTP** (File Transfer Protocol): for, well, file transfer across networks.
    - **SMTP**: Simple Mail Transfer Protocol
    - [**DNS**](https://howdns.works): It is like the phonebook of the internet, mapping IPs to domain names.
    - **SSH** (Secure Shell), **TLS/SSL** (security): These are like your guardians, keeping your connections secure and private.
    - **WebSockets**: For real-time and interactive experience -- think live chat or game updates without constantly refreshing.
    - **RPC frameworks**: These let programs on different computers "call" functions on each other as if they were local. Modern ones like [gRPC](https://grpc.io/blog/grpc-on-http2) leverage the power of HTTP/2's multiplexed streams. Check out [this article](https://www.cncf.io/blog/2018/07/03/http-2-smarter-at-scale/) for in-depth overview on HTTP/2.
    - **HTTP/3**: This runs over [QUIC](https://en.wikipedia.org/wiki/QUIC), promising faster and more reliable connections.

2. Transport Layer:
    - This layer manages end-to-end data transmission between systems using protocols like TCP and UDP.
        - TCP ensures every data packet arrives in order.
            - TCP [3-way handshake](https://www.geeksforgeeks.org/computer-networks/tcp-3-way-handshake-process/)
            - [TCP congestion control](https://en.wikipedia.org/wiki/TCP_congestion_control)
        - UDP is more like throwing the packet -- fast but unreliable. Some use cases for UDP include media streaming, DNS queries, and certain online games where speed outweighs the guaranteed delivery. 
    - Protocols like [RTP](https://datatracker.ietf.org/doc/html/rfc3550) (Real-time Transport Protocol) and RTCP belong here too, crucial for streaming audio and video data over IP networks.

3. [Network / Internet Layer](ca.indeed.com/career-advice/career-development/network-layer):
This layer is responsible for routing packets between different networks -- more like a GPS for your data.
    - IP (Internet Protocol): It gives every device a unique address. ([Difference between IPv4 and IPv6](https://www.geeksforgeeks.org/computer-networks/differences-between-ipv4-and-ipv6/))
    - [ICMP](https://www.cloudns.net/blog/what-is-icmp-internet-service-message-protocol/): This helps with error messages and diagnostics (example: `ping`, [ARP](https://www.geeksforgeeks.org/ethical-hacking/how-address-resolution-protocol-arp-works/)).
    - You may want to get familiar with concepts such as [Subnetting](https://www.cloudflare.com/learning/network-layer/what-is-a-subnet/), [NAT](https://www.youtube.com/watch?v=01ajHxPLxAw), [CIDR](https://aws.amazon.com/what-is/cidr/).

4. [Data Link Layer](https://www.geeksforgeeks.org/computer-networks/data-link-layer/)
It manages MAC addressing, switches/bridges, etc ensuring error-free transmission of data.
    - MAC address vs IP address: MAC addresses are hardware-based unique identifiers for network interfaces, while IP address are logical addresses used for identifying devices and routing data across networks.
    
<div id='id-AdvancedNetworkProgrammingConcepts'></div>
### Advanced Network Programming Concepts


<div id='id-BlockingandNon-BlockingIO'></div>
#### [Blocking and Non-Blocking I/O](https://medium.com/coderscorner/tale-of-client-server-and-socket-a6ef54a74763)
With blocking I/O, when a client makes a request to connect with the server, the thread that handles that connection is blocked until there is some data to read, or the data is fully written.
With non-blocking I/O, we can use a single thread to handle multiple concurrent connections.
> [This](https://www.youtube.com/watch?v=os7KcmJvtN4) explains how frameworks like `Node.js` handle concurrency efficiently.

<div id='id-ConcurrencyModels'></div>
#### Concurrency Models
These are strategies for how a program handles multiple tasks or operations at the same time.

1. Threading vs Processes
[This](https://www.geeksforgeeks.org/operating-systems/difference-between-process-and-thread/) article explains this well.
2. Event-Driven Programming
In this model, instead of setting new threads/processes for every connection, the program setup "event listeners". When something happens (like new data arriving on a socket), an "event" is triggered, and a small piece of code (a "callback") is executed. This allows a single thread to manage thousands of connections efficiently by simply reacting to events as they occur.

<div id='id-PerformanceOptimization'></div>
#### Performance Optimization
1. **Caching**: Storing frequently accessed data closer to where it is needed (e.g., in memory, local file system or dedicated caching server like Redis). This reduces the need to refetch data over the network, speeding things up significantly.
2. **Load Balancing**: Distributing traffic across multiple servers in a server farm. This ensures no single server is overwhelmed. This also improves response time and provides high-availability.



<div id='id-SocketProgramming'></div>
### Socket Programming

1. [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)
   A good resource. Skim through it and get an introduction, learn the basics like what sockets are, about TCP/IP connections, how to send/receive data, and open/close a TCP/IP connection. 
    - To get a high-level overview, check out [Bytemonk's video](https://www.youtube.com/watch?v=NvZEZ-mZsuI).
    - [This](https://medium.com/@onix_react/what-are-sockets-and-what-are-sockets-for-8eef56436b7b) is a nice, quick read for understanding the core concept.

2.  [Linux IP Networking](https://www.cs.unh.edu/cnrg/people/gherrin/linux-net.html) - (chapter 2-9)
For those who want to get deep into the Linux side of things, this is a excellent resource. It gets a bit more advanced, but it's incredibly rewarding.

3.[Deep dive into iptables and netfilter architecture](https://www.digitalocean.com/community/tutorials/a-deep-dive-into-iptables-and-netfilter-architecture) 
Understanding `iptables` and `netfilter` is crucial for network security and traffic control on Linux. 

4. [Linux Networking Stack](https://linux-kernel-labs.github.io/refs/heads/master/labs/networking.html#)

<div id='id-SomePracticalAssignments'></div>
### Some Practical Assignments

Here are some project ideas with increasing complexity to solidify your understanding. Pick a language (Python is great for quick prototyping, C++ is great for performance).

1. Simple Chat Application (Command Line)
    - Goal: Build a client-server chat app.
2. File Transfer
    - Goal: Create a program to send and receive files from one computer to another with proper error handling.
3. HTTP Proxy Server
    - Goal: Build a basic HTTP proxy that forwards requests.
4. Websocket-Based Real-Time Application
    - You can brainstorm ideas for this. An example may include a collaborative whiteboard.


<div id='id-FurtherResources'></div>
### Further Resources
1. Building a TCP/IP stack - [Video](https://www.youtube.com/watch?v=KSQNDu8et-s)
2. Some channels and blogs you can follow
    - NetworkChuck
    - ByteMonk
    - LowLevel
    - [Beej's Blog](https://beej.us/blog/)

3. Books
    - Computer Networking: A Top-down Approach by Kurose and Ross (Classic one)
    - For Socket Programming
Unix Network Programming, Volume 1: The Sockets Networking API by W. Richard Stevens.
    - High-Performance Browser Networking by Ilya Grigorik
    - Designing Data-Intensive Applications by Martin Kleppmann


**Contributors**

- Austin Shijo \| +91 8946061070
- Shivansh Jaiswal \| +91 9971104638
- Sujal Satish Montangi \| +91 7349439500
