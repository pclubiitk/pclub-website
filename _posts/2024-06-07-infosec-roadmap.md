---
layout: post
title: "Information Security Roadmap"
date: 2024-06-7 02:00:00 +0530
author: Pclub
website: https://github.com/life-iitk
category: Roadmap
tags:
- roadmap
- infosec
- CTF
categories:
- roadmap
image:
  url: 
---


# Roadmap to Information Security

### What is Information Security?
Information Security is not only about securing information from unauthorized access. Information Security is basically the practice of preventing unauthorized access, use, disclosure, disruption, modification, inspection, recording, or destruction of information. Information can be physical or electronic. Information can be anything like your details or, we can say, your profile on social media, your data in mobile phone, your biometrics, etc. Thus, Information Security spans so many research areas like Cryptography, Mobile Computing, Cyber Forensics, Online Social Media, etc.

### Why learn Information Security?
Information security involves considering available countermeasures or controls stimulated through uncovered vulnerabilities and identifying areas where more work is needed. The need for Information security includes:

- Protecting the functionality of the organization
- Enabling the safe operation of applications
- Protecting the data that the organization collects and uses
- Safeguarding technology assets in organizations


*Note:* In case of any doubts while going through this roadmap, You
can post your query on the Infosec Channel on the discord server of
Programming Club, IIT Kanpur. The roadmap is more inclined towards the
offensive side of infosec.

## Week 1 ( Computer Organisation and Fundamentals)

To dive into the field of Information security, one first needs to
understand the basic fundamentals like how computers, networks and other
things work. To break the rule, you need to know the rule. But, to make
it secure you need to understand the rule.

| Day Number | Resources |
|----------------|-------------|
| Day 1 | Computers only understand 0s and 1s, so first we need to understand [Number system](https://www.rapidtables.com/math/number/Numeral_system.html) ( Binary, Octal, Decimal and Hexadecimal).<br/><br/> Computers are good with 0s and 1s, but we need some other system to represent other symbols. [ASCII encoding](https://www.ascii-code.com/) |
| Day 2 | First you need to understand the device on which you will work. Since, it lets you know how exactly each instruction is executed at micro level.<br/><br/> Cover the complete [Computer Component](https://www.javatpoint.com/computer-components) and [Computer Memory](https://www.javatpoint.com/computer-memory) heading and all its subheadings. |
| Day 3 | Most of us use android in our day to day life. So, it becomes really important to have basic android fundamentals, which you can read [here](https://en.wikipedia.org/wiki/Android_(operating_system)). Must cover the [FEATURES](https://en.wikipedia.org/wiki/Android_(operating_system)#Features) and [DEVELOPMENT](https://en.wikipedia.org/wiki/Android_(operating_system)#Development) sections. |
| Day 4 | Network security is one of the most important aspects to consider when working over the internet, LAN or other method. While there is no network that is immune to attacks, a stable and efficient network security system is essential to protect client data. But, before that we need to know the network fundamentals, which you can cover [here](https://www.geeksforgeeks.org/computer-network-tutorials/).<br/><br/> Must cover BASICS and NETWORK SECURITY AND CRYPTOGRAPHY headings. |
| Day 5 | There are different methodologies used in the field of cybersecurity. To secure a system against a particular method you need to know it. You can read some of them [here](https://www.geeksforgeeks.org/types-of-hacking/). |
| Day 6 | Before we proceed further, we need to know about [operating systems](https://www.techtarget.com/whatis/definition/operating-system-OS) and why to choose [linux over windows](https://www.mygreatlearning.com/blog/linux-vs-windows/).<br/><br/> [Here](https://www.stackscale.com/blog/popular-linux-distributions/), you can get an overview of the popular linux distros available. Now, you will [Install Kali Linux in VirtualBox](https://www.youtube.com/watch?v=irGTD6jmYhc). |
| Day 7 | Building and working with the fundamentals can be tedious sometimes. So, you need tools developed by cyber communities over years to tackle problems easily. You can learn about different [important tools](https://www.geeksforgeeks.org/top-10-kali-linux-tools-for-hacking/) that are used in the security field. |

### TASKS:
1. Develop a Base 12 Number System.
2. Convert your name (ASCII text) to base 2, 8, 10, 12, and 16 number systems.
3. Identify network devices around you and figure out what purpose they serve and how.
4. Install a different OS distro in the VirtualBox, run some services, and test it with the tools installed in the first OS and test it with the tools installed in the first OS.

## Week 2: More on Linux, Python and Bash

Throughout the journey exploring info-security the Linux operating
system will play a very crucial role may it be from initially getting
onto challenges or till the reverse engineering. And thus, it becomes
important to have a good understanding of Linux - the basic commands,
and thus, the Unix filesystem.

And Python & bash are scripting languages and really helpful when it
comes to automating the facilities of an existing system. This will be
very useful, while searching for some critical keywords, in a very big
file, for example.

| Day Number | Resources |
|----------------|-------------|
| Day 1 & 2 | Linux Fundamentals<br/> Basic commands: To access the Linux OS from the terminal, you must know some basic commands. Like given [here](https://media.discordapp.net/attachments/977876748475707444/979451763067338752/8938af4-1.png?width=415&height=605).<br/><br/> Unix Filesystem<br/> When working with Linux, it becomes important to understand the filesystem and the hierarchy tree, this will be useful till RE as well.<br/> Which can be covered from [here](https://homepages.uc.edu/~thomam/Intro_Unix_Text/File_System.html). |
| Day 3 & 4 | OverTheWire: Bandit [[Link](https://overthewire.org/wargames/bandit/) ] <br/><br/> These are interesting level-based challenges that will help you learn useful commands! This will help you understand how powerful and useful the terminal is!<br/><br/> “... You will encounter many situations in which you have no idea what you are supposed to do. Don’t panic! Don’t give up! The purpose of this game is for you to learn the basics. Part of learning the basics is reading a lot of new information …” [It is recommended to read through the instructions and manual pages before getting started!] |
| Day 5 | Python<br/>It will be used in scripting, like for searching some flags, when it will not be possible to do so manually.<br/><br/> Basic Python - [Link](https://www.w3schools.com/python/default.asp)<br/><br/> Python for scripting and automation: [Link](https://learn.microsoft.com/en-us/windows/python/scripting) |
| Day 6 | Bash scripting<br/><br/> Bash scripting is one of the easiest types of scripting to learn, and is best compared to Windows Batch scripting. Bash is very flexible, and has many advanced features that you won’t see in batch scripts.<br/><br/> Learn Bash scripting - see this [article](https://linuxhint.com/3hr_bash_tutorial/) or [check this](https://help.ubuntu.com/community/Beginners/BashScripting). |
| Day 7 | Practising scripting<br/><br/> Here are some examples on python scripting: [Link](https://linuxhint.com/python_scripts_beginners_guide/), you should try solving them on your own.<br/><br/> To practise bash scripting, you can check [this website](https://exercism.org/tracks/bash), it has some good exercises for practice, or [this](https://tryhackme.com/room/bashscripting) also. |


Get familiar with Git fundamentals [<u>Part
1</u>](https://medium.com/programming-club-iit-kanpur/a-guide-to-git-the-fundamentals-2d3db6b4df53),
[<u>Part
2</u>](https://medium.com/programming-club-iit-kanpur/a-guide-to-git-branches-and-merging-ae27f6b72f3b)
,
[<u>article</u>](https://www.freecodecamp.org/news/git-and-github-for-beginners/)

Learn regex - [<u>tutorial</u>](https://regexlearn.com/learn/regex101)
OR
[<u>Video</u>](https://www.youtube.com/playlist?list=PL4cUxeGkcC9g6m_6Sld9Q4jzqdqHd2HiD)

## Week 3: Cyber Security and Web Exploitation**

Now after getting all your fundamentals cleared, let's dive into cyber
security and its applications. This week will mainly focus on learning
web exploitation. After the end of this week, you will be able to
penetrate some loosely built websites and might even find bugs on IITK
websites as well ?!

| Day Number | Resources |
|----------------|-------------|
| Day 1 | Cyber Security doesn’t refer to exploiting systems rather as the name suggests it is taking measures against them.<br/><br/> Understand cyber-security and domains under it:<br/> [Kaspersky](https://www.kaspersky.co.in/resource-center/definitions/what-is-cyber-security) , [IBM](https://www.ibm.com/in-en/topics/cybersecurity)<br/><br/> Some important resources and communities you will find in your cyber security journey -<br/>[HackTheBox](https://www.hackthebox.com/) , [Null Byte](https://null-byte.wonderhowto.com/) , [TryHackMe](https://tryhackme.com/) |
| Day 2 | What is [Ethical Hacking?](https://youtu.be/gK73JLEbDs0?t=14)<br/><br/> What is Penetration Testing?<br/>Also called Pen testing, basically it is a thorough vulnerability check in information systems. <br>[HackTheBox](https://www.hackthebox.com/blog/what-is-penetration-testing)<br/><br/> What is Web Exploitation and what does it cover?<br/> [Open Source For You](https://www.opensourceforu.com/2012/03/cyber-attacks-explained-web-exploitation/)<br/><br/>Now let's start learning languages such as HTML and JS which build up the client side of the web application. On the client side, HTML gives structure to the web application while JS gives logical code of how we can interact with it. <br> Learn **HTML** from - [HTML playlist](https://www.youtube.com/playlist?list=PLr6-GrHUlVf_ZNmuQSXdS197Oyr1L9sPB) (Watch till video 20) |
| Day 3 | Do a crash course in **JS** - [JS crash course](https://www.youtube.com/watch?v=hdI2bqOjy3c)<br/> MDN web doc reference for JS - [JS](https://developer.mozilla.org/en-US/docs/Learn/Getting_started_with_the_web/JavaScript_basics)<br/><br/> Go through this video - [Hacker101 - JavaScript for Hackers (Created by @STÖK)](https://youtu.be/FTeE3OrTNoA?t=120)<br/><br/> Learn **Fetch API** from this crash course - [Fetch API](https://www.youtube.com/watch?v=Oive66jrwBs&t=0s) and from MDN web docs - [Fetch API - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)<br/>Also try to learn the terms that you find new in the MDN web docs.<br><br/> An important tool for web exploitation is the developer tools that almost all browsers provide. Here is a crash course on it - https://www.youtube.com/watch?v=gTVpBbFWry8 |
| Day 4 | SQL is the language for querying data from databases and that is where all information is stored. Knowledge of SQL is important in interfering with these queries - [SQL tutorial](https://sqlzoo.net/wiki/SQL_Tutorial)<br/><br/> **PHP** is a server-side scripting language and a lot of old web applications are still using them. Crash Course on PHP - [PHP Crash Course for beginners  2020](https://youtu.be/6mO1UA1r-6Q)<br/><br/> Refer to PHP docs - [PHP docs](https://www.php.net/manual/en/) |
| Day 5 | Now after spending the last two days solely learning, now let's try to apply our knowledge and that's how you actually learn hacking!<br/> Here you can **apply** your knowledge of JS - [HTS](https://www.hackthissite.org/missions/javascript/)<br/> Go to the JavaScript challenges and try to solve all of them.<br/><br/> **XSS attacks** are a type of injection attack in which a vulnerable website is manipulated so as to send malicious scripts to some other client’s browser - [XSS attacks](https://owasp.org/www-community/attacks/xss/)<br/><br/> Here is a game where you can try XSS attacks yourselves - [XSS game](https://xss-game.appspot.com/)<br/><br/> Another popular attack vector is **SQL injection** where the SQL queries that get data out of databases are interfered to take out data without proper authentication - [SQL injection](https://www.hacksplaining.com/prevention/sql-injection) |
| Day 6 | **OWASP Top-10** lists the top-10 web application vulnerabilities in the present day so this is a must know information for people into web security - [OWASP Top-10](https://owasp.org/www-project-top-ten/)<br/><br/> [The TOP 10 VULNERABILITIES In Web Applications In 2022 OWASP Top 10 Explained](https://youtu.be/xGecBCc3lEk)<br/><br/> BurpSuite is another handy tool for web exploitation. You can intercept and manipulate the requests sent from your browser and many more things! - [BurpSuite](https://linuxhint.com/burpsuite_tutorial_beginners/)<br/> [Burpsuite Basics (FREE Community Edition)](https://youtu.be/G3hpAeoZ4ek)<br/><br/> **DVWA** (Damn Vulnerable Web Application) will let you discover various vulnerabilities and bugs on a MySQL+PHP-based web application.<br/> First of all run the web application in a dockerized environment - [Installing Docker](https://docs.docker.com/engine/install/ubuntu/) + [DVWA docker image](https://hub.docker.com/r/vulnerables/web-dvwa/) [DVWA solutions](https://www.youtube.com/playlist?list=PLHUKi1UlEgOJLPSFZaFKMoexpM6qhOb4Q)<br/><br/> This is not to be done completely in a day but you should go on doing it at your own pace. |
| Day 7 | Now getting better in web exploitation or any other application of cyber security is only through practice. Try solving the **OTW Natas** challenges - [Natas](https://overthewire.org/wargames/natas/) and solving various web-exploitation CTF challenges on picoCTF - [picoCTF](https://play.picoctf.org/practice)<br/><br/> Have a look at [CTF checklists](https://fareedfauzi.gitbook.io/ctf-checklist-for-beginner/web) to help you through these challenges.<br/><br/> Also you may find this liveoverflow playlist really informative - [Web Exploitation](https://www.youtube.com/playlist?list=PLhixgUqwRTjx2BmNF5-GddyqZcizwLLGP)<br/><br/> Whenever stuck remember Google Is Your Best Friend. |





<br/>

## Week 4: Cryptography and Reverse Engineering**

| Day Number | Resources |
|---------|----------------|
| Day 1 | Intro to Cryptography – refer [doc](https://docs.google.com/document/d/1zU-Yp7tQTTbbaXmVg2DIIyFojZfZ2QS4zvW0kdS80M8/edit#) <br/><br/> The main idea behind cryptography is to transform data into form which can only be understood by intended targets. Even if someone interferes in between, the information remains secure <br/><br/> There are many types of **cryptography techniques** some common ones are mentioned here - <ul><li>[Base encoding](https://code.tutsplus.com/tutorials/base-what-a-practical-introduction-to-base-encoding--net-27590)</li><li>[Vigenere cipher](https://www.geeksforgeeks.org/vigenere-cipher/)</li><li>[Caesar cipher](https://www.geeksforgeeks.org/caesar-cipher-in-cryptography/)</li><li>[Morse code](https://www.youtube.com/watch?v=D8tPkb98Fkk)</li><li>[Hashing Functions](https://www.tutorialspoint.com/cryptography/cryptography_hash_functions.htm)</li><li>[Symmetric vs Asymmetric Encryption](https://www.javatpoint.com/symmetric-encryption-vs-asymmetric-encryption): [Video](https://www.youtube.com/watch?v=ERp8420ucGs)</li></ul> |
| Day 2 | Some more **advanced cryptography algorithms** : <ul><li>[Public Key Cryptography](https://www.tutorialspoint.com/cryptography/public_key_encryption.htm)</li><li>[MD-5](https://infosecwriteups.com/breaking-down-md5-algorithm-92803c485d25)</li><li>[RSA](https://bitsdeep.com/posts/attacking-rsa-for-fun-and-ctf-points-part-1/) or [article](https://www.di-mgt.com.au/rsa_alg.html) , [video](https://www.youtube.com/watch?v=LYmb8Adr6Wc)</li><li>[SHA-256](https://infosecwriteups.com/breaking-down-sha-256-algorithm-2ce61d86f7a3)</li><li>(optional) [Post Quantum Cryptography](https://www.youtube.com/watch?v=6qD-T1gjtKw) , [selected algo](https://csrc.nist.gov/projects/post-quantum-cryptography/selected-algorithms-2022)</li></ul> |
| Day 3 | Let’s do some **practice** on cryptography <ul><li>[OTW Krypton](https://overthewire.org/wargames/krypton/), [Picoctf](https://play.picoctf.org/practice?category=2&page=1)</li><li>Still free? try [Cryptopals](https://cryptopals.com/),</li></ul> Some tools - [dcode](https://www.dcode.fr/), [cyberchef](https://gchq.github.io/CyberChef/), [cryptolab](https://manansingh.github.io/Cryptolab-Offline/cryptolab.html), [xortool](https://github.com/hellman/xortool), [John the Ripper](http://www.openwall.com/john/), [Ciphey](https://github.com/ciphey/ciphey) |
| Day 4 | **Reverse-engineering** is the act of dismantling an object to see how it works. Here, we’ll be dismantling the codes and applications. [Reverse Engineering Basics (click)](https://docs.google.com/document/d/1kDLLi0rs76Vhkg7MpJs-LRanbmNUeJy6hsg9BuQFrJc/edit?usp=sharing) <ul><li>What is Reverse Engineering?</li><li>Introduction to assembly - [intro](https://www.youtube.com/watch?v=4gwYkEK0gOk) , [x86 assembly](https://www.youtube.com/watch?v=75gBFiFtAb8)</li><li>[Memory layout](https://aticleworld.com/memory-layout-of-c-program/)</li><li>[Registers](https://www.youtube.com/watch?v=1GfMuBn6ZB0)</li><li>Assembly Instructions</li></ul> |
| Day 5 | Let’s **practise** some assembly commands and get familiar with reverse engineering. <ul><li>[Microcorruption](https://microcorruption.com/map) - exciting RE levels to get started.</li></ul> |
| Day 6 | Ghidra is a software reverse engineering framework that helps in analysing and reversing software binaries, decompile a software binary and study the source code underneath. <ul><li>Installing Ghidra [Link](https://github.com/dannyquist/re/blob/master/ghidra/ghidra-getting-started.md) + [Link](https://htmlpreview.github.io/?https://github.com/NationalSecurityAgency/ghidra/blob/stable/GhidraDocs/InstallationGuide.html)</li><li>Ghidra Getting started: [Video](https://www.youtube.com/watch?v=fTGTnrgjuGA&t=67s) (Linux) OR [Video](https://ghidra-sre.org/GhidraGettingStartedVideo/GhidraGettingStartedVideo.mp4) (Windows) OR [Video](https://www.youtube.com/watch?v=oTD_ki86c9I)</li></ul> This playlist will guide you on Reverse Engineering with Ghidra: [Playlist](https://www.youtube.com/playlist?list=PL_tws4AXg7auglkFo6ZRoWGXnWL0FHAEi). |
| Day 7 | You can watch the playlist till day 7 as well. Some interesting challenges archive - [Flareon challenges](https://github.com/fareedfauzi/Flare-On-Challenges/tree/master/Challenges) |


 If you’re more interested in Reverse Engineering you can try
 [<u>this</u>](https://challenges.re/).

## Week 5: Binary Exploitation, Network Tools and Forensics

In this week we’ll be covering various tools and forensics. Binary
 exploitation and forensic part is quite large. I’ve tried covering a
few of them.

| Day Number | Resources |
|---------|----------------|
| Day 1 | Binary Exploitation - finding a vulnerability in the program and exploiting it to gain control of a shell or modifying the program’s functions. <br/><br/> Really [good series](https://www.youtube.com/playlist?list=PLhixgUqwRTjxglIswKp9mpkfPNfHkzyeN) to understand how to actually perform binary exploitation. The [walkthrough](https://ir0nstone.gitbook.io/notes/types/stack/introduction) cover a lot of things, try doing as much as you can. |
| Day 2 | Try out some [picoctf](https://play.picoctf.org/practice?category=6&page=1) challenges, I’m sure you’ll learn lots of new things. |
| Day 3 | **Nmap** scans the network that a computer is connected to and outputs a list of ports, device names, operating systems, and several other identifiers that help the user understand the details behind their connection status. <br/><br/> This [playlist](https://www.youtube.com/playlist?list=PLBf0hzazHTGM8V_3OEKhvCM9Xah3qDdIx) will guide you on using nmap for scanning networks. <br/><br/> Wireshark is a packet sniffer and analysis tool. It captures network traffic from ethernet, Bluetooth, wireless, etc.., and stores that data for offline analysis. <br/><br/> [Wireshark playlist](https://www.youtube.com/playlist?list=PLBf0hzazHTGPgyxeEj_9LBHiqjtNEjsgt) |
| Day 4 | The above tools helps in analyzing the traffic and these ones are for attacking them. <br/><br/> Aircrack-ng - It is a packet sniffer, WEP and WPA/WPA2 cracker, analyzing tool and a hash capturing tool <br/><br/> Aircrack-ng - [article](https://linuxhint.com/how_to_aircrack_ng/), [video](https://www.youtube.com/watch?v=uKZb3D-PHS0) <br/><br/> Sqlmap - Penetration testing tool that automates the process of detecting and exploiting SQL injection flaws and taking over of database servers. <br/><br/> Sqlmap - [video](https://www.youtube.com/watch?v=nVj8MUKkzQk) |
| Day 5 | Metasploit - World’s most used penetration testing framework used for both penetration testing and development platform for creating security tools and exploits. <br/><br/> Metasploit - [playlist](https://www.youtube.com/playlist?list=PLBf0hzazHTGN31ZPTzBbk70bohTYT7HSm) <br/><br/> BeEF - It utilizes the client side attack vectors to asses the security level of the target environment. Beef hacking involves hooking one or more web browsers and using them to launch command modules to attack the target system within the browser context <br/><br/> BeEF - [video](https://www.youtube.com/watch?v=ZOOkeUnQsjk) |
| Day 6 | Forensics: The art of recovering the digital trail left on a computer. There are plenty of methods to find data which is seemingly deleted, not stored, or worse, covertly recorded. <br/><br/> [Metadata](https://youtu.be/HU_euJyxYB4), often described as data about data, helps in understanding the history of a particular electronic file, including when the file was created, modified and accessed, among other information that can be used to describe the file. <br/><br/> Steganography is a method of hiding secret data, by embedding it into an audio, video, image, text, etc.. - <br/><br/> [Quick walkthrough](https://youtu.be/2e6DyZayvEs) <br/><br/> [Ghiro](https://www.getghiro.org/) – A fully automated tool designed to run forensics analysis over a massive amount of images <br/><br/> [Steghide](https://github.com/StefanoDeVuono/steghide), [sherloq](https://github.com/GuidoBartoli/sherloq) |
| Day 7 | Memory forensics <br/><br/> [volatility](https://www.youtube.com/watch?v=Uk3DEgY5Ue8) – The memory forensic framework <br/><br/> [Rekall](https://github.com/google/rekall) – Memory Forensic Framework <br/><br/> For practise try out some [picoctf](https://play.picoctf.org/practice?category=4&page=1) <br/><br/> Some tools: [Binwalk](https://github.com/ReFirmLabs/binwalk), [autopsy](https://www.autopsy.com/download/), [apktool](https://www.youtube.com/watch?v=K35AkvE8ulY). |

If you’re interested in learning more about Binary Exploitation in deep
[this nightmare](https://guyinatuxedo.github.io/index.html) might be
good walkthrough.

### What’s Next?

Information security is designed to protect the confidentiality,
integrity and availability of computer systems and physical data from
unauthorised access whether with malicious intent or not.

Noone becomes a successful Information Security person overnight, days,
weeks or months; it takes years; it’s a continuous process of learning,
revising and adapting. Emerging technologies and cyber-threats will
continue to evolve. Data breaches and security incidents will happen.
Rather than putting a full stop, one needs to follow up with the
emerging methods and technologies.

Connect with the infosec community, build your team, participate in
events, ctfs, hackathons, bug bounties; learn about new vulnerabilities,
read research papers about security and most important practice what you
learn in a safe environment, causing no data loss or system failure.

**Contributors -**

- Harshit Patel 6306342981
- Krishnansh 8317084914
- Nikhil Meena 7791037827
- Pradeep Chahal 9053466181
- Shivam Mishra 8604397668



