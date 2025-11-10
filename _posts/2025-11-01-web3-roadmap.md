---
layout: post
title: "Web3 Roadmap"
date: 2025-11-01
author: singu
category: Roadmap
tags:
  - roadmap
  - web3
  - blockchain
  - cryptocurrency
  - bitcoin
  - ethereum
  - iitkpclub
  - filecoin
  - ipfs 
  - rust
categories:
  - roadmap
hidden: true
image:
  url: /images/web3-roadmap/web3fnl.png
---

# Web3 Roadmap

# What is Web3?
So, what even is Web3?

Asking this simple question sends you down a rabbit hole of corporate jargon. You'll find official sounding descriptions of a "new iteration of the World Wide Web," a supposed revolution built on the holy trinity of decentralization, blockchain, and token-based economics.
In short, it’s the internet's next complicated phase, and no one has the elevator pitch quite figured out.
Fear not, brave reader! We at Pclub shall now attempt the impossible: explaining Web3 in a way that doesn't require a Ph.D. in buzzwords. (oh we will use buzzwords!)
![Untitled](https://hackmd.io/_uploads/B192j4zPlg.jpg)


Let us first try to understand the term Web1/Web2/Web3 through a simple graphic before we move on to the "**Great web3 movement**"
![65ba756942b53e27972bef9e_Web1 vs Web2 vs Web3](https://hackmd.io/_uploads/B1db2EMPxg.png)
This particular graphic should make it bit clearer on what exactly this web123 drama is.

### THE WEB3 MOVEMENT

To understand the Web3 movement is to understand its quasi religious fervor. Its proponents speak not merely of a new technology but of a digital promised land, a "brave new world" that will finally "democratise the internet". This is not presented as a simple upgrade, but as a moral and political crusade. It is a righteous transition from the "monarchy" of Web 2.0, where users are subjects under the thumb of corporate kings, to the glorious "democracy" of Web3, where every user is a sovereign citizen.
At the heart of this new gospel is a simple yet profound faith: **"read-write-own"**. The first two commandments, reading and writing, were granted to us in previous eras of the web. But the final, most sacred right **ownership** has been withheld. In the current fallen state of Web 2.0, our data, our content, and our digital identities are not our own. They are assets harvested by tech giants, used to sell us things we don't need and to manipulate our behavior. Web3, the gospel proclaims, is the great emancipation. Through its divine mechanics, users will finally achieve true "ownership and control" over their digital selves, becoming not just consumers of the internet but "shareholders and participants" in its creation and governance.








---
## So Why Web3?
Lets divide the answer into 3 categories

### Censorship
The centralized nature of Web 2.0 creates single points of failure, both technical and political. A platform can be shut down, a user can be deplatformed, and a government can demand the removal of content. Web3 promises a world with "no single point of failure," **a distributed network where no single entity holds the power to silence another.** This architecture is designed to be inherently "**censorship-resistant**," limiting the power of both Big Tech and authoritarian regimes to control the flow of information. It is a vision of a truly free and open public square, invulnerable to the whims of the powerful.

### Privacy
In the world of Web 2.0, privacy is a commodity to be sold. Our every click, search, and "like" is tracked, cataloged, and monetized. Web3 offers a path to digital sanctification through "enhanced privacy". Transactions may be public, but one's true identity can remain "confidential". The ultimate goal is "self-sovereign identity," a state of grace where you, and you alone, control the keys to your digital soul, deciding who can access your information and on what terms. It is the final absolution from the original sin of surveillance capitalism that has plagued the modern internet. 

### Financial Liberation

Perhaps the most potent promise of Web3 is that of collective financial salvation. In this new economy, participation is rewarded. Users can earn digital currency **cryptocurrency** for their online activities, transforming them from passive consumers into active stakeholders. This promise is encapsulated in the community's most fervent prayer, its most hopeful mantra: "WAGMI," an acronym for "We're All Gonna Make It". It is a declaration of faith in a future of shared prosperity, a belief that the rising tide of the **blockchain** will lift all boats, carrying the faithful to a state of financial grace.
The Web3 ecosystem, in its current form, is beset by daunting challenges: transactions can be painfully slow and prohibitively expensive, the user experience is often bewilderingly complex, and the space is rife with scams, hacks, and catastrophic failures.
In this framework, Web3 cannot truly fail; it can only be "too early." Any criticism of its current state is met with the patient, knowing smile of the true believer, secure in the knowledge that salvation is just around the corner.
So, perhaps this is the sign you’ve been waiting for: become a disciple of Web3 and chant WAGMI with the faithful.



#### Okay, enough of the miracles or why  how do we start building the Web3 future???

lets go on and set weekly targets to breakdown this mess and learn!!
![cooked web3](https://hackmd.io/_uploads/HyeuSrfPle.jpg)

## Table of Contents

#### [Week 1 (Foundation)](#id-Week-1-Foundation)
#### [Week 2: Bitcoin](#id-Week-2-Bitcoin)
#### [Week 3: Ethereum and Smart Contracts](#id-Week-3-Ethereum-Smart-Contracts)
#### [Week 4: Solidity](#id-Week-4-Solidity)
#### [Week 5: The Web3 Landscape](#id-Week-5-Web3-Landscape)
#### [Week 6: WebDev](#id-Week-6-WebDev)
#### [Week 7: Dapps](#id-Week-7-Dapps)
#### [Solana Rust Development](#id-Solana-Rust-Development)
#### [Layers of Blockchain](#id-Layers-of-Blockchain)
#### [Extra Resources](#id-Extra-Resources)


<div id='id-Week-1-Foundation'></div>
## Week 1 (Foundation)

### Day 1
Let's understand about the use of cryptography which makes the blockchains work like they work:

#### Cryptographic Hash Functions
Cryptography is blockchain’s bulletproof vest. It uses hardcore math and computer science to craft algorithms that laugh off attacks, making them nearly impossible to crack in the real world, even if perfect security’s just a pipe dream.

##### Cryptographic Hash Function

Picture a cryptographic hash function as a digital shredder that chews up any data be it a tweet, a Bitcoin transaction, or your grade sheet, and spits out a fixed-size string of gibberish called a hash. Think SHA256 (Bitcoin’s fave) or MD5 (old school, but still cool). This isn’t just any shredder; it’s a one way, no take backs machine that keeps blockchain’s trust game airtight.

##### Why They’re the MVP

- Good luck reverse-engineering the original data from a hash. It’s like trying to unscramble an egg. (Preimage resistance, nerds.)
- Finding two different inputs with the same hash? About as likely as winning the lottery while getting struck by lightning. (Second preimage resistance.)
- Change one letter in your input, and the hash flips out, becoming a totally different beast.
- Whether you’re hashing a single emoji or War and Peace, the output’s always the same length. Consistency is key.


#### Symmetric Encryption
Symmetric encryption is like passing a locked diary back and forth with one shared key. Both sender and receiver use the same key to lock (encrypt) and unlock (decrypt) data.

**Key Features:**
- Blazing fast for encrypting big chunks of data, like your crypto wallet’s secrets.
- You gotta sneak that key to your buddy securely, or some hacker’s reading your diary.The Catch: You gotta sneak that key to your buddy securely, or some hacker’s reading your diary.


**Examples:** Data Encryption Standard (DES), Advanced Encryption Standard (AES).

#### Asymmetric Encryption

Asymmetric encryption uses a public key (shout it from the rooftops) and a private key (guard it like your evil secret).

- Encrypt with the public key, decrypt with the private. Even if everyone knows the public key, your secrets stay safe.
- No need to whisper keys in dark alleys. It’s the backbone of blockchain wallets and secure chats.
- Combine it with symmetric encryption for hybrid encryption—use asymmetric to swap symmetric keys securely.
- **Example**: RSA, math so fancy, it’s practically wizardry.




#### Digital Signatures: Blockchain ID Card

Digital signatures are like signing your name in unbreakable ink to prove it’s you and your message didn’t get messed with.

**Typical Digital Signature Process:**
1. The sender creates a hash (digest) of the message.
2. The hash is encrypted with the sender’s private key.
3. The encrypted hash (digital signature) is attached to the message.
4. Anyone can use the sender’s public key to decrypt the signature and verify the hash.

This system guarantees:
- The message truly comes from the sender (authenticity).
- The message was not altered after signing (integrity).

### Day 2:
Now that you understand how cryptographic primitives work, let’s explore how they're applied in blockchains—starting with the core idea of what a blockchain is. A blockchain is a system of recording information in a decentralized way that makes it difficult or impossible to change, hack, or cheat the system.

A blockchain is essentially a digital ledger of transactions that is duplicated and distributed across the entire network of computer systems on the blockchain. To get a better idea of how this is achieved you can refer to the video [What is Blockchain?](https://www.youtube.com/watch?v=SSo_EIwHSd4)

#### The Immutable Scripture
Blockchain is the bedrock of Web3, a digital fortress where trust is baked into the code, not handed out like cheap candy. Imagine a spreadsheet that’s so paranoid, it’s copied across thousands of computers worldwide, updated in real-time. Want to sneak in and fudge the numbers? Good luck you’d need to hack every single copy simultaneously, and even then, the system would laugh in your face.

Here’s the deal: blockchain is a distributed ledger, a record of transactions that’s shared, transparent, and damn near impossible to tamper with. Every transaction whether it’s sending crypto, minting an NFT, or voting in a decentralized poll gets bundled into a block. These blocks are chained together (hence, blockchain) in chronological order, forming an unbreakable timeline of truth. Each block is sealed with a cryptographic lock, and every participant in the network (called nodes) holds an identical copy. Try to mess with one block, and the whole chain screams, “Nice try, buddy!”

#### How It Works (Without Boring You to Death)
- Someone Does Something: Let’s say Alice sends Bob 1 Bitcoin. That transaction gets broadcast to the network.
- Miners or Validators Get to Work: Depending on the blockchain, a group of number crunching nerds (miners in Proof-of-Work systems like Bitcoin) or stake holding gatekeepers (validators in Proof-of-Stake systems like Ethereum 2.0) verify the transaction. They make sure Alice isn’t pulling a fast one, like spending coins she doesn’t have.
- Block Party: Verified transactions are grouped into a block. Think of it as a page in the ledger. This block gets a unique code called a hash, which is like its DNA change one thing, and the hash breaks, alerting everyone.
- Chain It Up: The new block is linked to the previous one using that hash, forming a chain. This is why you can’t go back and edit history,every block is tied to the one before it.
- Everyone Updates: Every node in the network gets the updated blockchain. No central server, no single point of failure, no shady middleman.

#### Types of Blockchain
Understanding the different types of blockchain is essential for selecting the right solution for specific needs. Read more [here.](https://www.geeksforgeeks.org/ethical-hacking/types-of-blockchain/)

For better understanding of blockchains, [read](https://onezero.medium.com/how-does-the-blockchain-work-98c8cd01d2ae). 

### Day 3:
#### Structure of a block
Valid transactions are stored in a temporary pool called the **mempool** (or transaction pool). These transactions wait to be included in a block by a block producer (miner or validator). Blocks hold batches of valid transactions that are hashed and encoded into a Merkle tree (covered later). Each block includes the cryptographic hash of the prior block in the blockchain, linking the two.  A block header may also contain the address of the proposer of the block, i.e., the miner, and the proof of his work, i.e. the solution to a cryptographic puzzle, known as the nonce. The linked blocks form a chain, i.e. a blockchain, similar to a linked list. This iterative process confirms the integrity of the previous block, all the way back to the initial block, which is known as the genesis block. 

#### Bitcoin
In recent times, there has been a lot of buzz about bitcoins and cryptocurrency, with lots of investors investing in it, calling it the future. Let’s go behind the curtains to find out [how does bitcoin actually work?](https://www.youtube.com/watch?v=bBC-nXj3Ng4) 

Checkout the Bitcoin Whitepaper [here](https://bitcoin.org/bitcoin.pdf)
[Here](https://andersbrownworth.com/blockchain/blockchain) is a nice visual demo of [blocks](https://andersbrownworth.com/blockchain/block) in a blockchain.

### Day 4:
Extra Reading to deepen your grasp:
• [Fork blockchain](https://www.geeksforgeeks.org/computer-networks/blockchain-forks/)
• [Merkle tree](https://en.wikipedia.org/wiki/Merkle_tree)

### Day 5-7:
The two most widely known [consensus mechanisms](https://ethereum.org/en/developers/docs/consensus-mechanisms/) are:
• [Proof of Work | ethereum.org](https://ethereum.org/en/developers/docs/consensus-mechanisms/pow/)
• [Proof of Stake | ethereum.org](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/)

Here are some more resources that we think might help you in exploring the topic more. However, don’t limit yourself to this; there is a lot of literature to explore on the internet.
- [Proof of Work vs Proof of Stake:](https://www.geeksforgeeks.org/techtips/difference-between-proof-of-work-pow-and-proof-of-stake-pos-in-blockchain/)
- [51% Attack: Definition, Who Is At Risk, Example, and Cost](https://www.investopedia.com/terms/1/51-attack.asp)
- [Consensus Mechanisms](https://hacken.io/discover/consensus-mechanisms/)

<div id='id-Week-2-Bitcoin'></div>
## Week 2: Bitcoin

Bitcoin is a decentralized ledger using Proof of Work (PoW) to secure transactions. Skim the Bitcoin Whitepaper for Satoshi’s vision and Bitcoin.org for dev basics like addresses, transactions, and blocks.

I believe Bitcoin represents such a fundamental shift in how we think about money and systems that the best way to truly understand it is through hands on experience. Start by using devnet/regnet, send transactions, and track them using blockchain explorers. Experiment with different types of wallets. Over time, dig deeper try using Bitcoin Core through the command line. It’s a gradual process, and it can take years to fully grasp the depth of it all.



### What to learn?
- Cryptographic Hash Functions workings
- Addresses and wallets 
- Transactions
- POW
- Segregated witness and BIPs

### Resources 
- [Learn Me a Bitcoin](https://learnmeabitcoin.com) is hands down one of the best resource to learn about bitcoin development, it provides 2 pathways, Beginner friendly and Technical.
- [Grokking Bitcoin](https://github.com/kallerosenbaum/grokkingbitcoin/blob/master/ch01-introduction-to-bitcoin.adoc) 

### Assignment
Lets do quick assignment (solved) to look a bit deeper in bitcoin development 
### Interacting with a Bitcoin Node
 you'll learn how to use Bitcoin Core's RPC to interact with a running Bitcoin node. The tasks involve connecting to a Bitcoin Core RPC daemon, creating, and broadcasting a transaction. You'll need a Bitcoin node running in regtest mode on your local machine to test our solution.

#### Objective
Successfully send a Payment + OP_Return Transaction.

Your tasks are to:

    Connect to a Bitcoin node in regtest mode using RPC.
    Create and load a wallet named “testwallet.”
    Generate an address from the wallet.
    Mine blocks to that address.
    Send 100 BTC to a provided address.
    Include a second output with an OP_RETURN message: “We are all Satoshi!!”
    Set the fee rate to 21 sats/vB.
    Output the transaction ID (txid) to an out.txt file.

#### Requirements
Input

Create a transaction with the following outputs:

    Output 1:
        Address: bcrt1qq2yshcmzdlznnpxx258xswqlmqcxjs4dssfxt2
        Amount: 100 BTC
    Output 2:
        Data: "We are all Satoshi!!" (This should be an OP_RETURN output with the binary encoding of the string.)

Output
After creating and broadcasting the transaction, save the txid to out.txt.


- Note : You will have to connect to bitcoin node on your local machine and run it in cli , refer to grokking bitcoin for how? and/or follow [bitcoin-cli ](https://youtu.be/9rbKmCZiehk?si=2nkHvQigExb0oNeU)



<div id='id-Week-3-Ethereum-Smart-Contracts'></div>
## Week 3: Ethereum and Smart Contracts

Welcome to Ethereum, the blockchain that’s basically a global supercomputer with a PhD in decentralization. This week, we’re diving into Ethereum’s origin story, its ecofriendly makeover, and smart contracts the self executing code that makes banks cry and dApps fly. Let’s crank up the Web3 vibes and get you ready to rule this programmable playground.
### Day 1:
Smart contracts allow participants to transact with each other without a trusted central authority. A sender must sign transactions and spend Ether, Ethereum’s native cryptocurrency, as a cost of processing transactions on the network.

- Check this now: [Intro to Ethereum](https://ethereum.org/en/developers/docs/intro-to-ethereum)
- here -> [Vitalik Buterin Describing Ethereum ](https://www.youtube.com/watch?v=TDGq4aeevgY)

Ethereum was conceived in 2013 by programmer Vitalik Buterin when he released the Ethereum Whitepaper 
[(Ethereum Whitepaper](https://ethereum.org/en/whitepaper)

In 2014, the development work began and was crowdfunded, and the network went live on 30th July 2015. Ethereum currently runs on the Proof of Stake(PoS) consensus mechanism post “The Merge” which shifted Ethereum from Proof of Work(PoW) to Proof of Stake. Read more about the merge here: 
[The Merge ](https://ethereum.org/en/upgrades/merge/#:~:text=by%20~99.95%25.-,What%20was%20The%20Merge%3F,be%20secured%20using%20staked%20ETH) 
The Merge marked the end of proof-of-work for Ethereum and started the era of a more sustainable, eco-friendly Ethereum. Ethereum's energy consumption dropped by an estimated 99.95%, making Ethereum a green blockchain.

### Day 2:
[Smart contracts](https://www.vationventures.com/glossary/smart-contracts-definition-explanation-and-use-cases) are the fundamental building blocks of [Ethereum applications](https://ethereum.org/en/dapps/). Nick Szabo coined the term “smart contract”. In 1994, he wrote [an introduction to the concept](https://www.fon.hum.uva.nl/rob/Courses/InformationInSpeech/CDROM/Literature/LOTwinterschool2006/szabo.best.vwh.net/smart.contracts.html) and, in 1996, [an exploration of what smart contracts could do.](https://www.fon.hum.uva.nl/rob/Courses/InformationInSpeech/CDROM/Literature/LOTwinterschool2006/szabo.best.vwh.net/smart_contracts_2.html)
Here is an interesting explanation of Smart Contracts by Vitalik Buterin: [Smart Contracts - Vitalik Buterin.](https://www.youtube.com/watch?v=r0S4qIMf4Pg&ab_channel=TheBitcoin%26CryptoPodcastwithJeffKirdeikis)
Anyone can write a smart contract and deploy it to the network. You just need to learn how to code in a [smart contract language](https://ethereum.org/en/developers/docs/smart-contracts/languages/) and have enough ETH to deploy your contract.
#### Contract Accounts: How Smart Contracts Live on Ethereum
Ethereum handles [two types of accounts](https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#accounts):
- Externally Owned Accounts (EOAs): Controlled by private keys, these act like regular user wallets.
- Contract Accounts: Controlled exclusively by smart contract code deployed on the blockchain.

Key properties of contract accounts:

- Like any other Ethereum account, a contract account has its own address on the blockchain.
- A contract account can hold, send, and receive Ether, enabling it to participate in transactions just like a human-controlled user account.
- When someone sends a transaction to a contract account, it triggers the execution of the smart contract’s code. This code can include transferring Ether, modifying stored data, interacting with other contracts, or enforcing complex business logic.
- Smart contracts are not just passive scripts but active, autonomous entities on Ethereum that can manage assets, enforce agreements, and interact with both people and other contracts.

### Day 3:

The Ethereum Virtual Machine (EVM) is the core computational engine of the Ethereum blockchain, serving as a fully isolated and sandboxed environment for executing smart contracts. Code running in the EVM is not allowed to access the external internet, file system, or other operating system processes, which guarantees security and deterministic execution. Even when smart contracts interact with each other, these interactions are tightly controlled to prevent risks and vulnerabilities.

When a smart contract is deployed or called, its code is run not just on a single computer but simultaneously across thousands of distributed nodes worldwide that participate in the Ethereum network. Each node independently executes the contract, and all nodes must reach consensus—that is, they must agree on the result of every computation and transaction. This consensus process ensures the global Ethereum ledger remains synchronized, accurate, and tamper-resistant, even in a decentralized, trustless system.
Read about the Ethereum Virtual Machine:
- [Ethereum Virtual Machine (EVM)](https://ethereum.org/en/developers/docs/evm)
- [Ethereum Virtual Machine (Solidity Documentation)](https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#index-6)


### Day 4:
To operate efficiently and fairly, Ethereum uses the concept of [gas](https://ethereum.org/en/developers/docs/gas). Gas is a unit that measures the computational work required to perform operations within the EVM, such as executing a contract function, storing data, or deploying a new smart contract. Every operation has a fixed gas cost, and users pay for gas in Ether (ETH), Ethereum’s native cryptocurrency.
 [Gas fees](https://etherscan.io/gastracker) serve multiple essential purposes: they reward miners or validators for processing and verifying transactions, encourage efficient code (since more computationally expensive operations cost more), and protect the network from spam or abuse by making attacks financially costly.

Importantly, deploying a new smart contract on Ethereum requires a transaction—and is typically far more expensive in terms of gas than a simple ETH transfer. This is because contract deployment involves storing bytecode and initial data on the blockchain, a resource-intensive operation.

By running smart contracts across many nodes and requiring consensus on every transaction, the EVM and gas model together provide a secure, decentralized, and efficient framework for the execution of smart contracts and decentralized applications on Ethereum.


### Day 5: 

In addition to Ethereum, there are several other blockchain platforms with their own mainnets (live networks) and testnets (development networks). A key example is **Solana**, a high-throughput blockchain known for fast transactions and low fees. Just like Ethereum’s **mainnet** processes real-value transactions, its **Sepolia testnet** is used by developers to test smart contracts without risk. Sepolia is Ethereum’s default proof-of-stake testnet and is ideal for experimenting with contracts and wallet setups during development.

Solana development differs significantly from Ethereum. While Ethereum smart contracts are written in **Solidity**, a language specifically designed for the Ethereum Virtual Machine (EVM), Solana uses **Rust**, a systems-level programming language known for its performance and memory safety. Solidity has a syntax similar to JavaScript and is widely adopted in the developer community. Rust, on the other hand, is more complex but offers high efficiency and flexibility, making it suitable for Solana’s high-performance architecture.

When comparing Ethereum and Solana development, beginners should start by learning Solidity and deploying contracts on Ethereum’s Sepolia testnet. Then, try building and deploying simple programs on Solana using the Rust toolchain. 

### Day 6:

Day 6 focuses on Web3 wallets, which are central to interacting with decentralized applications and managing assets:
Learn more about the types of web3 wallets [here](https://www.quicknode.com/guides/web3-fundamentals-security/basics-to-web3-wallets).
MetaMask is the most popular non-custodial wallet for Ethereum and EVM-compatible chains. Setting up MetaMask involves a few steps:
- Install the [browser extension](https://chromewebstore.google.com/detail/metamask/nkbihfbeogaeaoehlefnkodbefgpgknn) or app.
- Create or import a wallet and safeguard your recovery phrase.
- Enable test networks, such as Sepolia, in settings.
- Copy your wallet address and learn how to access the private key securely (never share it).
With MetaMask, you can receive test ETH, deploy contracts, view your private key and interact freely with Ethereum’s live and test environments.

### Day 7: 

To obtain **test ETH** on the Sepolia network, developers rely on **faucets**—special tools that distribute free test tokens. These tokens are essential for smart contract deployment, testing transactions, and simulating dApp behavior without using real ETH. However, many faucets impose **claim limitations** to prevent abuse and ensure fair access for all developers.

Although Sepolia ETH has no monetary value, unrestricted access would lead to spamming, hoarding, or automation abuse by bots. To maintain reliability and fairness, faucets use several methods to limit token distribution. Some impose [**time-based limits**](https://cloud.google.com/application/web3/faucet/ethereum/sepolia), allowing users to claim once every 12–24 hours. Others check the [**mainnet balance**](https://www.alchemy.com/faucets/ethereum-sepolia) of the requesting wallet and block claims from addresses that don't have a minimum ETH balance—ensuring test ETH is reserved for true experimentation, not hoarding. Additionally, many faucets require users to pass a [**proof-of-work**](https://sepolia-faucet.pk910.de/) or CAPTCHA-like challenge to verify they are human and prevent automated draining by scripts.

These safeguards help maintain a stable and useful test environment, giving all developers fair access to the resources they need to build and test on the Ethereum network without disruption.

Use [*Google Cloud Web3 Ethereum Sepolia Faucet*](https://cloud.google.com/application/web3/faucet/ethereum/sepolia) to claim 0.05 sepolia per day per Google account/ETH address.

<div id='id-Week-4-Solidity'></div>
## Week 4: Solidity
Now, having learnt about what smart contracts are and what they are capable of doing, the next thing that you should know is how do you write a smart contract. We will first discuss writing smart contracts in Solidity.

### Day 1:
It is mentioned that solidity is like an object-oriented language, or OOP for short. 
Well, simply speaking in OOP, everything around us is an object. The notion that everything is an object is the concept that underlies object-oriented programming. These objects contain data, which we also refer to as attributes or properties, and methods. Objects can also interact with each other.

To learn more about the OOP paradigm, you can refer to this [link](https://www.devopsschool.com/blog/object-oriented-programming-oop-concept-simplified/). 

While contracts in Solidity are similar to classes in OOP languages, there are important differences. Contracts are deployed as single instances on the blockchain, while classes can be instantiated as multiple objects in applications.

### Day 2:
Now, we are good to go forward to learn about Solidity. It is influenced by C++, Python, and JavaScript and is designed to work on Ethereum, specifically the Ethereum Virtual Machine (EVM). Solidity is statically typed, and supports inheritance, libraries and complex user-defined types among other features. 
The best way to try out Solidity right now is using Remix. Remix is a web browser-based IDE that allows you to write Solidity smart contracts and then deploy and run the smart contracts.
- Visit https://remix.ethereum.org.
- Create a new Solidity file. 
- Write the contract code and compile using the Solidity compiler.
- Go to the "Deploy and Run Transactions" tab.
- Select "Injected Web3" to use MetaMask or use "Remix VM" for testing.
- Provide constructor arguments if required and click "Deploy".

Moving on to references and tutorials for solidity, [TutorialsPoint](https://www.tutorialspoint.com/solidity/index.htm) is a very nice source for learning the basics of solidity. **You can skip the Environment setup part and follow the remaining basic part for a quick review of the syntax of solidity.**


### Day 3:
Solidity has different types of functions, such as view functions that do not modify the state (any variable of the smart contract outside the function) and pure functions that do not read nor modify the state. These are mainly for security purposes and to ensure that unauthorized access to the state does not happen when it is not needed. Constructors are special functions that run once, when the contract is deployed, to initialize state. The compiled bytecode does not contain it.

[Modifiers](https://www.tutorialspoint.com/solidity/solidity_function_modifiers.htm) are commonly used to add access control, preconditions, or validation checks before a function is executed. 
For example, a modifier can ensure that only the owner of a contract can call certain functions. When applied, the modifier runs its logic first, and then the actual function is executed where the underscore symbol` (_) `is replaced within the modifier's body. 

Read till *Solidity Common Patterns*.

### Day 4:

Go through further topics in the [tutorial](https://www.tutorialspoint.com/solidity/index.htm). (after functions till *error handling*)

To get familiar with the OOP concepts of solidity, read about Contracts (basically like classes), *Inheritance* and *Constructors* in solidity.

### Day 5:

Since Solidity does not support floating point numbers, math in solidity is non-trivial.Developers must strike a balance between gas consumption and accuracy, making optimization essential for real-world use. This is how writing smart contracts involves concepts from number theory and competitive programming. Read more on [math in solidity in this blog series](https://medium.com/coinmonks/math-in-solidity-part-1-numbers-384c8377f26d).

### Day 6-7 :
Now, you are familiar with Solidity and Remix. Here is one assignment you should try out to check if you have completely understood the topics. [Assignment](https://docs.google.com/document/d/1wCvzXhwPgOYUu13LM_OI_w4j--JaMdV-S3ElOATTnb0/edit?tab=t.0)

---
If you prefer to do a step-by-step tutorial and search away on google and docs whenever a new term pops up, then follow this [tutorial from Dapp University](https://www.dappuniversity.com/articles/solidity-tutorial). However, please have a look at the above pages after you are done with it! [Solidity](https://docs.soliditylang.org/en/latest/) is the official solidity documentation if you like to follow official docs.
[CryptoZombies](https://cryptozombies.io/) uses a gamified, step-by-step approach that makes learning Solidity fun and interactive, guiding you from basic concepts to building playable blockchain games while earning NFT certificates and engaging with the community. [Solidity by Example](https://solidity-by-example.org/) provides clear, concise code snippets and explanations on core Solidity topics, making it easy to quickly reference, understand, and experiment with real smart contract code for more hands-on, self-directed practice. There also exists [*PClub’s collection of smart contracts*](https://github.com/pclubiitk/smart-contract-hub.git) with functionalities corresponding to real life applications.

![1689946544639](https://hackmd.io/_uploads/H1ru6ZQwlg.jpg)


<div id='id-Week-5-Web3-Landscape'></div>
## Week 5: The Web3 Landscape
### Day 1:
#### ERC standards
If you’ve ever bought crypto, traded an NFT, or played a blockchain-based game, you’ve already crossed paths with ERC (Ethereum Request for Comments) token standards—even if you didn’t realize it. These standards are basically the "rules of the road" for Ethereum, ensuring everything from cryptocurrencies to NFTs works smoothly and stays compatible across wallets, exchanges, and decentralized apps. The most important standards are [ERC-20](https://ethereum.org/en/developers/docs/standards/tokens/erc-20/) (for interchangeable, fungible tokens like most cryptocurrencies), [ERC-721](https://ethereum.org/en/developers/docs/standards/tokens/erc-721/) (for non-fungible tokens, i.e. digital collectibles and NFTs), and [ERC-1155](https://ethereum.org/en/developers/docs/standards/tokens/erc-1155/) (for managing both fungible and non-fungible tokens in a single contract, especially useful in gaming or asset marketplaces). These standards make it easy for new tokens to plug into the existing Ethereum ecosystem, guaranteeing that developers and users enjoy broad compatibility and reliability. If you want to dig deeper into why these token standards matter (and see more examples), check out this [complete guide to ERC standards](https://www.webopedia.com/crypto/learn/erc-token-standards-complete-guide/), or browse the [official Ethereum developer docs](https://ethereum.org/en/developers/docs/standards/tokens/).

#### OpenZeppelin Contracts Wizard
When you build your own tokens or contracts on Ethereum, it’s crucial to use secure, proven code. This is where [OpenZeppelin Contracts](https://wizard.openzeppelin.com/) comes in: it’s a widely trusted, open-source library of smart contracts that have been rigorously audited for security and best practices. OpenZeppelin’s **Contracts Wizard** is an interactive tool that helps you create customized token contracts by letting you select features—such as minting, burning, and access control—and then instantly generates Solidity code ready for deployment. This saves time, reduces risk, and ensures your smart contracts are built to industry standards.

### Day 2: IPFS, URIs, and Pinata IPFS

#### What is IPFS?
The [**InterPlanetary File System (IPFS)**](https://pinata.cloud/ipfs) is a decentralized, peer-to-peer protocol for storing and sharing files across a global network. Unlike traditional HTTP, which fetches content by its location (URL), [IPFS retrieves data](https://pinata.cloud/blog/file-storage-for-blockchains/) by its content—even if the server goes offline—thanks to distributed storage and **content addressing**. Each file saved on IPFS is assigned a unique [**Content Identifier (CID)**](https://pinata.cloud/blog/how-ipfs-works-in-practice-for-developers/), which acts as a digital fingerprint for that specific content. This approach guarantees immutability and resilience against censorship or single-point failures.

#### Why IPFS Is Important in Web3
- **Decentralization:** Removes reliance on single servers or authorities.
- **Data Integrity:** Every file is verified by its hash (CID); tampering is immediately obvious.
- **Immutability:** Data, once uploaded, cannot be changed or erased silently.
- **Persistence:** As long as one node holds the data, it’s accessible using its CID.

#### URIs in Web3 and IPFS
A **Uniform Resource Identifier (URI)** is a string of characters used to identify a resource on the web. Rather than storing bulky files directly on the blockchain (which is costly), NFTs store a URI—usually pointing to an IPFS CID—which references off-chain metadata or media files (like images, videos, audio). This keeps the blockchain lightweight, cost-effective, and scalable, while maintaining integrity and accessibility of assets.

#### Pinata and IPFS Pinning
While IPFS is decentralized, availability of files can be lost if no nodes choose to “pin” a particular file, meaning content might become hard to find or disappear if not actively hosted. [**Pinata**](https://pinata.cloud/) solves this by acting as a professional IPFS pinning service.

### Day 3: Oracles
Oracles are trusted intermediaries that connect smart contracts to real-world data, enabling decentralized applications to react to external events. Since blockchains are isolated systems that cannot access off-chain information directly, oracles fetch, verify, and securely deliver accurate data -- such as price feeds, weather updates, or event outcomes -- to smart contracts. Chainlink is a leading decentralized oracle network, known for its reliable and tamper-proof solutions that empower many DeFi protocols and blockchain applications.

**Useful Resources:**
- [Official Ethereum Oracles Documentation](https://ethereum.org/en/developers/docs/oracles/)
- [Chainlink Oracle Network](https://chain.link/)
- [Introduction to Blockchain Oracles - Cointelegraph](https://cointelegraph.com/learn/articles/what-is-a-blockchain-oracle-and-how-does-it-work)

### Day 4: NFTS
Non-Fungible Tokens(NFTs) represent ownership of digitally scarce goods such as pieces of art or collectibles. These tokens can be implemented on any smart contract based blockchains. It’s like the owner of the token owns the information stored under the token, because they store this information (metadata), they can be sold and bought just like any other physical collectible. But does that mean there is a single unique NFT of a type? No, you might have seen or bought multiple copies of the same NFTs, it totally depends on the owner of the NFT on deciding the number of copies to exist, like an artwork which has multiple copies around the world. This is sort of a technical mistake. Technically nothing stops you from creating multiple NFTs pointing to the same metadata, NFTs that contain unauthorized copies of some copyright content etc.

[OpenSea](https://opensea.io/) is the world’s largest NFT marketplace, enabling users to create (mint), buy, sell, and explore a vast array of NFTs.
Follow [this tutorial](https://github.com/soos3d/mint-erc721-tokens?tab=readme-ov-file) to mint your own NTF (use Pinata instead of Firebase).

[**Etherscan**](https://etherscan.io/) is a public blockchain explorer for Ethereum. It allows anyone to search and verify NFT transactions, contract metadata, and ownership records. With Etherscan, you can:
- Check Token Transfers: View the complete history of an NFT, including minting, sales, and transfers.
- Read Metadata: See the metadata and associated files or URIs for any token.
- Verify Ownership and Contract Details: Inspect contract source code, ownership, and transaction logs for complete transparency.

Etherscan is a vital tool for verifying the legitimacy and provenance of NFTs, helping buyers and sellers avoid scams or unauthorized duplicates. 
Inspect your freshly minted NTF on [Sepolia Etherscan](https://sepolia.etherscan.io/) and [OpenSea Testnet](https://testnets.opensea.io/).

### Day 5: DeFi
Now let’s move to the most widely used purpose of a blockchain, its Digital Finance or DeFi. Blockchain technology has enabled permissionless networks that can be used by anyone, where built-in economic incentives ensure that network services can be maintained indefinitely without the aid of any individual company or central authority. Isn’t it great? This means that there is no third party lurking around our transaction and we are no longer dependent on them to verify it. But there is a downside to everything; Volatility is one of them and also you have to maintain your own records for tax purposes. Regulations can vary from region to region. NFTs are also used in DeFi, they can be used as collateral while taking a digital loan.
[Here](https://www.investopedia.com/decentralized-finance-defi-5113835#:~:text=Decentralized%20finance%2C%20or%20DeFi%2C%20uses,enables%20the%20development%20of%20applications) is a blog to better understand DeFi.

### Day 6: Zero Knowledge Proofs 

Zero-Knowledge Proofs (ZKPs) are cryptographic techniques that let someone prove the truth of a statement without revealing any underlying information, enabling both privacy and security in verification. ZKPs are foundational for Web3 applications, such as [Polygon ID](https://polygon.technology/blog/introducing-polygon-id-zero-knowledge-own-your-identity-for-web3), which empower users to own and control their digital identities without exposing sensitive data. Learn more on zk-proofs [here](https://www.youtube.com/watch?v=_MYpZQVZdiM)

### Day 7: Stablecoins 
Stablecoins are digital currencies designed to maintain a stable value by being pegged to traditional assets like the US dollar, offering the benefits of cryptocurrency (speed, security, programmability) without the volatility that makes Bitcoin and Ethereum unsuitable for everyday payments. Unlike volatile cryptocurrencies, stablecoins serve as reliable mediums of exchange and stores of value, enabling efficient cross-border payments, DeFi applications, and providing financial access to unbanked populations worldwide.

**Key Types of Stablecoins:**
- **Fiat-backed** (USDT, USDC): Backed by traditional currency reserves
- **Crypto-backed** (DAI): Collateralized by other cryptocurrencies
- **Algorithmic**: Use smart contracts to maintain price stability

**Central Bank Digital Currencies (CBDCs)** represent government-issued digital versions of national currencies, offering similar stability but with state control rather than private issuance. While CBDCs provide regulatory oversight and government backing, stablecoins offer innovation and market-driven solutions, creating healthy competition that drives financial innovation.

**Useful Resources:**
- [What Are Stablecoins?](https://www.investopedia.com/terms/s/stablecoin.asp)
- [Quick overview of stablecoins and their importance - YouTube](https://www.youtube.com/watch?v=vx_JyxuV1DE)
- [Stablecoins vs CBDCs: What's the Difference? - Forbes](https://www.forbes.com/sites/digital-assets/article/stablecoins-vs-cbdcs/) 

<div id='id-Week-6-WebDev'></div>
## Week 6: WebDev
In this week, we will learn about web development which is necessary for developing Decentralized Applications

Also, feel free to take reference from the WebDev Roadmap. [(link)](https://pclub.in/roadmap/2024/06/06/webdev-roadmap/)

In the first 4 days, we will cover the basic building blocks of front-end web development(HTML, CSS,JS).

### Day 1:
[HTML Tutorial (w3schools.com)](https://www.w3schools.com/html/default.asp) (Read till HTML Forms Section)

**Web3 Context:** Understanding HTML forms is crucial for dApps as they handle user inputs for wallet connections, transaction data, and smart contract interactions. Focus on form validation and data handling patterns essential for secure Web3 applications.

### Day 2:
[CSS Tutorial (w3schools.com)](https://www.w3schools.com/css/default.asp)

(No need to go much deep into CSS for now, read till CSS Align)

**Additional Focus for Web3:** Learn CSS Grid and Flexbox for responsive dApp layouts, CSS variables for theming (important for light/dark modes common in Web3 UIs), and basic animations for transaction status indicators.

### Day 3-4:
[JavaScript Tutorial (w3schools.com)](https://www.w3schools.com/js/default.asp) (Read till JS Modules)

**Web3-Specific Enhancements:** Pay special attention to Promises and async/await (essential for blockchain interactions), error handling patterns (crucial for transaction failures), ES6 modules for organizing dApp code, and working with JSON (for smart contract ABIs and API responses).

### Day 5-6:
React is a popular front-end Javascript library developed by Facebook. Here are some resources to learn it.

JavaScript code is traditionally run on a web browser as part of a website, but it can be also executed as a standalone process using Node. [Install Node.js](https://nodejs.org/en/download) for working with React.

**Setting up React with Vite:**
Instead of Create React App, use Vite for faster development and better performance with Web3 libraries:
```bash
npm create vite@latest my-dapp -- --template react
cd my-dapp
npm install
npm run dev
```

**Why Vite for Web3 Development:**
- Lightning-fast Hot Module Replacement for instant updates when developing dApp interfaces
- Optimized bundle sizes for better performance with heavy Web3 libraries like ethers.js
- Native ES modules support that aligns with modern Web3 development practices



**Resources:**
- [Getting Started – React](https://reactjs.org/docs/getting-started.html)
- [Full Modern React Tutorial - YouTube](https://youtube.com/playlist?list=PL4cUxeGkcC9gZD-Tvwfod2gaISzfRiP9d)
- [Vite React Setup Guide](https://www.geeksforgeeks.org/reactjs/how-to-setup-reactjs-with-vite/)
- [React with Vite Tutorial 2025](https://www.youtube.com/watch?v=qe3mrBmeno8)

### Day 7:
Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine. It is a popular choice for developing backends if you are already familiar with Javascript.

Here is a video to get you started:
[Node.js Tutorial for Beginners: Learn Node in 1 Hour](https://youtu.be/TlB_eWDSMt4)

About Rest API:
[RESTful APIs in 100 Seconds (Build an API from Scratch with Node.js Express)](https://youtu.be/-MTSQjw5DrM)

**Web3 Backend Context:** Understanding how REST APIs complement decentralized backends, setting up proxy servers for blockchain RPC calls, and CORS handling for dApp development. Node.js serves as a bridge between traditional web services and blockchain networks in many Web3 applications.


<div id='id-Week-7-Dapps'></div>
## Week 7: Dapps
After learning about the blockchain, the nodes, the consensus, all the major components of this technology and then frontend development, you’ll start to wonder, “What kind of applications can I develop using all this knowledge?” The applications built on top of Blockchain are called Decentralised Applications, or DApps.

### Day 1:
Generally Dapps have a standard frontend built using JavaScript frameworks/libraries like React, Vue, etc. and a Solidity/Rust backend, built on top of the blockchain. Check this: [What is a dApp? Decentralized Application on the Blockchain](https://www.youtube.com/watch?v=F50OrwV6Uk8)

Now moving on, as you guys know how to create Smart Contracts, you need a way to connect your DApp frontend with your local or remote Solidity backend, using anything from HTTP to Websockets. To do so you can choose between two JavaScript Libraries:

1) Web3.js - web3.js is a collection of libraries that allow you to connect with a local or remote Ethereum node using HTTP, Websockets, and other communication protocols directly from your JavaScript Based frontend.
2) Ethers.js - Ethers.js is a lightweight JavaScript library used as an alternative to Web3.js to connect the JavaScript frontend with Smart Contacts.

Check this: [Master Ethers.js for Blockchain](https://www.youtube.com/watch?v=XLkjkw0Y-ok)

### Day 2:
If you don’t like JavaScript we have an alternative Web3.py - A Python library for interacting with Ethereum, inspired by Web3.js, many functions are similar.

[Intro to Web3.py · Ethereum For Python Developers	Dapp University](https://www.dappuniversity.com/articles/web3-py-intro)

### Day 3 :
Follow along [this tutorial ](https://hardhat.org/tutorial/deploying-to-a-live-network) [(or this)](https://docs.openzeppelin.com/learn/) in configuring HardHat, and deploying smart contracts in HardHat locally, and on Sepolia testnet using [Alchemy](https://www.alchemy.com/).


### Day 4-5 :

Take a look at [Solidity's security recommendations](https://solidity.readthedocs.io/en/latest/security-considerations.html), which nicely go over the differences between blockchains and traditional software platforms.

[Consensys' best practices](https://consensysdiligence.github.io/smart-contract-best-practices/) are quite extensive, and include both proven patterns to learn from and known pitfalls to avoid.

The [Ethernaut](https://ethernaut.openzeppelin.com/) web-based game will have you look for subtle vulnerabilities in smart contracts as you advance through levels of increasing difficulty.

The [Hack Solidity Playlist](https://www.youtube.com/playlist?list=PLO5VPQH6OWdWsCgXJT9UuzgbC8SPvTRi5) describes the best practices and vernabilities in Solidity. 

### Day 6-7 : Projects
Having followed this roadmap this far, we would recommend testing your newly developed skill using a self project.

Follow [this tutorial](https://www.youtube.com/watch?v=NxDGHynpA4s) to make your own simple DApp. 

Some of the fascinating ideas to think about are: 
- A Crowdfunding Platform using smart contracts- this would enable a safe way of funding, nowadays, the fundings get mixed or displaced, goes to someone else. Many problems like these would be tackled by this idea.
- Peer to Peer Ridesharing - think an app like Uber developed on blockchains.
- For beginners you can also try a To-Do List app powered by Ethereum smart contract. Here is a [YouTube link](https://www.youtube.com/watch?v=coQ5dg8wM2o) for reference from DApp university.

You can explore more guided projects from [Meta School](https://metaschool.so/), [Build Space Projects](https://github.com/buildspace/buildspace-projects) and
[Quick Node](https://www.quicknode.com/sample-app-library/welcome)



-----
<div id='id-Solana-Rust-Development'></div>
# Solana Rust Development 

You've just wrapped up an incredible dive into web3, exploring blockchains, smart contracts, and decentralized ecosystems. But as the market evolves especially with high throughput chains like Solana dominating DeFi, NFTs, and gaming,the spotlight is firmly on Rust. This systems programming language is the backbone of Solana's on-chain programs (smart contracts), thanks to its speed, memory safety, and concurrency features that prevent common bugs like those in other languages. In a 2025 landscape where Solana's TVL has surged past $10B and developer jobs emphasize Rust proficiency, learning it isn't just a skill it's a career accelerator. By the end of this guide, you'll have a clear path to install Rust, grasp the basics, and build your first Solana program.
This guide provides you with resources and tuts to follow instead of phase wise resource breakdown 

## Rust Foundation Resources
Mastering Rust is crucial for Solana development, as Solana programs are primarily written in Rust. These resources cover Rust’s syntax, ownership, and best practices.

- [The Rust Programming Language Book](https://doc.rust-lang.org/book/): Official Rust book for beginners and intermediate learners, explaining fundamentals like ownership and lifetimes.
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/): Hands-on examples to practice Rust concepts through concise code snippets.
- [Rustlings](https://github.com/rust-lang/rustlings): Small exercises to build familiarity with Rust syntax and features, ideal for beginners.
- [The Rust Reference](https://doc.rust-lang.org/reference/): In-depth reference for Rust’s language features, useful for advanced topics.

## Solana Learning Resources
These resources focus on Solana development with Rust, providing tutorials and videos to understand Solana’s architecture and program creation.

- [Solana Bootcamp YouTube Playlist](https://www.youtube.com/playlist?list=PLilwLeBwGuK6NsYMPP_BlVkeQgff0NwvU): Video series covering Solana development basics, including setup and program writing.
- [Coding & Crypto (Rust on Solana)](https://www.youtube.com/@CodingCrypto/): Practical tutorials on building Solana programs with Rust.
- [SolAndy Soldev YouTube Playlist](https://www.youtube.com/playlist?list=PLmAMfj0qP2wwfnuRJQge2ss4sJxnhIqyt): Beginner-friendly videos with step-by-step Solana development guidance.
- [Solana Developers Portal](https://solana.com/developers): Official hub with guides, documentation, and courses for Solana development.

## Programs to Learn From
Explore these repositories, ordered from easiest to hardest, to study and modify Solana programs written in Rust. They help you apply Rust skills practically.


- [Anchor Example Contracts](https://github.com/tgaye/AnchorExampleContracts/): More complex Anchor programs showcasing real-world use cases.
- [Solana Program Examples](https://github.com/solana-developers/program-examples): Official examples covering various program types, from basic to intermediate.
- [Solana Program Library](https://github.com/solana-labs/solana-program-library): Production-ready programs for studying advanced Rust patterns on Solana.
- [Solana Open-Source List](https://github.com/StockpileLabs/awesome-solana-oss): Curated list of open-source Solana projects for real-world exploration.
- [Solana Programming Resources](https://github.com/SolanaNatives/Solana-Programming-Resources): Community-driven collection of tools and tutorials.

## Community and Further Learning
Connect with the Solana developer community and explore additional resources for continuous learning.

- [Web3 Builders Alliance](https://web3builders.dev/builders): Community for Solana developers to collaborate and share knowledge.
- [Solana Developers Portal](https://solana.com/developers): Comprehensive resource for advanced guides, SDKs, and ecosystem updates.


---
<div id='id-Layers-of-Blockchain'></div>
## Layers of Blockchain

Understanding the **layers of blockchain architecture** is crucial to grasp how blockchain networks achieve scalability, security, decentralization, and functionality. These layers mirror how the internet stacks protocols for complex tasks and allow developers to innovate while preserving the core foundations.

### 1. Layer 0: Network Framework (Interoperability/Infrastructure)
The foundational protocols and infrastructure enabling multiple, heterogeneous blockchains to communicate and interoperate. Polkadot, Cosmos, Avalanche Subnets are some common examples
- **Features:**
  - Handles underlying networking (peer-to-peer protocols).
  - Enables cross-chain communication, shared security, and consensus frameworks.
  - Provides tools for building interconnected, customizable blockchains (parachains in Polkadot, zones in Cosmos).

### 2. Layer 1: The Base Blockchain (Mainnet)
The actual blockchain network protocol and consensus responsible for transactions, asset storage, smart contract execution, and security.examples include Bitcoin, Ethereum, Solana, BNB Smart Chain.
- **Features:**
  - Native tokens (ETH, BTC, SOL, etc.).
  - Transaction finality, block production, validation/mining/staking.
  - Security and decentralization guarantees via consensus protocols (PoW, PoS, etc.).
 
Challenges faced are Scalability and throughput (network congestion, high fees on mainnet during spikes).

### 3. Layer 2: Scalability Solutions (Off-chain/Overlay)
Protocols that sit atop Layer 1 to increase transaction speed, lower costs, and reduce mainnet congestion—without compromising security.
Key examples are Arbitrum, Optimism (Ethereum rollups); Lightning Network (Bitcoin); zkSync, StarkNet (Zero-knowledge rollups, validity proofs).
- **Features:**
  - **Rollups:** Aggregate transactions off-chain, submit compressed proofs/results on-chain.
  - **State Channels/Payment Channels:** Direct, instant off-chain settlements (Lightning, Raiden).
  - Users interact mostly with Layer 2; only summary data and dispute resolution go to Layer 1.
- **Note:** Layer 2 inherits security from Layer 1 for the final state.

### 4. Layer 3: Application Layer (DApps & Protocols)
User-facing decentralized applications, protocols, and interfaces built on top of Layer 1 or Layer 2 solutions.
Uniswap, OpenSea, Aave, games, NFT marketplaces, wallets are some known layer 3 applications
- **Features:**
  - Smart contracts and business logic.
  - Web interfaces, API integrations.
  - Protocol-specific tokenomics, communities, governance.

---

### Visualization of Blockchain Layers

| Layer       | Purpose                        | Examples                                         |
|-------------|-------------------------------|--------------------------------------------------|
| Layer 0     | Network, interoperability      | Polkadot, Cosmos, Avalanche Subnets              |
| Layer 1     | Core blockchain protocol       | Bitcoin, Ethereum, Solana                        |
| Layer 2     | Scalability/off-chain solutions| Arbitrum, Optimism, Lightning, zkSync, StarkNet  |
| Layer 3     | Applications/protocols         | Uniswap, OpenSea, Aave, dApps, wallets           |

---

### Why Layered Architecture Matters

Layered blockchain architecture offers several key advantages: interoperability enables different blockchains to communicate and share value securely; scalability ensures networks can handle millions of users and transactions without degrading performance; development flexibility allows innovation at the application and scalability layers without risking the integrity of the base chain; and specialization lets each layer optimize for specific tasks : such as maximizing security at Layer 1, improving transaction speed at Layer 2, and enhancing user experience at Layer 3.

---

**Recommended Resources:**
- [Ethereum Foundation – Layer 2 rollups](https://ethereum.org/developers/docs/scaling/)
- [Polkadot – Layer 0 Protocol](https://docs.polkadot.com/polkadot-protocol/architecture/polkadot-chain/overview/)
- [Layer 0-3 Overview – medium Blog](https://medium.com/the-crypto-masters-guide-tcmg/blockchain-layers-0-3-55c6b2b8989)

---

**Tip:**  
When developing for blockchain, always consider which layer best matches your goals (e.g., maximizing security vs. optimizing transaction fees/speed) and the trade-offs between decentralization, scalability, and usability.



<div id='id-Extra-Resources'></div>
# EXTRA RESOURCES

some tools and extra stuff I personally used/know



### Tutorials

- [Build Your First Blockchain App Using Ethereum Smart Contracts and Solidity](https://www.youtube.com/watch?v=coQ5dg8wM2o)
- [IBM Blockchain 101: Quick-start guide for developers](https://developer.ibm.com/technologies/blockchain/tutorials/cl-ibm-blockchain-101-quick-start-guide-for-developers-bluemix-trs/)
- [Build Your Own Blockchain: A Python Tutorial](http://ecomunsing.com/build-your-own-blockchain)
- [Learn Blockchains by Building One](https://hackernoon.com/learn-blockchains-by-building-one-117428612f46)
- [Blockchain Developers Essentials](https://www.youtube.com/watch?v=YDSJpIrPmgM&list=PLQeiVDgMaJcU_I5RSQCdUJkFukUXpHhK_)
- [Official Ethereum Dapp Tutorial](https://archive.devcon.org/devcon-1/building-a-dapp-what-are-dapps-and-why-meteor/?tab=YouTube)
- [Ethereum Development Walkthrough](https://hackernoon.com/ethereum-development-walkthrough-part-1-smart-contracts-b3979e6e573e)
- [Full Stack Hello World Voting Ethereum Dapp Tutorial](https://medium.com/@mvmurthy/full-stack-hello-world-voting-ethereum-dapp-tutorial-part-1-40d2d0d807c2)
- [Learning Solidity Tutorials on Youtube](https://www.youtube.com/playlist?list=PL16WqdAj66SCOdL6XIFbke-XQg2GW_Avg)
- [Beginners' Guide to Smart Contracts in Solidity](https://www.youtube.com/watch?v=R_CiemcFKis&list=PLQeiVDgMaJcWnAZLElXKLZhS5a71Sxzw0)
- [Create Your Own Ethereum Blockchain](https://www.youtube.com/watch?v=SKXYYnmjauQ&list=PLQeiVDgMaJcVYH3hH29lgBpxog9S82o6a)
- [Solana Development Course](https://www.youtube.com/playlist?list=PL53JxaGwWUqCr3xm4qvqbgpJ4Xbs4lCs7)



### Development tools

- [Eth Build](http://eth.build) — An Educational Sandbox For Web3
- [Ethereum developer tool-list](https://github.com/ConsenSys/ethereum-developer-tools-list) by Consensys — +100 tools
- [TheGraph](http://thegraph.com) — indexing protocol for querying networks like Ethereum and IPFS
- [Filecoin](https://filecoin.io/) — a descentralized storage network
- [Moralis](http://moralis.io) — web3 development platform, build dApps
- [Alchemy](http://alchemy.com) — build and scale you dApps
- [Dune](https://dune.com/home) — explore, create and share crypto data
- [CREATE3](https://github.com/ZeframLou/create3-factory?utm_source=substack&utm_medium=email#readme) — Deploy contract with same address to all blockchains
- [Cryptohack](https://cryptohack.org) — Learn modern Cryptography 

**Contributors**

- Anany Rai \| +91 9870401439
- Arnab Datta \| +91 900775913


