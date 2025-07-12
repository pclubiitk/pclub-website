---
layout: post
title: "ScyllaAgent: Scalable and Low Latency Agentic Chatbot"
date: 2025-05-26 19:30:00 +0530
author: Harshit Jaiswal, Himanshu Sharma
category: Project
tags:
- summer25
- project
categories:
- project
hidden: true
summary:
- Deploy a Scalable and Low-Latency Agentic Chatbot  in python using cutting edge techniques like Cache Augmented Retrieval. Implement a professional MLOps Pipeline to ensure visibility, scalability, low latency, logging and continuous feature integration and deployment.

image:
  url: "https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fcda0u9fpwl9ahstphoux.png"
---

# About the Project
Deploy a Scalable and Low-Latency Agentic Chatbot  in python using cutting edge techniques like Cache Augmented Retrieval. Implement a professional MLOps Pipeline to ensure visibility, scalability, low latency, logging and continuous feature integration and deployment.

# Resources
## Week 0
1. [Understand Regression and Grad Descent](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week2-Day5)
2. [Neural Networks](https://pclub.in/roadmap/2024/06/06/ml-roadmap/#id-Week5-Day4)
3. Coding an NN in PyTorch, and a [micrograd framework from scratch](https://www.youtube.com/watch?v=VMj-3S1tku0&ab_channel=AndrejKarpathy)

## Week 1 | Introduction to NLP and Sequence Models

### Basic NLP
1. [NLP Playlist by Tensorflow](https://www.youtube.com/watch?v=fNxaJsNG3-s&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S&index=1&ab_channel=TensorFlow)
2. Side by side, refer to this [github repo](https://colab.research.google.com/drive/124v1SEUCMoDvcY9m8e0HTIZpNdLcW4hP?usp=sharing)
3. [Text Pre processing](https://ayselaydin.medium.com/1-text-preprocessing-techniques-for-nlp-37544483c007)
4. [Text Normalization](https://towardsdatascience.com/text-normalization-7ecc8e084e31/)
5. [Bag of words representation](https://ayselaydin.medium.com/4-bag-of-words-model-in-nlp-434cb38cdd1b)
6. [Term Frequency-Inverse Document Frequency](https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%20-%20Inverse%20Document%20Frequency%20(TF-IDF)%20is,%2C%20relative%20to%20a%20corpus)
7. [Continuous Bag of Words](https://www.geeksforgeeks.org/continuous-bag-of-words-cbow-in-nlp/)
8. [One Hot Encodings](https://www.geeksforgeeks.org/ml-one-hot-encoding/,)

### Sequence Models 
1. [Recurrent Neural Networks](https://youtu.be/AsNTP8Kwu80?si=dxOteYtI0fHo-5hc)
2. [Mathematics of RNNs](https://youtu.be/6niqTuYFZLQ?si=oTd72m_YQX7MB7QL)
3. [Long Short Term Memory](https://youtu.be/YCzL96nL7j0?si=g0ZloJNLZjQb1cua)
4. overall idea of lstms and rnns with a little maths - [video link](https://youtu.be/_h66BW-xNgk?si=tr7pxAerfEkP2Gnc)
5. [Introduction to Transformers](https://youtu.be/wjZofJX0v4M?si=tBerpgJ9omK7DP7Z)
6. [Attention in Transformers](https://youtu.be/eMlx5fFNoYc?si=13QywkWbEWzjWjyw)

## Week 1 Additional Resources
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
4. [Fine Tuning Repository](https://github.com/HARSHITJAIS14/NLP_INTER_IIT/tree/main/FIne_tuning)

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

## Week 2 | Retrieval Augmented Generation

### APIs, LLMs and HuggingFace
1. Requests and APIs:
  1. [APIs](https://www.postman.com/what-is-an-api/)
  2. [REST API](https://blog.postman.com/rest-api-examples/)
  3. [Requests](https://realpython.com/python-requests/)
2. Accessing LLMs via APIs:
   1. [Tokens](https://blogs.nvidia.com/blog/ai-tokens-explained/)
   2. [Tokens and Pricing](https://www.youtube.com/watch?v=ZUCVRppXPSc)
   3. [LLM APIs](https://medium.com/@springs_apps/large-language-model-llm-api-full-guide-2024-02ec9b6948f0)
3. Explore models on Huggingface, resources are as follows:
    1. [https://huggingface.co/docs/transformers/en/llm_tutorial](https://huggingface.co/docs/transformers/en/llm_tutorial)
    2. [https://www.analyticsvidhya.com/blog/2023/12/large-language-models-on-hugging-face/](https://www.analyticsvidhya.com/blog/2023/12/large-language-models-on-hugging-face/)
    3. [https://huggingface.co/models?other=LLM](https://huggingface.co/models?other=LLM)

### RAGs 
1. LangChain Ecosystem
   1. [LangChain Crash Course](https://www.youtube.com/watch?v=Cyv-dgv80kE&t=264s)
   2. LangSmith
      1. [Introduction](https://www.datacamp.com/tutorial/introduction-to-langsmith)
      2. [Docs and Getting Started](https://docs.smith.langchain.com/)
   3. [LangGraph Crash Course](https://youtu.be/jGg_1h0qzaM?si=5EVVzwKc--mHWZYD)
2. Prompting
   1. [12 Prompting Techniques](https://www.promptingguide.ai/techniques)
   2. [Prompt engineering by HugginFace](https://huggingface.co/docs/transformers/en/tasks/prompting)
3. RAGs
   1. [RAG for Knowledge-Intensive NLP Tasks Paper](https://arxiv.org/pdf/2005.11401)
   2. [LangChain Implementation](https://medium.com/data-science/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)


## Week 3 | Agentic Systems
1. Data Validation and Typing
   1. [Typing module](https://medium.com/@moraneus/exploring-the-power-of-pythons-typing-library-ff32cec44981)
   2. [Typing Docs](https://docs.python.org/3/library/typing.html)
   3. [Intro to Pydantic](https://www.youtube.com/watch?v=XIdQ6gO3Anc&ab_channel=pixegami)
   4. [Pydantic in Detail](https://www.youtube.com/watch?v=7aBRk_JP-qY&ab_channel=CodingCrashCourses)
   5. [Pydantic Docs](https://docs.pydantic.dev/latest/)
2. Concurrency
   1. [Concurrency, Parallelism, and asyncio](https://testdriven.io/blog/concurrency-parallelism-asyncio/)
   2. [Repo](https://github.com/based-jace/concurrency-parallelism-and-asyncio/tree/master) for codes in (2.1)
   3. [Async Programming in python](https://betterstack.com/community/guides/scaling-python/python-async-programming)
   4. [A nice video tutorial](https://www.youtube.com/watch?v=Qb9s3UiMSTA&ab_channel=TechWithTim) (alternative to 2.3)
   5. [Exception Handling in python asyncio](https://betterstack.com/community/guides/scaling-python/python-async-programming/)
   6. Nice clarificaton from [stackoverflow](https://stackoverflow.com/questions/41204129/what-part-of-asyncio-is-concurrent-would-like-implementation-details)
   7. [Parallelism v/s Concurrency](https://www.youtube.com/watch?v=jfgQxS4WDxg&ab_channel=TheCodingGopher)
3. Design Patterns
   1. Abstract Factory and Abstract Base Classes
      1. [Refactoring Guru Article](https://refactoring.guru/design-patterns/abstract-factory)
      2. [Abstract Factory Python Implementation](https://refactoring.guru/design-patterns/abstract-factory/python/example#example-0) 
      3. [Intro to ABC]( https://geekpython.in/abc-in-python)
   2. Factory Method, Composite Patterns, Decorators, State, Iterators etc from [Refactoring Guru Design Patterns](https://refactoring.guru/design-patterns/python)
   3. [Grokking OOPs](https://github.com/tssovi/grokking-the-object-oriented-design-interview)
4. [LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/)
5. [Tools, Agents, Agentic Orchestration]
4. [Llamaindex Workflows]


## Week 4 | Agentic and Advanced RAGs

### Mini Advanced RAGs Roadmap
1. Ingestion
	1. Data Preprocessing/Cleaning
	2. [Chunking](https://www.pinecone.io/learn/chunking-strategies/)
		1. Fixed Size Chunking
		2. Content-aware Chunking
			1. Simple Sentence and Paragraph splitting
			2. **Recursive Character Level Chunking**
		3. Document structure-based chunking
		4. [Semantic Chunking](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
		5. Contextual Retrieval: Provides scalability for larger documents
			1. [Contextual BM25](https://www.anthropic.com/news/contextual-retrieval)
			2. [Chunk + General Doc Summary](https://aclanthology.org/W02-0405.pdf)
			3. [HyDE](https://arxiv.org/abs/2212.10496)
			4. [Summary Based Indexing](https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec)
	3. Embedding:
		1. Semantic Embeddings
		2. Lexical Embeddings
			1. BM-25 (Best Matching 25): Lexical Matching which builds upon TF-IDF (Term Frequency-Inverse Document Frequency)
2. Retrieval
	1. Search
		1. Semantic Search (dense vectors)
		2. Lexical Search (sparse vectors)
		3. Hybrid Search
			1. Querying Hybrid Index
			2. Querying Sparse and Dense Index and reranking
	2. [Reranking](https://www.pinecone.io/learn/refine-with-rerank/): Increases quality of retrieved documents
		1. [BGE Reranker](https://docs.pinecone.io/models/bge-reranker-v2-m3) 
		2. [Passage Reranking with BERT](https://arxiv.org/pdf/1901.04085)
3. Augmentation
4. Generation
5. [Evaluation](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
	1. [Offline Metrics](https://www.pinecone.io/learn/offline-evaluation/)
		1. Binary Relevance Metrics
			1. Order-unaware: 
				1. Precision@k: TP/(TP+FP)
					how many items in the result set are relevant
				2. Recall@k: TP/(TP+FN)
					how many relevant results your retrieval step returns from all existing relevant results for the query
				3. F1@k: (2 * Precision@k * Recall@k)/(Precision@k + Recall@k)
			2. Order-aware: 
				1. Mean Reciprocal Rank (MRR)
				2. Mean Average Precision@K (MAP@K)
		2. Graded Relevance Metrics
			1. Discounted Cumulative Gain (DCG@k)
			2. Normalized Discounted Cumulative Gain (DCG@k)
	2. Online Metrics: Based on user data, RL-based
	3. Frameworks and Tooling
		1. Arize
		2. ARES
		3. RAGAS
		4. TraceLoop
		5. TruLens
		6. Galileo
6. [Benchmarking AI Assistants](https://www.pinecone.io/blog/ai-assistant-quality/)

[Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main)

### Additional Resources
1. [Advanced RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)
2. [RAG Optimizations implementations in LangChain](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3954s)


## Week 5 and 6| Creating a Python Module for Advanced RAGs

### System Design and Patterns Review
1. [Python Design Patterns](https://www.youtube.com/watch?v=tAuRQs_d9F8&ab_channel=NeetCode)
2. [Design Patterns](https://refactoring.guru/design-patterns)
3. [ABCs](https://www.youtube.com/watch?v=kaZceE16m5A&ab_channel=k0nze)
4. [ABCs v/s protocols](https://www.youtube.com/watch?v=dryNwWvSd4M&ab_channel=ArjanCodes)
5. [Sequence Diagrams](https://github.com/tssovi/grokking-the-object-oriented-design-interview/blob/master/object-oriented-design-and-uml/sequence-diagram.md)
6. [Activity Diagrams](https://github.com/tssovi/grokking-the-object-oriented-design-interview/blob/master/object-oriented-design-and-uml/activity-diagrams.md)
7. [Some case studies](https://github.com/tssovi/grokking-the-object-oriented-design-interview/tree/master/object-oriented-design-case-studies)

### Project Management
1. [Structuring Python Projects](https://www.youtube.com/watch?v=Lr1koR-YkMw&ab_channel=InfoWorld)
2. [Poetry](https://www.youtube.com/watch?v=Ji2XDxmXSOM&ab_channel=ArjanCodes)
3. [Building Python Packages](https://www.youtube.com/watch?v=5KEObONUkik&ab_channel=ArjanCodes)

### Repos of similar modules for reference
1. FlashRAG: https://github.com/RUC-NLPIR/FlashRAG
2. RAGligh: https://github.com/Bessouat40/RAGLight

### Building and structuring modules
1. [How to design modular python projects](https://labex.io/tutorials/python-how-to-design-modular-python-projects-420186)
2. [Designing Modules](https://hashedin.com/blog/designing-modules-in-python-ebook/)
3. [ Why the Plugin Architecture Gives You CRAZY Flexibility ](https://www.youtube.com/watch?v=iCE1bDoit9Q&ab_channel=ArjanCodes)
4. [Clean Architectures in Python](https://www.youtube.com/watch?v=C7MRkqP5NRI&ab_channel=EuroPythonConference)
5. [Why Use Design Patterns When Python Has Functions](https://www.youtube.com/watch?v=vzTrLpxPF54&ab_channel=ArjanCodes)

### Software testing:
1. [Unit, Integration and Functional Testing](https://www.headspin.io/blog/unit-integration-and-functional-testing-4-main-points-of-difference)
2. [Types of Software Testing](https://www.atlassian.com/continuous-delivery/software-testing/types-of-software-testing)
3. [Unit v/s Integration Testing](https://circleci.com/blog/unit-testing-vs-integration-testing/)
4. [Unit v/s Integration Testing](https://www.practitest.com/resource-center/article/unit-test-vs-integration-test/)

## Week 7 | Reading and Implementing Research Papers
1. [Cache Augmented Generation](https://arxiv.org/pdf/2412.15605v1)
2. [MetaGPT](https://arxiv.org/pdf/2308.00352)
3. [MedRAG](https://arxiv.org/pdf/2502.04413)
4. [Trading Agents](https://arxiv.org/pdf/2412.20138)
5. [Knowledge Augmented Generation](https://arxiv.org/pdf/2409.13731)

## Week 8 | AI Dev