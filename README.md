# sentence_embedding
词向量组成句子向量的逻辑 参考论文a simple but tough-to-beat baseline for sentence embeddings

用于计算论文A Simple but Tough-to-Beat Baseline for Sentence Embeddings
参考代码： https://github.com/PrincetonML/SIF
主要思路:
计算sentence的embedding，不再使用沙雕的相加的方式，而是带有权重的加法，使用怎么样的权重，请参考论文
测试效果不是特别的好，可能因为测试样本太少的原因
代码 sentence_embedding.py

另外测试使用词向量加权系数的思路测试，类似news_embedding的思路
