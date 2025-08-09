# Amazon Product Pricer

## Problem Statement and Goal

E-commerce businesses, particularly those operating on platforms like Amazon, face significant challenges in accurately and efficiently pricing products. Traditional methods relying on manual market research, competitor analysis, and subjective judgment are time-consuming, prone to inaccuracies, and not scalable. This leads to suboptimal pricing decisions, resulting in lost revenue, diminished profit margins, and a hindered competitive advantage, especially for new product launches or in rapidly changing market conditions where historical sales data is scarce or irrelevant. The lack of a data-driven, automated system for pricing based on fundamental product attributes, such as those conveyed in a product's description, creates a critical gap in current e-commerce operational strategies.

This project aims to develop an advanced AI model capable of accurately predicting Amazon product prices directly from their textual descriptions, thereby providing e-commerce businesses with a data-driven solution for optimal pricing, enhanced competitive intelligence, and scalable operational efficiency. This project aims to transform the qualitative art of product valuation into a quantifiable science, leveraging state-of-the-art language models to uncover the inherent value and market positioning embedded within product text.

Find a motivation [here](https://github.com/JoKerDii/amazon-product-price-predictor/blob/main/Background.md), and a tech documentation [here](https://github.com/JoKerDii/amazon-product-price-predictor/blob/main/Documentation.md)

## Leaderboard

|           | Random Guess | Random Forest + Word2Ve Embedding | GPT-4o-mini | Fine-tuned GPT-4o-mini | Llama 3.1 | Fine-tuned Llama 3.1 |
| --------- | ------------ | --------------------------------- | ----------- | ---------------------- | --------- | -------------------- |
| Abs Error | 340          | 99                                | 81          | 100                    | 396       | 47.8                 |
| RMSLE     | 1.72         | 0.89                              | 0.59        | 0.79                   | 1.49      | 0.39                 |
