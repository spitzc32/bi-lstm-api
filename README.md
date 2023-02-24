# bi-lstm-api
Deployed app on Huggingface for thesis 2 model, entitiled BiLSTM-SPAN-CRF

## Statement of the Problem
The study aims to solve the problem of the recurring error in the developments of clinical NER in identifying and categorizing long combined text such as LOCATION and ORGANIZATION. The researchers will solve the problem by incorporating RoBERTa and bi-directional pre-trained FLAIR as transformer word embeddings and context-based embeddings respectively and ensemble learning, specifically average ensemble in combining the results of each input embedding. The researchers would also introduce a novel approach of initially performing BIO tagging using Bi-LSTM-CRF model followed by relation extraction using Span classifier thereby combining the capabilities of BIO tagging and Span-based NER.

## Research Objectives/Questions
At the end of the study, the researchers aim to achieve the following research objectives:
How effective is the proposed model for the task of de-identification?
In terms of it’s Precision, Recall, and F1 measures
Based on Micro-F strict category
How accurate the model would be in identifying entities that the previous researches had a difficulty with?
Evaluate the Precision, Recall, and F1 measures of each category
Conduct an error analysis for each of the categories by creating a confusion matrix.
How effective are ensemble learning and span-based classifiers in improving the BI-LSTM-CRF de-identification model?
Compare to baseline papers (Syed et al., 2022; Tang et al., 2020)


