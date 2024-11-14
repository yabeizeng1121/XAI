# Introduction to Explainable AI (XAI)

Explainable Artificial Intelligence (XAI) refers to methods and techniques in the application of artificial intelligence technology such that the results of the solution can be understood by human experts. It contrasts with the concept of the "black box" in machine learning where even their designers cannot explain why the AI arrived at a specific decision.

## What is SHAP?

SHAP (SHapley Additive exPlanations) is a game theory approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.

## Tutorial: Explainable AI with DistilBERT and SHAP

In this tutorial, we will apply SHAP to explain the predictions of a text classification model trained with DistilBERT. We'll use the IMDB dataset to analyze sentiment classification.

### Setup

- Install the necessary packages:

```bash
pip install transformers torch shap numpy scipy datasets
```

- Import Libraries

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap
import numpy as np
import scipy as sp
from datasets import load_dataset
```

### Understanding DistilBERT and the Need for Model Explanation

DistilBERT is a smaller, faster, and lighter version of BERT, a state-of-the-art transformer model known for its exceptional performance in natural language processing tasks. DistilBERT retains most of the predictive power of BERT but with fewer parameters, making it more suitable for practical applications where resource efficiency is important.

DistilBERT has been fine-tuned on the SST-2 English corpus for sentiment analysis, a common use case in NLP where the goal is to determine the sentiment expressed in a piece of text as positive or negative. Despite its effectiveness, like many deep learning models, DistilBERT acts as a "black box" where the rationale behind its predictions is not immediately obvious. This opacity can be a barrier in critical applications where understanding the reasoning behind decisions is crucial.

By applying explainable AI techniques such as SHAP, we can demystify the decision-making process of models like DistilBERT, providing insights into which words or phrases most influence its predictions. This transparency is vital for trust and accountability in AI deployments, particularly in sensitive areas like healthcare, finance, and public policy.

### Initialize Model and Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model.eval()
```

### Define Prediction Function

To make predictions using the DistilBERT model, we first need a function that can take text inputs, process them through the tokenizer, and feed them into the model to get sentiment predictions. This function should handle preprocessing, including tokenization and padding, and then compute the logit scores from the output probabilities. These scores represent the confidence levels of predictions and are essential for computing SHAP values.

```python
def f(texts):
    tv = torch.tensor([tokenizer.encode(v, padding="max_length", max_length=500, truncation=True) for v in texts]).to(device)
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    return sp.special.logit(scores[:, 1])
```

### Load Data and Explain Predictions

Now that we have our prediction function defined, the next step is to load a dataset and apply our model to generate explanations. The SHAP explainer will use the prediction function to understand how each input token in our dataset influences the model's prediction. This process helps in visualizing the contribution of each word or token towards the final decision made by the model. In this example, I used `shap_values[3]` as an example.

```python
imdb_train = load_dataset("imdb", split='train')
explainer = shap.Explainer(f, tokenizer)
shap_values = explainer(imdb_train[:10]['text'], fixed_context=1)
shap.plots.text(shap_values[3])
```

### Visualization of SHAP Values

![SHAP Output](https://raw.githubusercontent.com/yabeizeng1121/XAI/main/Assignment10/plot.png "SHAP Text Plot Visualization")


The SHAP text plot visualizes how individual words in a text influence a model's prediction. Words that impact the prediction positively are shown in red, while those with a negative impact are shown in blue. For instance, positive words like "lovable", "impressive", and "still" enhance the sentiment towards favorable, whereas negative words like "not" can diminish the effect of nearby positive words.

This plot also shows a base value, which is the model’s average output over a background dataset, and illustrates how each word's contribution shifts the prediction from this baseline to the final output. The color intensity and the length of each block around the words reflect the magnitude of their impact, providing a clear visual representation of their significance in the decision-making process of the model.


## Conclusion

Throughout this tutorial, we've utilized SHAP, a powerful tool within the field of Explainable AI (XAI), to demystify the inner workings of the DistilBERT model. By applying XAI principles, we provided a window into how individual words influence the sentiment predictions in text data. This visibility is crucial in fields requiring transparency and accountability, allowing stakeholders to trust and validate AI decisions. Moreover, the insights gained through such explanations not only foster ethical AI deployment but also support the refinement and understanding of AI models, making them more accessible and interpretable to a broader audience.


