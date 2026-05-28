# Xiaomi AIoT Product Intelligence and Customer Voice Analytics

## Project Overview

This project develops a Streamlit-based deep learning application for Xiaomi customer service staff. The application processes one English consumer review at a time and automatically predicts product category and customer sentiment.

## Business Use Case

The application supports early customer review triage:

- Negative reviews are routed to the relevant proposed product team for priority investigation.
- Neutral reviews are routed for further review and monitoring.
- Positive reviews require no immediate departmental follow-up.

## Application Workflow

Customer Review Text  
→ Fine-tuned Product Category Classification Model  
→ Fine-tuned Sentiment Analysis Model  
→ Customer Service Routing Decision

## Product Categories

- Electronics
- Home
- Home Entertainment
- Office Products
- PC
- Watches

## Fine-tuned Models

| Task | Hugging Face Model ID | Final Test Accuracy |
|---|---|---:|
| Product Category Classification | hanlongyang/amazon-category-distilbart-finetuned | 83.50% |
| Sentiment Analysis | hanlongyang/amazon-sentiment-bertweet-finetuned | 78.83% |

## Decision Rule

| Sentiment | Action |
|---|---|
| Negative | Route to relevant product team for priority investigation |
| Neutral | Route to relevant product team for further review |
| Positive | No immediate departmental follow-up required |

## Data Limitation

The application is designed as a prototype for Xiaomi AIoT customer voice analytics. However, the current models were trained and evaluated using labelled English Amazon product review data. Future development can improve domain alignment by fine-tuning models with Xiaomi-specific customer review data.

## Deployment

This application is deployed using Streamlit Community Cloud and loads two fine-tuned Transformer models from Hugging Face.
