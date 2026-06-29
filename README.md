## Architecture Overview

Heres a diagram listing the path from a submission, to a transparency label:


[Client Request]
              │
              ▼
      POST /submit_text
     (Content + Timestamp)
              │
              ├───► [ Signal 1: Groq LLM Vibe Check ]
              │                    │
              ▼                    ▼
           Content             Result 1
          Timestamp                │
              │                    │
              ▼                    ▼
      [ Signal 2: Stylometric Heuristics ]
              │                    │
              ▼                    ▼
           Content             Result 2
          Timestamp                │
              │                    │
              ▼                    ▼
      [ Signal 3: Perplexity & Burstiness ]
              │                    │
              ▼                    ▼
           Content             Result 3
          Timestamp                │
              │                    │
              ▼                    ▼
          [ Machine Learning Confidence Scorer ]
                               │
                               ▼
                        Confidence Score
                               │
                               ▼
                  [ Transparency Labeling Engine ]
                               │
                               ▼
                       Detection Label
                               │
                               ▼
                  [ JSON Audit Log Formatter ]
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │ WRITE TO AUDIT LOG:          │
                  │ - Content & Timestamp        │
                  │ - Signals 1, 2, & 3 Results  │
                  │ - Confidence Score           │
                  │ - Transparency Label         │
                  └────────────┬─────────────────┘
                               │
                               ▼
                       [ API Response ]
                  { "status": "processed", ... }


## Detection Signals

In this system, I have 3 detection signals.

1. **LLM based classification:** an llm is prompted through the Groq API to return a score from 0-1 on how likely it is that the provided post is AI-generated.

2. **Stylometric Heuristics:** various measures such as sentence length variance and punctuation variance are taken, and another score from 0-1 is calculated. Since this is more deterministic, its weighed more heavily in the final score. However, posts with intentional repeated phrases or sentence structure might make false positives on this signal.

3. **Perplexity and Burstiness:** We use the formula for perplexity, which is the measure of how unexpected a given token is based on all previous ones to an LLM. We use a distilGPT-2 model to get the probability values of each signal. This is also more deterministic, but might fail on some unimaginatively writen posts. 


## Confidence Scoring

To combine the 3 scores, we take their weighted average. Depending on where it falls between 0.0-1.0, we can give it a label. Here are two different posts:


Text: "
No man is an island,
Entire of itself,
Every man is a piece of the continent,
A part of the main.
If a clod be washed away by the sea,
Europe is the less.
As well as if a promontory were.
As well as if a manor of thy friend’s
Or of thine own were:
Any man’s death diminishes me,
Because I am involved in mankind,
And therefore never send to know for whom the bell tolls;
It tolls for thee."

Score: 0.003, strongly human


Text: "The system is designed to help users complete their daily tasks efficiently. The platform provides a comprehensive set of tools for managing information effectively. The application supports a wide range of features to improve productivity. The service is available to all users who register with a valid email address. The interface is simple and easy to use for people with any level of experience. The data is stored securely and protected with industry-standard encryption protocols. The results are displayed in a clear and organized format for easy review. The process is automated to reduce the time required to complete each task."

Score: 0.863, strongly ai

## Transparency Labels

- 'ai-generated': This label is given when the confidence score is high. The text displayed is: 'Our analysis strongly suggests this content was written by an AI.'

- 'uncertain' : This label is given to confidence scores between 0.4 and 0.6. The text displayed is: 'We couldn't confidently determine whether this content was written by a human or an AI.'

- 'human' : This label is given when the confidence score is low. The text displayed is: 'Our analysis suggests this content was likely written by a human.'

## Rate limiting

The limits I have currently are 10 submissions per minute, and 50 per hour. This way, humans who might have multiple submissions prepared that they want to submit at once can do so, but large attacks will still be rate limited. 

## Known limitations

Sophisticated ai-generated poems are hard to catch, specifically because they score highly on the perplexity signal. Creative word choices might influence the decision, and thats why that signal is weighed the least. However, it is genrally hard to classify some ai-generated poems, as they work around the signals I defined quite well by varying sentence length and using creative word choices.

## Spec Reflection

The spec helped me as a primary tool to prompt my AI Tool, and allowed me to organize my thoughts. One diversion I had from it was that I had orignally decided to use an llm-as-a-judge approach for the perplexity signal, but upon some reasearch, I found that it was easily computable, so that approach would be far faster and more secure. 

## AI Usage

I had Claude Code help me with the implementation and testing of the signals and end to end functionality. For instance, I gave it my signal 1 description, asked it to implement, and then manually wrote some tests for it to run. When I had originally had it implement signal 3, it used my spec exactly, including my llm-as-a-judge approach, however, once I found out about the computation, I updated my planning document and had it redo that section.