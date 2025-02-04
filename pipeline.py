from transformers import pipeline
import pandas as pd

text = """Dear Amazon, last week I ordered an Optimus Prime action figure from your
online store in Germany. Unfortunately, when i opened the package, I discovered to
my horror that i had been sent an action figure of Magatron istead! As a lifelong
enemy of the Decepticons, I hope you can understand my dillemma. To resolve the
issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered.
EnClosed are copies of my recored concerning this purchase. I expect to hear from
you soo. Sincerely, Bumblebee."""

ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
print(pd.DataFrame(outputs))
