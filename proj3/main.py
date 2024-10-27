# step1 : import module
from transformers import pipeline

# step2 : create infernce object(instance)

classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# step3 : prepare data
text = "독감이 유행하며 마스크를 찾는 사람이 늘어났다."

#step 4. inference
result = classifier(text)

#step 5.
print(result)