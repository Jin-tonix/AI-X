from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, PreTrainedTokenizerFast
import torch

# 감정 분석 모델 초기화 (KOBERT 사용)
emotion_model_name = "skt/kobert-base-v1"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# KoBART 모델 초기화
kobart_model_name = "gogamza/kobart-base-v2"
kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(kobart_model_name)
kobart_model = BartForConditionalGeneration.from_pretrained(kobart_model_name)

# 감정 분석 함수
def analyze_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])  # token_type_ids를 0으로 설정
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        emotions = {0: "슬픔", 1: "중립", 2: "기쁨"}
        return emotions.get(predicted_class, "알 수 없음")

# 피드백 후처리 함수
def refine_feedback(feedback):
    # 불필요한 반복을 제거하고 문장을 자연스럽게 이어주는 작업
    sentences = feedback.split(". ")
    refined_sentences = []
    
    for sentence in sentences:
        if sentence and sentence not in refined_sentences:  # 중복 문장 제거
            refined_sentences.append(sentence.strip())
    
    # 문장을 마침표로 결합하여 자연스럽게 출력
    refined_feedback = ". ".join(refined_sentences).strip()
    if not refined_feedback.endswith("."):
        refined_feedback += "."
    
    return refined_feedback

# 피드백 생성 함수
def generate_detailed_feedback(emotion, description):
    if emotion == "기쁨":
        prompt = (
            f"그림 묘사: {description}\n"
            f"이 그림은 밝고 긍정적인 감정을 담고 있습니다. "
            "이 감정을 더 풍부하게 표현하고, 주변을 더욱 밝게 할 수 있는 방법을 제안해 주세요."
        )
    elif emotion == "슬픔":
        prompt = (
            f"그림 묘사: {description}\n"
            f"이 그림은 외롭고 슬픈 감정을 담고 있습니다. "
            "슬픔을 표현함으로써 위로를 받을 수 있도록 도와주는 조언을 제시해 주세요."
        )
    else:
        prompt = (
            f"그림 묘사: {description}\n"
            f"이 그림은 중립적이거나 복잡한 감정을 담고 있습니다. "
            "다양한 감정을 표현하며 자신을 탐색하는 데 도움이 되는 구체적이고 감정적인 피드백을 제공해 주세요."
        )

    inputs = kobart_tokenizer(prompt, return_tensors="pt")
    output = kobart_model.generate(
        inputs["input_ids"],
        max_length=150,
        temperature=0.85,
        top_p=0.92,
        no_repeat_ngram_size=3
    )
    
    feedback = kobart_tokenizer.decode(output[0], skip_special_tokens=True)
    feedback = refine_feedback(feedback)  # 후처리 함수로 자연스럽게 다듬기
    
    return feedback

# 실행부
if __name__ == "__main__":
    description = input("그림을 묘사한 텍스트를 입력하세요: ")
    emotion = analyze_emotion(description)
    feedback = generate_detailed_feedback(emotion, description)

    print(f"예측된 감정: {emotion}")
    print(f"상황 맞춤형 피드백: {feedback}")
