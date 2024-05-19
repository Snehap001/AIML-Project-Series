from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model_name = "gpt2-medium" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '50256'})

model.eval()


def generate_followup_questions(question, num_followups=3, max_length=50):
    prompt = f"Generate {num_followups} follow-up questions for the question: {question}\n1."    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True
    )

    followups = tokenizer.decode(outputs[0], skip_special_tokens=True)
    followup_list = followups.split('\n')[1:num_followups+1]  
    return followup_list


question = "I want to take admission into the college"
followup_questions = generate_followup_questions(question)
for i, fq in enumerate(followup_questions, 1):
    print(f"Follow-up question {i}: {fq}")
