import spacy

nlp = spacy.load("en_core_web_lg")


class ConversationContext:
    def __init__(self):
        self.previous_questions = []
    def extract_important_words_spacy(self,sentence):
        doc = nlp(sentence)
        important_words = set([token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'PROPN']])
        return important_words 
    def add_question(self, question):
        if(len(self.previous_questions)>3):
            self.previous_questions.pop(0)
            
        self.previous_questions.append(question)
    def combine_questions(self,this_question):
        parsed_questions = [nlp(question) for question in self.previous_questions]
        common_nouns = set()
        
        for parsed in parsed_questions:
            nouns=self.extract_important_words_spacy(parsed)
            if not common_nouns:
                common_nouns = nouns
            else:
                common_nouns.intersection_update(nouns)
        
        if not common_nouns:
            return this_question

        new_question = "I know about " + " and ".join(common_nouns) +" tell me about "+this_question+"?"
        return new_question

    

    

