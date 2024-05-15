from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Input text


# Parse the input text
def summary(text_lst):
    text=""
    for i in text_lst:
        text+=i

    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Use the LSA summarizer
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)  # Summarize to 2 sentences

    # Print the summary
    sum_response=""
    for sentence in summary:
        sum_response+=str(sentence)
    return sum_response
