import os
import time
import pypdf
import pickledb
import pandas as pd
import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI,OpenAIChat
import openai
# Image banner for Streamlit app
st.sidebar.image("Img/sidebar_img.jpeg")

# Get papers (format as pdfs)
papers  = [l.split('.')[0] for l in os.listdir("Papers/") if l.endswith('.pdf')]
selectbox = st.sidebar.radio('Which paper to distill?',papers)
os.environ["OPENAI_API_KEY"]='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'#'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'#'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'
# Paper distillation
chat_pref = [
    {
        "role": "system",
        "content": "You are a scholarly researcher that answers in an unbiased, scholarly tone. "
        "You sometimes refuse to answer if there is insufficient information.",
    }
]
def get_last_line(generated):
    if '\n' not in generated:
        last_line =  generated
    else:
        last_line = generated.split('\n')[-1]


    return last_line


def extract_question(generated):
    if '\n' not in generated:
        last_line = generated
    else:
        last_line = generated.split('\n')[-1]

    if 'Follow up:' not in last_line:
        print('we probably should never get here...' + generated)

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(':')[-1]

    if ' ' == after_colon[0]:
        after_colon = after_colon[1:]
    if '?' != after_colon[-1]:
        print('we probably should never get here...' + generated)

    return after_colon


class PaperDistiller:

    def __init__(self,paper_name):

        self.name = paper_name
        self.answers = {}
        # Use pickledb as local q-a store (save cost)
        self.cached_answers = pickledb.load('distller.db',auto_dump=False,sig=False)
        self.prompt = ['''Question: Who lived longer, Muhammad Ali or Alan Turing?
                        Are follow up questions needed here: Yes.
                        Follow up: How old was Muhammad Ali when he died?
                        Intermediate answer: Muhammad Ali was 74 years old when he died.
                        Follow up: How old was Alan Turing when he died?
                        Intermediate answer: Alan Turing was 41 years old when he died.
                        So the final answer is: Muhammad Ali 
                        
                        Question: When was the founder of craigslist born?
                        Are follow up questions needed here: Yes.
                        Follow up: Who was the founder of craigslist?
                        Intermediate answer: Craigslist was founded by Craig Newmark.
                        Follow up: When was Craig Newmark born?
                        Intermediate answer: Craig Newmark was born on December 6, 1952.
                        So the final answer is: December 6, 1952
                        
                        Question: Who was the maternal grandfather of George Washington?
                        Are follow up questions needed here: Yes.
                        Follow up: Who was the mother of George Washington?
                        Intermediate answer: The mother of George Washington was Mary Ball Washington.
                        Follow up: Who was the father of Mary Ball Washington?
                        Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
                        So the final answer is: Joseph Ball 
                        
                        Question: Are both the directors of Jaws and Casino Royale from the same country? 
                        Are follow up questions needed here: Yes. 
                        Follow up: Who is the director of Jaws? 
                        Intermediate Answer: The director of Jaws is Steven Spielberg. 
                        Follow up: Where is Steven Spielberg from? 
                        Intermediate Answer: The United States. 
                        Follow up: Who is the director of Casino Royale? 
                        Intermediate Answer: The director of Casino Royale is Martin Campbell. 
                        Follow up: Where is Martin Campbell from? 
                        Intermediate Answer: New Zealand. 
                        So the final answer is: No
                        
                        Question: ''',
                        '''
                        Are follow up questions needed here:''', ]


    def split_pdf(self,chunk_chars=2000,overlap=50):
        """
        Pre-process PDF into chunks
        Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
        """

        pdfFileObj = open("Papers/%s.pdf"%self.name, "rb")
        pdfReader = pypdf.PdfReader(pdfFileObj)
        splits = []
        split = ""
        for i, page in enumerate(pdfReader.pages):
            split += page.extract_text()
            if len(split) > chunk_chars:
                splits.append(split[:chunk_chars])
                split = split[chunk_chars - overlap:]
        pdfFileObj.close()
        return splits


    def read_or_create_index(self):
        """
        Read or generate embeddings for pdf
        """

        if os.path.isdir('Index/%s'%self.name):
            print("Index Found!")
            self.ix = FAISS.load_local('Index/%s'%self.name,OpenAIEmbeddings())
        else:
            print("Creating index!")
            self.ix = FAISS.from_texts(self.split_pdf(), OpenAIEmbeddings())
            # Save index to local (save cost)
            self.ix.save_local('Index/%s'%self.name)
    def get_answer(self,query):
        if query in self.answers:
            print("Answer found!")
            return self.answers[query]
            # Answer cached (asked in the past) in pickledb
        elif self.cached_answers.get(query + "-%s" % self.name):
            print("Answered in the past!")
            return self.cached_answers.get(query + "-%s" % self.name)
            # Generate the answer
        else:
            print("Generating answer!")
            query_results = self.ix.similarity_search(query, k=2)
            self.answers[query] = self.get_answer_with_cahtgpt(query_results,
                                                               query)
            self.cached_answers.set(query + "-%s" % self.name, self.answers[query])

        return self.answers[query]


    def query_and_distill(self,query):
        """
        Query embeddings and pass relevant chunks to LLM for answer
        """

        # Answer already in memory
        if query in self.answers:
            print("Answer found!")
            return self.answers[query]
        # Answer cached (asked in the past) in pickledb
        elif self.cached_answers.get(query+"-%s"%self.name):
            print("Answered in the past!")
            return self.cached_answers.get(query+"-%s"%self.name)
        # Generate the answer
        else:
            print("Generating answer!")
            query_results = self.ix.similarity_search(query, k=2)
            self.answers[query] = self.get_answer_with_cahtgpt(query_results,
                                                                   query)

            query_int = input("\nWhether this answer fit your question?y/n: \n")
            if query_int == 'y':
                print('======Move to the next question=======')
            elif query_int == 'n':

             #   print('We try to find a intermediate question for you\n')
                res,query_int = self.promptf(query, self.prompt)
                if query_int  == 'n':
                    print('======Limited. Move to the next question=======')
                else:
                    print('======Move to the next question=======')

            # chain.run(input_documents=query_results, question=query)
            #    print(query_re.page_content,'------')
            #chain = load_qa_chain(OpenAI(temperature=0.25), chain_type="stuff")
            #chain = load_qa_chain(OpenAI(temperature=0.25), chain_type="stuff")
            #self.answers[query] = chain.run(input_documents=query_results, question=query)
            #self.answers[query] = self.get_answer_with_cahtgpt(query_results,query)#chain.run(input_documents=query_results, question=query)
            self.cached_answers.set(query+"-%s"%self.name,self.answers[query])
            return self.answers[query]

    def promptf(self,question, prompt, intermediate="\nIntermediate answer:", followup="Follow up:",
                finalans='\nSo the final answer is:'):
      #  print('We try to find another intermediate question for you\n')

        cur_prompt = prompt[0] + question+prompt[1]



        ret_text = self.call_gpt(cur_prompt, intermediate)
        count = 0
        while followup in get_last_line(ret_text):

            cur_prompt += ret_text
            question = extract_question(ret_text)
          #  print('intermedia question is : ', question)
            external_answer = self.get_answer(question)
            query_int = input("\nWhether this answer fit your question?y/n: \n")

            if query_int == 'y':
                self.cached_answers()
                print('======Move to the next question=======')

                break
            else:
              #  print('trying to find another intermediate question for you\n')
                # We only get here in the very rare case that Google returns no answer.
                cur_prompt += intermediate

                ret_text = self.call_gpt(cur_prompt, ['\n' + followup, finalans])
               # external_answer = self.get_answer(question)
                # cur_prompt += gpt_answer
            # external_answer = get_answer(question)
            count += 1
            # print(count,'----------')
            if count == 3:
                break
        return external_answer,query_int

    def call_gpt(self,cur_prompt, stop):
        ans = openai.Completion.create(
            model="text-davinci-003",
            max_tokens=256,
            stop=stop,
            prompt=cur_prompt,
            temperature=0)
        returned = ans['choices'][0]['text']
        print((returned), end='')

        return returned
    def get_answer_with_cahtgpt(self,query_results,query):

        llm = OpenAIChat(temperature=0.1, max_tokens=1000, prefix_messages=chat_pref)  # OpenAI(model='',temperature=0)

        chain = load_qa_chain(llm, chain_type="stuff")
        self.answers[query] = chain.run(input_documents=query_results, question=query)


        print('AI: ', self.answers[query])
        return self.answers[query]


    def cache_answers(self):
        """
        Write answers to local pickledb
        """
        self.cached_answers.dump()

def pdfQA(filename):
    p = PaperDistiller(filename)
    p.read_or_create_index()

    query = input('What do you want to ask the bot?   \n')
    while query != 'quit':

        p.query_and_distill(query)
        p.cache_answers()
        query = input('What do you want to ask the bot?   \n')

    return  0

if __name__ == "__main__":
    # Select paper via radio button
    # print(selectbox)
    # p=PaperDistiller(selectbox)
    # p.read_or_create_index()

    # Pre-set queries for each paper
    # TO DO: improve this w/ prompt engineering
    # queries = ["What is the main innovation or new idea in the paper?",
    #            "How many tokens or examples are in the training set?",
    #            "Where is the training set scraped from or obtained and what modalities does it include?",
    #            "What are the tasks performed by the model?",
    #            "How is evaluation performed?",
    #            "What is the model architecture and what prior work used similar architecture?"]
    #
    # # UI headers
    # headers = ["Innovation","Dataset size","Dataset source","Tasks","Evaluation","Architecture"]
    #
    # # Outputs
    # st.header("`Paper Distiller`")
    # # for q,header in zip(queries,headers):
    #     st.subheader("`%s`"%header)
    #     st.info("`%s`"%p.query_and_distill(q))
        # time.sleep(3) # may be needed for OpenAI API limit
    # query = input('What do you want to ask the bot?   \n')
    #     if query == 'quit':
    #         return 0
    # Cache the answers
    pdfQA('attention_is_all_you_need')
