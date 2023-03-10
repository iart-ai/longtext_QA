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
from langchain.llms import OpenAI, OpenAIChat
import openai
from test_summ import summary_text,split_pdf
# Image banner for Streamlit app
st.sidebar.image("Img/sidebar_img.jpeg")

# Get papers (format as pdfs)
papers = [l.split('.')[0] for l in os.listdir("Papers/") if l.endswith('.pdf')]
selectbox = st.sidebar.radio('Which paper to distill?', papers)
os.environ["OPENAI_API_KEY"] = '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'
#'#'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'  # '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'#'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'
# Paper distillation
chat_pref = [
    {
        "role": "system",
        "content": "You are a scholarly researcher that answers in an unbiased, scholarly tone. "
                  "You sometimes refuse to answer if there is insufficient information.",
    }
]

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
def get_last_line(generated):
    if '\n' not in generated:
        last_line = generated
    else:
        last_line = generated.split('\n')[-1]

    return last_line




class PaperDistiller:

    def __init__(self, paper_name):

        self.name = paper_name
        self.answers = {}
        self.answers_source = {}
        self.query_store = []
        # Use pickledb as local q-a store (save cost)
        self.cached_answers = pickledb.load('distller.db', auto_dump=False, sig=False)
        self.loc_dic = pickledb.load('%s_loc_dic.db'%self.name, auto_dump=False, sig=False)
        self.prompt = ['''Question: Who lived longer, Muhammad Ali or Alan Turing?
                             Are follow up questions needed here: Yes.
                             Follow up: How old was Muhammad Ali when he died?


                             Question: How many tokens or examples are in the training set?
                             Are follow up questions needed here: Yes.
                             Follow up: What is the training set?


                             Question: When was the founder of craigslist born?
                             Are follow up questions needed here: Yes.
                             Follow up: Who was the founder of craigslist?



                             Question: Who was the maternal grandfather of George Washington?
                             Are follow up questions needed here: Yes.
                             Follow up: Who was the mother of George Washington?
                             Intermediate answer: The mother of George Washington was Mary Ball Washington.
                             Follow up: Who was the father of Mary Ball Washington?


                             Question: Are both the directors of Jaws and Casino Royale from the same country? 
                             Are follow up questions needed here: Yes. 
                             Follow up: Who is the director of Jaws? 
                             Intermediate Answer: The director of Jaws is Steven Spielberg. 
                             Follow up: Where is Steven Spielberg from? 
                             Intermediate Answer: The United States. 
                             Follow up: Who is the director of Casino Royale? 
                             Intermediate Answer: The director of Casino Royale is Martin Campbell. 
                             Follow up: Where is Martin Campbell from? 


                             Question: ''',
                       '''
                       Are follow up questions needed here:''', ]
        # self.prompt = ['''Question: Who lived longer, Muhammad Ali or Alan Turing?
        #                 Are follow up questions needed here: Yes.
        #                 Follow up: How old was Muhammad Ali when he died?
        #                 Intermediate answer: Muhammad Ali was 74 years old when he died.
        #                 Follow up: How old was Alan Turing when he died?
        #                 Intermediate answer: Alan Turing was 41 years old when he died.
        #                 So the final answer is: Muhammad Ali
        #
        #                 Question: When was the founder of craigslist born?
        #                 Are follow up questions needed here: Yes.
        #                 Follow up: Who was the founder of craigslist?
        #                 Intermediate answer: Craigslist was founded by Craig Newmark.
        #                 Follow up: When was Craig Newmark born?
        #                 Intermediate answer: Craig Newmark was born on December 6, 1952.
        #                 So the final answer is: December 6, 1952
        #
        #                 Question: Who was the maternal grandfather of George Washington?
        #                 Are follow up questions needed here: Yes.
        #                 Follow up: Who was the mother of George Washington?
        #                 Intermediate answer: The mother of George Washington was Mary Ball Washington.
        #                 Follow up: Who was the father of Mary Ball Washington?
        #                 Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
        #                 So the final answer is: Joseph Ball
        #
        #                 Question: Are both the directors of Jaws and Casino Royale from the same country?
        #                 Are follow up questions needed here: Yes.
        #                 Follow up: Who is the director of Jaws?
        #                 Intermediate Answer: The director of Jaws is Steven Spielberg.
        #                 Follow up: Where is Steven Spielberg from?
        #                 Intermediate Answer: The United States.
        #                 Follow up: Who is the director of Casino Royale?
        #                 Intermediate Answer: The director of Casino Royale is Martin Campbell.
        #                 Follow up: Where is Martin Campbell from?
        #                 Intermediate Answer: New Zealand.
        #                 So the final answer is: No
        #
        #                 Question: ''',
        #                '''
        #                Are follow up questions needed here:''', ]

    # def split_pdf(self, chunk_chars=3000, overlap=50):
    #     """
    #     Pre-process PDF into chunks
    #     Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
    #     """
    #
    #     pdfFileObj = open("Papers/%s.pdf" % self.name, "rb")
    #     pdfReader = pypdf.PdfReader(pdfFileObj)
    #     splits = []
    #     split = ""
    #     for i, page in enumerate(pdfReader.pages):
    #         split += page.extract_text()
    #         while len(split) > chunk_chars:
    #             splits.append(split[:chunk_chars])
    #             split = split[chunk_chars - overlap:]
    #         splits.append(split)
    #     pdfFileObj.close()
    #     return splits
    def split_pdf(self, chunk_chars=4000, overlap=50):
        pdfFileObj = open("Papers/%s.pdf" % self.name, "rb")
        pdfReader = pypdf.PdfReader(pdfFileObj)
        splits = []
        split = ""
        loc_dic = {}
        j = 0
        for i, page in enumerate(pdfReader.pages):
            split += page.extract_text()
            while len(split) > chunk_chars:

                splits.append(split[:chunk_chars])
                split = split[chunk_chars - overlap:]
                #self.loc_dic[str(j)]=i
                j += 1
            splits.append("chunk %dth-"%j+"-page %d : "%i +split)
            loc_dic[str(j)] = i
        pdfFileObj.close()
        #self.loc_dic = loc_dic
        #print(i,'----',len(splits))
        return splits#,loc_dic

    def read_or_create_index(self):
        """
        Read or generate embeddings for pdf
        """

        if os.path.isdir('Index/%s' % self.name):
            print("Index Found!")
            self.ix = FAISS.load_local('Index/%s' % self.name, OpenAIEmbeddings())
        else:
            print("Creating index!")
            self.splits = self.split_pdf()
            self.ix = FAISS.from_texts(self.splits, OpenAIEmbeddings())
            # Save index to local (save cost)
            self.ix.save_local('Index/%s' % self.name)
            self.loc_dic.dump()

    def creat_query_store_index(self,new_query):
          #  self.query_store.append(new_query)
        self.query_ix = []
        if len(self.query_store):
            self.query_ix = FAISS.from_texts(self.query_store, OpenAIEmbeddings())

        self.query_store.append(new_query)

    def query_similarity_search(
            self, query, k
    ) :
        #self.creat_query_store_index(query)
        querys = []
        if self.query_ix:
            query_results = self.query_ix.similarity_search_with_score(query, k)
            for (query_re, score) in query_results:
                if score>0.2:
                    querys.append(query_re)
                print(score)
        return querys

    def extend_infor_with_his(self,query_results,sim_querys):
        #
        for sim_query in sim_querys:
            sq = sim_query.page_content
            if sq in self.answers:
                sim_query.page_content+='? The answer is: '+ self.answers[sq]
                query_results.append(sim_query)
                print('recall history : ', sim_query.page_content)
        return query_results

    def get_answer(self, query):
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
            sim_querys = []
            self.creat_query_store_index(query)
            if len(self.query_store)>0:
                #
                if len(self.query_store)==1:
                    sim_querys = self.query_similarity_search(query, k=1)
                else:
                    sim_querys = self.query_similarity_search(query, k=2)
            if len(sim_querys):
                query_results =self.extend_infor_with_his(query_results,sim_querys)

            self.answers[query] = self.get_answer_with_cahtgpt(query_results,
                                                                   query)
            self.cached_answers.set(query + "-%s" % self.name, self.answers[query])

        return self.answers[query]



    def query_and_distill(self, query):
        """
        Query embeddings and pass relevant chunks to LLM for answer
        """

        # Answer already in memory
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
            self.creat_query_store_index(query)
            sim_querys = []
            if len(self.query_store)>0:
                if len(self.query_store)>1:
                    sim_querys = self.query_similarity_search(query, k=2)
                else:
                    sim_querys = self.query_similarity_search(query, k=1)
            if len(sim_querys):
                query_results  = self.extend_infor_with_his(query_results,sim_querys)
            self.answers[query],self.answers_source[query] = self.get_answer_with_source(query_results,
                                                               query)

            query_int = input("\nWhether this answer fit your question?y/n: \n")
            if query_int == 'y':
                print('======Move to the next question=======')
                self.cached_answers.set(query + "-%s" % self.name, self.answers[query])
            elif query_int == 'n':

                #   print('We try to find a intermediate question for you\n')
                del self.answers[query]
                res, query_int = self.promptf(query, self.prompt)
                if query_int == 'n':
                    print('======Limited. Move to the next question=======')
                else:
                    print('======Move to the next question=======')
                    self.cached_answers.set(query + "-%s" % self.name, self.answers[query])

            # chain.run(input_documents=query_results, question=query)
            #    print(query_re.page_content,'------')
            # chain = load_qa_chain(OpenAI(temperature=0.25), chain_type="stuff")
            # chain = load_qa_chain(OpenAI(temperature=0.25), chain_type="stuff")
            # self.answers[query] = chain.run(input_documents=query_results, question=query)
            # self.answers[query] = self.get_answer_with_cahtgpt(query_results,query)#chain.run(input_documents=query_results, question=query)

            return
    def call_gpt(self, cur_prompt, stop):
        ans = openai.Completion.create(
            model="text-davinci-003",
            max_tokens=256,
            stop=stop,
            prompt=cur_prompt,
            temperature=0)
        returned = ans['choices'][0]['text']
        print((returned), end='')

        return returned

    def promptf(self, question, prompt, intermediate="\nIntermediate answer:", followup="Follow up:",
                finalans='\nSo the final answer is:'):
        #  print('We try to find another intermediate question for you\n')

        cur_prompt = prompt[0] +'whats the follow up question for : ' + question # + prompt[1]

        ret_text = self.call_chatgpt(cur_prompt)
        count = 0
        query_int = 'n'
        while query_int!='y':

            cur_prompt += ret_text
            question_e = extract_question(ret_text)
            print('intermedia question is : ', question_e)
            external_answer = self.get_answer(question_e)
            cur_prompt = question + 'and the follow up question for is' + question_e+'\n the intermediate answer is: '+external_answer
            self.answers[question] = self.get_answer(cur_prompt+'\n so the final answer is:')
            query_int = input("\nWhether this answer fit your question?y/n: \n")
          #  del self.answers[question]
            if query_int == 'y':
             #   self.cached_answers()
                print('======Move to the next question=======')
                self.cached_answers.set(question + "-%s" % self.name, self.answers[question])
                break
            else:
                #  print('trying to find another intermediate question for you\n')
                # We only get here in the very rare case that Google returns no answer.
                del self.answers[question]
               # def self.cached_answers#[question + "-%s" % self.name]
                cur_prompt += 'so what is the follow up question?\n'
                #
                ret_text = self.call_chatgpt(cur_prompt)
                print('the follow up question is : ', ret_text)
            # external_answer = self.get_answer(question)
            # cur_prompt += gpt_answer
            # external_answer = get_answer(question)
            count += 1
            # print(count,'----------')
            if count == 5:
                break
        return external_answer, query_int

    def call_chatgpt(self, cur_prompt):
        completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": cur_prompt
                             }
                        ]
                    )
        returned =completion.choices[0].message["content"]


        return returned

    def get_answer_with_source(self, query_results, query):

        llm = OpenAIChat(temperature=0.1, max_tokens=1000, prefix_messages=chat_pref)  # OpenAI(model='',temperature=0)

        chain = load_qa_chain(llm, chain_type="stuff")
        self.answers[query] = chain.run(input_documents=query_results, question=query)

        self.answers_source[query]=chain.run(input_documents=query_results,question='question is'+query+'the answer is :'+ self.answers[query] +'So the final question is: the answer is extracted from where in the article, which chunk and page?')

        print('AI: ', self.answers[query])
        print('AI: The answer is from ', self.answers_source[query])
        return self.answers[query], self.answers_source[query]

    def get_answer_with_cahtgpt(self, query_results, query):

        llm = OpenAIChat(temperature=0.1, max_tokens=1000, prefix_messages=chat_pref)  # OpenAI(model='',temperature=0)

        chain = load_qa_chain(llm, chain_type="stuff")
        self.answers[query] = chain.run(input_documents=query_results, question=query)

        print('AI: ', self.answers[query])
        return self.answers[query]
    # def get_answer_with_cahtgpt(self,conten
    # ts,query):
    #     ans_his = ""
    #     for content in contents:
    #         completion = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "user", "content": content.page_content
    #                  }
    #             ]
    #         )
    #         ans_his+=completion.choices[0].message["content"]
    #     sum_ans = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "user", "content": ans_his+'question: '+query}
    #         ]
    #     )
    #     print('AI: ', sum_ans.choices[0].message["content"])
    #     return sum_ans.choices[0].message["content"]


    def cache_answers(self):
        """
        Write answers to local pickledb
        """
        self.cached_answers.dump()

    def summary_and_suggest(self):
       # query = input('what is the scope of this article   \n')
        if 'summary' in self.answers:
            print('find summary: ', self.answers['summary'])
            print('find suggested quetions: ', self.answers['suggest'])
        else:


            #llm = OpenAIChat(temperature=0.1, max_tokens=1000, prefix_messages=chat_pref)  # OpenAI(model='',temperature=0)
            print('------summarying--------')
            _,_,text = split_pdf()
            summary_res = summary_text(self.name)
            print('Summary: ', summary_res)

            query = 'recommend a list of questions about '+ summary_res
            time.sleep(10)
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content":query
                     }
                ]
            )

            print('Suggested questions : ',completion.choices[0].message["content"] )
            self.answers['summary'] = summary_res
            self.answers['suggest'] = completion.choices[0].message["content"]

            self.cached_answers()

        return
        # chain = load_qa_chain(llm, chain_type="stuff")
        # print('-----------summarying------------')
        # query_results = self.ix.similarity_search(query, k=len(self.query_ix)//4)
        # self.answers[query] = chain.run(input_documents=query_results, question=query)
        #
        # print('AI: ', self.answers[query])
        # return self.answers[query]


def pdfQA(filename):
    p = PaperDistiller(filename)
    p.read_or_create_index()
 #   p.summary_and_suggest()

    query = input('What do you want to ask the bot?   \n')
    while query != 'quit':
        p.query_and_distill(query)
  #      p.cache_answers()
        query = input('What do you want to ask the bot?   \n')

    return 0


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
