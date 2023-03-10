import requests
import pypdf
def split_pdf(file, chunk_chars=1000, overlap=50):
    """
    Pre-process PDF into chunks
    Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
    """

    pdfFileObj = open("Papers/%s.pdf" % file, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    splits = []
    split = ""
    text = ""
    loc_dic = {}
    j = 0
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        text +=split
        while len(split) > chunk_chars:

            splits.append(split[:chunk_chars])
            split = split[chunk_chars - overlap:]
            loc_dic[str(j)]=i
            j += 1
        splits.append(split)
        loc_dic[str(j)] = i
        j+=1

    pdfFileObj.close()
    #print(i,'----',len(splits))
    return splits,loc_dic,text

splits,loc_dic,text = split_pdf('bert')
# for i,text in enumerate(splits):
#    # print(i,'\n',text)
#     print(loc_dic[str(i)])


import requests
# Edit this One AI API call using our studio at https://studio.oneai.com/?pipeline=WrpDCA&share=true

def summary_text(text):
    api_key = "8c3dc867-6a3f-46c1-8320-414ee7071389"
    url = "https://api.oneai.com/api/v0/pipeline"
    headers = {
      "api-key": api_key,
      "content-type": "application/json"
    }
    payload = {
      "input":text,# "One AI raises $8M to curate business-specific NLP models\nWhether to power translation to document summarization, enterprises are increasing their investments in natural language processing (NLP) technologies. According to a 2021 survey from John Snow Labs and Gradient Flow, 60% of tech leaders indicated that their NLP budgets grew by at least 10% compared to 2020, while a third said that spending climbed by more than 30%.\nIt’s a fiercely competitive market. Beyond well-resourced startups like OpenAI, Cohere, AI21 Labs and Hugging Face and tech giants including Google, Microsoft and Amazon, there’s a new crop of vendors building NLP services on top of open source AI models. But Yochai Levi isn’t discouraged. He’s one of the co-founders of One AI, an NLP platform that today emerged from stealth with $8 million led by Ariel Maislos, Tech Aviv, Sentinel One CEO Tomer Wiengarten and other unnamed venture firms and angel investors.\n“While the market is growing fast, advanced NLP is still used mainly by expert researchers, Big Tech and governments,” Levi told TechCrunch via email. “We believe that the technology is nearing its maturity point, and after building NLP from scratch several times in the past, we decided it was time to productize it and make it available for every developer.”\nLevi lays out what he believes are the major challenges plaguing NLP development. It’s often difficult to curate open source models, he argues, because they have to be matched both to the right domain and task. For example, a text-generating model trained to classify medical records would be a poor fit for an app designed to create advertisements. Moreover, models need to be constantly retrained with new data — lest they become “stale.” Case in point, OpenAI’s GPT-3 responds to the question “Who’s the president of the U.S.?” with the answer “Donald Trump” because it was trained on data from before the 2020 election.\nLevi believes the solution is a package of NLP models trained for particular business use cases — in other words, One AI’s product. He teamed up with CEO Amit Ben, CPO Aviv Dror and CSO Asi Sheffer in 2021 to pursue the idea. Ben previously was the head of AI at LogMeIn after the company acquired his second startup, Nanorep, an AI and chatbot vendor. Dror helped to co-found Nanorep and served as a platform product manager at Wix. Sheffer, a former data scientist at Nanorep, was the principal data scientist at LogMeIn. As for Levi, he was the VP of online marketing at LivePerson and the head of marketing at WeWork.\nOne AI offers a set of models that can be mixed and matched in a pipeline to process text via a single API call. Each model is selected and trained for its applicability to the enterprise, Levi said, and automatically matched by the platform to a customer’s task and domain (e.g., conversation summarization, sales insights, topic detection and proofreading). One AI’s models can also be combined with open source and proprietary models to extend One AI’s capabilities.\nThe platform’s API accepts text, voice and video inputs of various formats. With One AI’s language studio, users can experiment with the APIs and generate calls to use in code.\n“With the maturation of Language AI technologies, it is finally time for machines to start adapting to us,” Ben told TechCrunch via email. “The adoption of language comprehension tools by the broader developer community is the way to get there.”\nOne AI prices its NLP service across several tiers, including a free tier that includes processing for up to one million words a month. The stackable “growth tier” adds 100,000 words for $1.\nThe hurdle One AI will have to overcome is convincing customers that its services are more attractive than what’s already out there. In April, OpenAI said that tens of thousands of developers were using GPT-3 via its API to generate words for over 300 apps. But, with Fortune Business Insights pegging the NLP market at $16.53 billion in 2020, it could be argued that there’s a large enough slice of the pie for newcomers.\nAdded TehAviv founder and managing partner Yaron Samid: “Language is the most valuable untapped resource, beyond the reach of most products and companies. This is true to most industries and domains and results in negative outcomes that include everything from lost sales to lower levels of user engagement and loyalty to reputation damage. Unleashing Language AI allows us to harness the power of this unstructured data and turn it into useful information and insights.”\nOne AI says that a portion of the seed round proceeds will be put toward expanding its 22-person team, which includes 10 NLP data scientists.\n",
      "input_type": "article",
        "output_type": "json",
      "multilingual": {
        "enabled": True
      },
      "steps": [
          {
          "skill": "summarize"
        }
      ],
    }

    r = requests.post(url, json=payload, headers=headers)
    data = r.json()
    return data["output"][0]["contents"][0]["utterance"]



