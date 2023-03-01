from newspaper import Article

# print('--------------------------------')
# url = 'https://en.wikipedia.org/wiki/Bug_(comics)'#https://en.wikipedia.org/wiki/Barack_Obama'
# article = Article(url)
# article.download()
# article.parse()
# print(article.text)

from typing import List



def repeat_keyword(text):
    new_text = ""
    key_list = []
    count = 0
    j = 0

    for line in text.splitlines():
     #   print('------------',line,'--------------')
        if len(line)<20 and len(line)>5:
            key_list+=[line]
            j+=1

        else:
            if len(key_list) and len(line)>5:
                new_text+=key_list[j-1]+' : '+line
                if line[-1]=='.':
                    new_text+='\n'#'#+'\n'
            else:
                new_text+=line
                if len(line) and line[-1] == '.':
                    new_text += '\n'  # '#+'\n'
                #+'\n'

   # print(new_text)
    #print('======new======')

    return new_text



def html2arti(url):
    #"url = 'https://en.wikipedia.org/wiki/Barack_Obama'
    article = Article(url)
    article.download()
    article.parse()
    return article.text

if __name__ == "__main__":

    test_url= 'https://en.wikipedia.org/wiki/Barack_Obama'

    text = html2arti(test_url)
    text = repeat_keyword(text)
    text_path = 'obama_wiki'
    with open(text_path, "a") as f:
        f.write(text)