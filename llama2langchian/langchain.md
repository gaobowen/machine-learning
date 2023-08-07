```py
# !pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple 
# !pip install pypdf -i https://pypi.tuna.tsinghua.edu.cn/simple 
# !pip install sentence_transformers -i https://pypi.tuna.tsinghua.edu.cn/simple 
# !pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

```py
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "/data/text-generation-webui/models/Llama-2-7b-Chat-GPTQ-128"
model_basename = "gptq_model-4bit-128g"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, use_fast=True)

print('tokenizer done')

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path=model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

"""
To download from a specific branch, use the revision parameter, as in this example:

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision="gptq-4bit-32g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        quantize_config=None)
"""

prompt = "Tell me about AI in english"
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]'''

print("\n\n*** Generate:")

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# print('input_ids', type(input_ids))

# output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
# print('output tensor', output)
# outstring = tokenizer.decode(output[0])
# print('outstring', outstring)

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
```

```py
from langchain.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain.embeddings import HuggingFaceEmbeddings 

#加载本地文档

loader = DirectoryLoader('/tmp/philoai', 
                         glob="*.pdf", 
                         loader_cls=PyPDFLoader) 
documents = loader.load() 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                               chunk_overlap=100) #重叠50行
texts = text_splitter.split_documents(documents) 

# huggingface下载embeddings model ，可能会有网络问题需要重试
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                   model_kwargs={'device': 'cpu'}) 

# 生成保存 FAISS 向量数据库 生成一次即可
#vectorstore = FAISS.from_documents(texts, embeddings) 
#vectorstore.save_local('faisstest')

```

```py
from langchain import PromptTemplate 
from langchain.chains import RetrievalQA 
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS 

vectordb = FAISS.load_local('faisstest', embeddings)

retriever=vectordb.as_retriever(search_kwargs={'k':2})

qa_template = """Use the following pieces of information to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Context: {context} 
Question: {question} 
Only return the helpful answer below and nothing else. 
Helpful answer: 
"""

prompt = PromptTemplate(template=qa_template, 
                            input_variables=['context', 'question']) 

from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
#自定义langchain model
class CustomLLM(LLM):
    mymodel = ''
    def __init__(self, mymodel):
        super(CustomLLM, self).__init__()
        self.mymodel = mymodel

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        print('prompt', prompt)
        # print(type(self.mymodel))
        return self.mymodel(prompt)[0]['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"mymodel": self.mymodel}


#refine模式后续再用

dbqa = RetrievalQA.from_chain_type(llm=CustomLLM(pipe), 
                                       chain_type='stuff', 
                                       retriever=retriever, 
                                       return_source_documents=True, 
                                       chain_type_kwargs={'prompt': prompt}) 

response = dbqa({'query': 'Medical devices have historically'}) 

print(f'\nAnswer: {response["result"]}')
```
```py
    qa = """[INST] <<SYS>>\n
    Always answer the question, even if the context isn't helpful.
    Context information is below.\n
    ---------------------\n
    {context_str}\n
    ---------------------\n
    Given the context information and not prior knowledge. 
    if the context information isn't helpful, ignore the the context information.

    The answer always been translate into Chinese language.
    answer the question\n<</SYS>>[/INST][INST]
    """
    context_str = ''

    '''
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature. The answer always been translate into Chinese language.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
    The answer always been translate into Chinese language.
    '''
    splits = question.split('[INST]')

    # print('splits', splits, splits[len(splits)-1])

    lastq = splits[len(splits)-1].replace('[/INST]', '')

    print('lastq ==>', lastq)

    # 搜索知识库
    # myresults = vectordb.similarity_search(lastq, k=2)
    # print(myresults)
    # for i in range(len(myresults)):
    #     context_str += myresults[i].page_content + '\n\n'

    qa = qa.format(context_str=context_str)

    # print('qa ==>', qa)

    question = question.replace('[INST] <<SYS>>\nAnswer the questions.\n<</SYS>>', qa)
```

