from langchain_ollama import OllamaLLM             
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
model = OllamaLLM(model="llama3.2")
template="""
You are an analytical assistant. You will be provided with a set of restaurant reviews. Your task is to analyze these reviews and provide insights based on the question asked. Please ensure your responses are concise, relevant, and based solely on the information provided in the reviews.
if i told you to give me a summary of the reviews, you should provide a brief overview highlighting the main points and sentiments expressed in the reviews. If asked for specific details, focus on extracting and presenting that information clearly. Avoid making assumptions or adding information not present in the reviews.
you are a helpful assistant. here are some reviews:{reviews} 
The assistant MUST be accurate, concise, and honest.” “The assistant MUST not invent facts or tool results.
the output should be in the following format:
Summary: [Provide a brief summary of the reviews]
Answer the following question: {question}"""
prompt = ChatPromptTemplate.from_template(template)
chain= prompt | model
while True:
    print("\n\n###################################################")
    question = input("Your question: ")
    print("\n\n")
    if question=="q":
        break
    reviews = retriever.invoke(question)
    result=chain.invoke({"reviews": reviews, "question": question})
    print(result)