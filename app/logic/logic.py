import os
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from fastapi import UploadFile
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pinecone import IndexEmbed, Metric, Pinecone
from pinecone.exceptions import NotFoundException, PineconeException
from fastapi import HTTPException, status
from app.schemas import LlmResponseSchema, RemoveIndexResponseSchema, UploadFileSchemaOut

load_dotenv()


pine_cone_key = os.getenv("PINECONE_API_KEY")
gpt_key = os.getenv("CHATGPT_API_KEY")
pinecone_cloud = os.getenv('PINECONE_CLOUD')
pinecone_region = os.getenv('PINECONE_REGION')
# create a pincone client
pc = Pinecone(api_key=pine_cone_key)


index_name = "employees"


async def upload_file_logic(upload_file: UploadFile, category: str):
    try:
        # read and decode the file
        file_bytes = await upload_file.read()
        file_text = file_bytes.decode("utf-8")
        # create chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=100, chunk_overlap=0, separator='------')

        chunks = text_splitter.split_text(file_text)
        # check for index in DB
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud=pinecone_cloud,
                region=pinecone_region,
                embed=IndexEmbed(
                    model="llama-text-embed-v2",
                    metric=Metric.COSINE,
                    field_map={
                        "text": "chunk",
                    },
                ),
            )
        # creating list of records
        records = [
            {"_id": f"{upload_file.filename}-{i}", "chunk": chunk}
            for i, chunk in enumerate(chunks)
        ]
        dense_index = pc.Index(index_name)
        # insert data into DB
        dense_index.upsert_records(namespace=category, records=records)
        return UploadFileSchemaOut(success={"data saved": f"length: {len(records)}"})

    except PineconeException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An error occured: {str(e)}",
        )


async def remove_index(index: str):
    try:
        pc.delete_index(name=index)
        return RemoveIndexResponseSchema(response='index removed')
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No index with the name {index} found.",
        )


async def gpt_response(question: str, category: str):
    # extract only the chunk text from metadata
    dense_index = pc.Index(index_name)
    results = dense_index.search(
        namespace=category, query={"inputs": {"text": question}, "top_k": 2}
    )
    # create a prompt where where context and question are inserted
    template = ChatPromptTemplate.from_template("""
    You are a knowledgeable and helpful assistant. Always respond politely and clearly.
    Question:
    {question}

    Context:
    {context}

    Instructions:
    - Use the context provided to answer the question.
    - If the context does not contain relevant information, respond with:
    "I don't know based on the provided data."
    - Keep your response concise, informative, and easy to understand.
    """)
    # format the prompt to and the variables
    formated_template = template.format_prompt(
        question=question, context=results)
    # create a llm model
    model = ChatOpenAI(api_key=gpt_key, model="gpt-4.1")
    # create response
    response = model.invoke(formated_template)
    # return response
    return LlmResponseSchema(response=response.content)
