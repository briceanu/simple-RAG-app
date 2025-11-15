from fastapi import APIRouter, HTTPException, status, UploadFile, Body
from app.logic.logic import upload_file_logic, gpt_response, remove_index
from typing import Annotated
from app.schemas import (
    UploadFileSchemaOut,
    LlmResponseSchema,
    RemoveIndexResponseSchema,
)
from pydantic import Field

router = APIRouter(prefix="/v1")


@router.post(
    "/upload-file",
    response_model=UploadFileSchemaOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    upload_file: UploadFile,
    category: Annotated[
        str, Body(description="provide a category where to save the data")
    ],
):
    """
    Upload a text file, split it into chunks, and store the processed content in a vector database.

    This endpoint:
    - Accepts a text file from the client.
    - Passes the file to the `upload_file_logic` function, which:
        * Reads and decodes the file.
        * Splits the text into smaller chunks.
        * Ensures the Pinecone index exists (or creates it if missing).
        * Inserts the chunks into the Pinecone vector database under the specified category.
    - Returns a success response including how many chunks were stored.

    Args:
        upload_file (UploadFile):
            The uploaded text file containing the content to be chunked and stored.
        category (str):
            The namespace (category) within Pinecone where the data will be saved.

    Returns:
        UploadFileSchemaOut:
            A structured response indicating successful processing and the number of chunks saved.

    Raises:
        HTTPException:
            - 500: If an unexpected error occurs during the upload or storage process.
            - Other HTTPExceptions raised directly from underlying logic are re-raised.
    """
    try:
        result = await upload_file_logic(upload_file=upload_file, category=category)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured: {str(e)}",
        )


@router.get(
    "/ask-question", response_model=LlmResponseSchema, status_code=status.HTTP_200_OK
)
async def get_gpt_answer(
    question: Annotated[str, Field(..., max_length=200)],
    category: Annotated[
        str,
        Field(
            ...,
            max_length=200,
            description="provide the category that you saved the data in.",
        ),
    ],
):
    """
    Retrieve an LLM-generated answer based on a user question and stored context.

    This endpoint performs a Retrieval-Augmented Generation (RAG) operation:
    1. Sends the question and category to the `gpt_response` logic function.
    2. Retrieves relevant text chunks from the Pinecone vector index.
    3. Injects the question and retrieved context into a prompt template.
    4. Queries an LLM (e.g., GPT-4) to generate a final response.
    5. Returns the model's answer wrapped in `LlmResponseSchema`.

    Args:
        question (str):
            The user's question. Must not exceed 200 characters.
        category (str):
            The namespace/category inside Pinecone where chunks were stored.

    Returns:
        LlmResponseSchema:
            A structured response containing the LLM-generated answer.

    Raises:
        HTTPException:
            - 500: If any unexpected error occurs during processing or model inference.
    """
    try:
        result = await gpt_response(question, category)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured: {str(e)}",
        )


@router.delete(
    "/remove-index",
    response_model=RemoveIndexResponseSchema,
    status_code=status.HTTP_200_OK,
)
async def delete_index(index: Annotated[str, Field(..., max_length=40)]):
    """
    Delete a Pinecone index by name through an API endpoint.

    This endpoint acts as a wrapper around the `remove_index` service function.
    It attempts to remove the specified index from Pinecone and returns a success
    response if the deletion is successful.

    Args:
        index (str):
            The name of the Pinecone index to delete. Must be 40 characters or fewer.

    Returns:
        RemoveIndexResponseSchema:
            A response indicating that the index was successfully removed.

    Raises:
        HTTPException:
            - 404: If the index does not exist in Pinecone.
            - 500: If any unexpected server error occurs during deletion.
    """
    try:
        result = await remove_index(index)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured: {str(e)}",
        )
