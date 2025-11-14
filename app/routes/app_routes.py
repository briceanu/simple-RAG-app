from fastapi import APIRouter, HTTPException, status, UploadFile, Body
from app.logic.logic import upload_file_logic, gpt_response, remove_index
from typing import Annotated
from app.schemas import UploadFileSchemaOut
from pydantic import Field

router = APIRouter(prefix="/v1")


@router.post("/upload-file", response_model=UploadFileSchemaOut)
async def upload_file(
    upload_file: UploadFile,
    category: Annotated[
        str, Body(description="provide a category where to save the data")
    ],
):
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


@router.get("/ask-question")
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
    try:
        result = await gpt_response(question, category)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occured: {str(e)}",
        )


@router.delete("/remove-index")
async def delete_index(index: Annotated[str, Field(..., max_length=40)]):
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
