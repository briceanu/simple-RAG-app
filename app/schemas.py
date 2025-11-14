from pydantic import BaseModel


class LlmResponseSchema(BaseModel):
    response: str | int


class RemoveIndexResponseSchema(BaseModel):
    response: str


class UploadFileSchemaOut(BaseModel):
    success: dict[str, str]
