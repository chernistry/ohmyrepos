"""Exception handlers for FastAPI."""

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


class ErrorResponse:
    """Standard error response format."""

    def __init__(self, code: str, message: str, details: dict = None):
        self.error = {
            "code": code,
            "message": message,
        }
        if details:
            self.error["details"] = details

    def to_dict(self):
        return self.error


async def validation_exception_handler(request: Request, exc: Exception):
    """Handle validation errors."""
    logger.warning(
        "validation_error",
        path=request.url.path,
        error=str(exc),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            code="validation_error",
            message="Invalid request data",
            details={"errors": str(exc)},
        ).to_dict(),
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        "unhandled_exception",
        path=request.url.path,
        request_id=request_id,
        error=str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            code="internal_error",
            message="An internal error occurred",
            details={"request_id": request_id},
        ).to_dict(),
    )
