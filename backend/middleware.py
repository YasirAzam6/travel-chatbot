# middleware.py
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)
SESSION_COOKIE_NAME = "chat_session_id" # Define a constant for the cookie name

class ChatSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # 1. Try to get session_id from the cookie
        session_id = request.cookies.get(SESSION_COOKIE_NAME)
        new_session_created = False

        # 2. If no cookie/session_id, create a new one
        if not session_id:
            session_id = str(uuid.uuid4())
            new_session_created = True
            logger.info(f"No session cookie found. Generated new session: {session_id}")
        else:
            logger.debug(f"Using existing session cookie: {session_id}")

        # 3. Store the session_id in request.state so the endpoint can access it
        request.state.session_id = session_id

        # 4. Call the next middleware or the endpoint itself
        response = await call_next(request)

        # 5. If we created a new ID in step 2, set the cookie in the response
        if new_session_created:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_id,
                httponly=True,  # Prevent client-side JS access
                max_age=60 * 60 * 24 * 7, # Example: 7 days validity
                samesite="lax", # Recommended setting for most cases
                # secure=True, # Uncomment this if your app is served over HTTPS ONLY
            )
            logger.info(f"Setting new session cookie in response: {session_id}")

        # 6. Return the response
        return response