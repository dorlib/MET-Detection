"""Handles user related requests."""

from fastapi import APIRouter, HTTPException
from backend.app.services.user_service import UserService

router = APIRouter()
user_service = UserService()


@router.post("/signup")
async def signup(username: str, password: str, email: str):
    try:
        user = user_service.register_user(username, password, email)
        return {"message": "User created successfully", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(username: str, password: str):
    try:
        user = user_service.login_user(username, password)
        return {"message": "Login successful", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
