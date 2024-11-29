"""Handles user tasks."""

from backend.app.repositories.user_repo import UserRepository


class UserService:
    def __init__(self):
        self.user_repo = UserRepository()

    def register_user(self, username, password, email):
        if self.user_repo.find_user_by_username(username):
            raise ValueError("Username already exists")
        user_id = self.user_repo.create_user(username, password, email)
        return {"user_id": user_id, "username": username}

    def login_user(self, username, password):
        user = self.user_repo.find_user_by_username(username)
        if not user:
            raise ValueError("Invalid username or password")

        user_id, _, password_hash = user
        if not self.user_repo.verify_password(password_hash, password):
            raise ValueError("Invalid username or password")

        return {"user_id": user_id, "username": username, "message": "Login successful"}
