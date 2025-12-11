"""
Authentication and Access Control Module
Manages role-based access to chemistry and biology research assistants.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ResearcherRole(str, Enum):
    """Researcher role definitions"""
    CHEM_RESEARCHER = "CHEM_RESEARCHER"
    BIO_RESEARCHER = "BIO_RESEARCHER"


class Credentials(BaseModel):
    """User credentials"""
    researcher: str
    password: str


class AccessConfig:
    """Access control configuration"""

    # Password to role mapping
    PASSWORD_ROLE_MAP = {
        "1122": ResearcherRole.CHEM_RESEARCHER,
        "3344": ResearcherRole.BIO_RESEARCHER,
    }

    # Role to model adapter mapping
    ROLE_MODEL_MAP = {
        ResearcherRole.CHEM_RESEARCHER: "chem-expert",
        ResearcherRole.BIO_RESEARCHER: "bio-expert",
    }

    # Role to RAG database mapping
    ROLE_RAG_MAP = {
        ResearcherRole.CHEM_RESEARCHER: "chem_rag",
        ResearcherRole.BIO_RESEARCHER: "bio_rag",
    }


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails"""
    pass


class AuthService:
    """Authentication and authorization service"""

    def __init__(self):
        self.config = AccessConfig()

    def authenticate(self, credentials: Credentials) -> ResearcherRole:
        """
        Authenticate user and return their role.

        Args:
            credentials: User credentials

        Returns:
            ResearcherRole: User's role

        Raises:
            AuthenticationError: If authentication fails
        """
        password = credentials.password

        if password not in self.config.PASSWORD_ROLE_MAP:
            raise AuthenticationError("Invalid credentials")

        role = self.config.PASSWORD_ROLE_MAP[password]
        return role

    def get_model_adapter(self, role: ResearcherRole) -> str:
        """
        Get the model adapter name for a given role.

        Args:
            role: User role

        Returns:
            str: Model adapter name
        """
        return self.config.ROLE_MODEL_MAP[role]

    def get_rag_database(self, role: ResearcherRole) -> str:
        """
        Get the RAG database path for a given role.

        Args:
            role: User role

        Returns:
            str: RAG database identifier
        """
        return self.config.ROLE_RAG_MAP[role]

    def validate_access(self, credentials: Credentials, required_role: Optional[ResearcherRole] = None) -> tuple[ResearcherRole, str, str]:
        """
        Validate access and return role, model adapter, and RAG database.

        Args:
            credentials: User credentials
            required_role: Optional specific role requirement

        Returns:
            tuple: (role, model_adapter, rag_database)

        Raises:
            AuthenticationError: If authentication fails
            AuthorizationError: If authorization fails
        """
        role = self.authenticate(credentials)

        if required_role and role != required_role:
            raise AuthorizationError(f"Access denied. Required role: {required_role}")

        model_adapter = self.get_model_adapter(role)
        rag_database = self.get_rag_database(role)

        return role, model_adapter, rag_database
