import jwt
import requests
from fastapi import HTTPException, Request
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.security import OAuth2
from typing import Optional
from core.config import settings
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers


class Oauth2ClientCredentials(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        client_id: str,
        scheme_name: str = None,
        scopes: dict = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(clientCredentials={"tokenUrl": tokenUrl, "scopes": scopes})
        self.client_id = client_id
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.headers.get("Authorization")
        scheme, token = get_authorization_scheme_param(authorization)
        
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=401,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None

        # Verify the JWT token
        try:
            # Get the public keys from Microsoft
            jwks_url = f"https://login.microsoftonline.com/{settings.CABI_TenantId}/discovery/v2.0/keys"
            jwks = requests.get(jwks_url).json()

            # Decode and validate the token
            unverified_header = jwt.get_unverified_header(token)
            rsa_key = self.get_rsa_key(jwks, unverified_header)

            # unverified_payload = jwt.decode(token, options={"verify_signature": False})

            if rsa_key:
                public_key = self.convert_jwk_to_pem(rsa_key)
                payload = jwt.decode(
                    token,
                    public_key,
                    algorithms=["RS256"],
                    audience=f"api://{self.client_id}",  # This will check the 'aud' claim
                    issuer=f"https://sts.windows.net/{settings.CABI_TenantId}/"
                )
                
                # Optionally validate other claims if needed
                self.verify_additional_claims(payload)

                return payload
            else:
                raise HTTPException(status_code=401, detail="Invalid token")

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidAudienceError as e:
            raise HTTPException(status_code=401, detail=f"Invalid audience: {str(e)}")
        except jwt.InvalidIssuerError as e:
            raise HTTPException(status_code=401, detail=f"Invalid issuer: {str(e)}")
        except jwt.InvalidTokenError as e:  # General token validation error
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Token validation error: {str(e)}")

    def get_rsa_key(self, jwks, unverified_header):
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                return key
        return None

    def convert_jwk_to_pem(self, jwk):
        public_numbers = RSAPublicNumbers(
            e=int.from_bytes(self.base64url_decode(jwk['e']), 'big'),
            n=int.from_bytes(self.base64url_decode(jwk['n']), 'big')
        )
        public_key = public_numbers.public_key(default_backend())
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def base64url_decode(self, input):
        input += '=' * (4 - (len(input) % 4))  # Pad with "=" to make the length a multiple of 4
        return jwt.utils.base64url_decode(input)

    def verify_additional_claims(self, payload):
        if "roles" not in payload or "api.read" not in payload["roles"]:
            raise HTTPException(status_code=403, detail="Insufficient role in token")


def get_auth_scheme():
    return Oauth2ClientCredentials(
        tokenUrl=f"https://login.microsoftonline.com/{settings.CABI_TenantId}/oauth2/v2.0/token",
        client_id=settings.EVA_API_ClientId,
        scopes={f'api://{settings.EVA_API_ClientId}/.default': 'api.read'}
    )

auth_scheme = get_auth_scheme()

