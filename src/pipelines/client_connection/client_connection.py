# client_connection.py
import kfp
from client_auth import get_istio_auth_session  # Replace with your actual import

KUBEFLOW_ENDPOINT = "http://localhost:8080"
KUBEFLOW_USERNAME = "user@example.com"
KUBEFLOW_PASSWORD = "12341234"

def create_kfp_client():
    # Get the authentication session
    auth_session = get_istio_auth_session(
        url=KUBEFLOW_ENDPOINT,
        username=KUBEFLOW_USERNAME,
        password=KUBEFLOW_PASSWORD
    )

    # Create a Kubeflow Pipelines client
    client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline", cookies=auth_session["session_cookie"])
    return client
