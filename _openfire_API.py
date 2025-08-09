import json
import requests




REST_API_URL = "http://localhost:9090/plugins/restapi/v1/users"
SECRET_KEY = "abdoubvb"  


def create_user(username, password, name=None, email=None):
    """Create a user on the Openfire server via REST API."""
    headers = {
        "Authorization": SECRET_KEY,
        "Content-Type": "application/json",
    }
    user_data = {
        "username": username,
        "password": password,
    }

    if name:
        user_data["name"] = name
    if email:
        user_data["email"] = email

    response = requests.post(REST_API_URL, headers=headers, data=json.dumps(user_data))
    if response.status_code == 201:
        print(f"User {username} created successfully.")
    elif response.status_code == 409:
        pass
        #print(f"User {username} already exists.")
    else:
        print(f"Failed to create user {username}. Response: {response.text}")