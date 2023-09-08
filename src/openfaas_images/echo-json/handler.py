import json


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    inp = json.loads(req)

    return inp
