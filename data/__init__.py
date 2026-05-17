import os
import dotenv

def convert_to_absolute_path(variable_name):
    """
    Converts a relative path from an environment variable to an absolute path.
    :param variable_name: The name of the environment variable containing the relative path.
    :return:
    """

    # Load environment variables from .env file
    dotenv.load_dotenv()
    # Get the relative path from the environment variable
    relative_path = os.environ.get(variable_name)

    if relative_path is None:
        raise ValueError(f"Environment variable '{variable_name}' not found.")

    # Get the absolute path to the project root (where .env is located)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Combine the project root with the relative path from the .env file
    absolute_path = os.path.join(project_root, relative_path)
    # Make sure it resolves to an absolute path
    absolute_path = os.path.abspath(absolute_path)

    return absolute_path