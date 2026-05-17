import os
import re
import requests
from bs4 import BeautifulSoup


def download_historical_reports(destination_folder: os.PathLike, base_url="https://reports-public.ieso.ca/public/Demand/"):
    """
    Scrapes the IESO public Demand directory page, finds CSV links for
    non-versioned reports (i.e., those NOT containing '_v'), and downloads them.

    Args:
        base_url (str): The URL of the IESO Demand directory index page.
        destination_folder (str): Local folder to save the CSV files.
    """


    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        print(f"Destination folder '{destination_folder}' does not exist. Creating it.")
        os.makedirs(destination_folder)

    # Ensure local destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # 1) Request the index page
    response = requests.get(base_url)
    response.raise_for_status()

    # 2) Parse the HTML to find CSV links
    soup = BeautifulSoup(response.text, "html.parser")

    # Regex to match CSV links that do NOT contain _v
    # e.g. "PUB_Demand_2002.csv", "PUB_Demand_2025.csv", "PUB_Demand.csv"
    # We exclude versioned ones like "PUB_Demand_2002_v1.csv"
    csv_pattern = re.compile(r"(PUB_Demand(?:_\d{4})?\.csv)$", re.IGNORECASE)

    # 3) Loop through all <a> tags
    for link_tag in soup.find_all("a", href=True):
        href = link_tag["href"]
        filename_match = csv_pattern.search(href)
        if filename_match:
            # If the link is relative, combine it with base_url
            # The link in the HTML might be absolute already, but let's be safe
            csv_url = href
            if not csv_url.startswith("http"):
                csv_url = base_url + csv_url

            # Extract the CSV filename from the URL
            csv_filename = os.path.basename(csv_url)

            print(f"Found CSV: {csv_filename} -> {csv_url}")

            # 4) Download and save the file
            local_path = os.path.join(destination_folder, csv_filename)
            if not os.path.exists(local_path) or '2025' in local_path:
                print(f"Downloading {csv_filename}...")
                try:
                    file_resp = requests.get(csv_url, stream=True)
                    file_resp.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in file_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Saved to {local_path}\n")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {csv_filename}: {e}\n")
            else:
                print(f"Already exists, skipping: {local_path}\n")


if __name__ == "__main__":
    import dotenv
    import os

    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Get the relative path from the environment variable
    relative_path = os.environ.get("HISTORICAL_CSVS")

    if relative_path:
        # Get the absolute path to the project root (where .env is located)
        project_root = os.path.dirname(os.path.abspath(__file__))  # This gives the current script directory

        # If your script is in a subdirectory, you can get the project root like this:
        # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Go to parent directory

        # Combine the project root with the relative path from the .env file
        absolute_path = os.path.join(project_root, relative_path)
        absolute_path = os.path.abspath(absolute_path)  # Make sure it resolves to an absolute path

        print(f"Resolved absolute path: {absolute_path}")

        # Call the function with the absolute path
        download_historical_reports(absolute_path)
