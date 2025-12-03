import csv
import os

TEST_FILE_PATH = "../queries/"
TEST_PROMPT_CSV = "prompts.csv"
TEST_IMAGE_PATH = TEST_FILE_PATH + "images/"


def read_from_queries():
    data_list = []
    prompt_filename = TEST_FILE_PATH + TEST_PROMPT_CSV
    # Check if file exists to avoid crashing
    if not os.path.exists(prompt_filename):
        print(f"Error: The file '{prompt_filename}' does not exist.")
        return []

    try:
        with open(prompt_filename, mode='r', encoding='utf-8') as file:

            csv_reader = csv.DictReader(file)

            # Convert the reader object to a standard Python list
            data_list = list(csv_reader)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return data_list


def main():
    test_prompts = read_from_queries()

    for row in test_prompts:
        print("===========================================================")
        for key in row: 
            print("key:", key)
            print("Value:", row[key])
        print("===========================================================")


if __name__ == '__main__':
    main()