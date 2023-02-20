import json
import os



def prepare_data(path2data):
    """
    Prepare the list of dictionaries each dictionary with image and label keys 
    having path to image and label as correponding values
    """
    # Loading dataset.json file that contains metadata about dataset
    with open(os.path.join(path2data, "Task09_Spleen", "dataset.json")) as f:
        json_dataset = json.load(f)
        f.close()

    # Displaying the attributes in json file
    print(f"Attributes in json file: {json_dataset.keys()}\n")

    # "training" attribute in json file contains the list of dictionaries with image and label names
    print(f"json_dataset['training'][0] : {json_dataset['training'][0]}\n")

    # Creating data dictionary with paths to images and corresponding labels 
    data_dicts = [
        {"image": os.path.normpath(os.path.join(path2data, "Task09_Spleen", data["image"])), 
        "label": os.path.normpath(os.path.join(path2data, "Task09_Spleen", data["label"]))}
        for data in json_dataset["training"]
    ]

    # Visualizing one sample from our data_dict
    print(f"data_dicts[0] : {data_dicts[0]}\n")

    return data_dicts