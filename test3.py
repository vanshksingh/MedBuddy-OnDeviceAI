import csv

# Create a dictionary to store drug interactions
drug_interactions = {}


# Load CSV file and populate the dictionary
def load_interactions_from_csv(file_path):
    """
    Load drug interactions from a CSV file and store them in a dictionary.

    :param file_path: The path to the CSV file.
    """
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            drug1 = row['drug1_name'].strip()
            drug2 = row['drug2_name'].strip()
            interaction = row['interaction_type'].strip()

            # Store interaction both ways (drug1-drug2 and drug2-drug1)
            drug_interactions[(drug1, drug2)] = interaction
            drug_interactions[(drug2, drug1)] = interaction


# Function to check interaction between two drugs
def find_interaction(drug1_name, drug2_name):
    """
    Find the interaction between two drugs.

    :param drug1_name: Name of the first drug.
    :param drug2_name: Name of the second drug.
    :return: Interaction type or 'No known interaction' if not found.
    """
    drug1_name = drug1_name.strip()
    drug2_name = drug2_name.strip()

    interaction = drug_interactions.get((drug1_name, drug2_name), None)

    if interaction:
        return f"The interaction between {drug1_name} and {drug2_name} is: {interaction}."
    else:
        return f"No known interaction between {drug1_name} and {drug2_name}."


# Load interactions from the CSV file
csv_file_path = "/Users/vanshkumarsingh/Downloads/data of multiple-type drug-drug interactions/DDI_data.csv"  # Replace with your actual CSV file path
load_interactions_from_csv(csv_file_path)

# Example usage
drug1 = "Bivalirudin"
drug2 = "Apixaban"
print(find_interaction(drug1, drug2))

drug1 = "Bivalirudin"
drug2 = "Lovastatin"
print(find_interaction(drug1, drug2))

drug1 = "Bivalirudin"
drug2 = "UnknownDrug"
print(find_interaction(drug1, drug2))
