import requests


# Function to check food safety based on medical data
def check_food_safety(food_item, medical_data):
    # OpenFoodFacts API URL
    API_URL = "https://world.openfoodfacts.org/cgi/search.pl"

    # API parameters
    params = {
        "search_terms": food_item,
        "search_simple": 1,
        "action": "process",
        "json": 1,
    }

    try:
        # Make the API call
        response = requests.get(API_URL, params=params)

        # Check for a successful response
        if response.status_code == 200:
            data = response.json()

            # If no products are found
            if "products" not in data or len(data["products"]) == 0:
                print(f"No information found for the food item: {food_item}")
                return

            # Get the first result (assuming it's the most relevant)
            product = data["products"][0]
            food_label = product.get("product_name", "Unknown food")
            allergens_in_food = product.get("allergens_tags", [])
            ingredients_text = product.get("ingredients_text", "No ingredients info available.")

            # Check for allergens in the medical data
            allergens = medical_data.get("allergies", [])
            contains_allergen = any(allergen.lower() in allergens_in_food for allergen in allergens)

            # Print out food and ingredient information
            print(f"Food: {food_label}")
            print(f"Ingredients: {ingredients_text}")

            # Decision based on allergies
            if contains_allergen:
                print(f"{food_label} is NOT safe to eat due to the presence of allergens: {allergens_in_food}")
            else:
                print(f"{food_label} is safe to eat based on the allergy data provided.")
        else:
            print(f"API request failed with status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example input
food_item = ("pomegranate")
medical_data = {
    "allergies": ["peanuts", "gluten"],
}

# Call the function to check food safety
check_food_safety(food_item, medical_data)
