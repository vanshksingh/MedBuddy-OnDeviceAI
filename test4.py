import urllib.parse

def generate_search_url(query):
    base_url = "https://www.google.com/search?q="
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"{base_url}{encoded_query}"
    return search_url

# Example usage:
query = "pomegranate with diabetes safe or not"
url = generate_search_url(query)
print(url)





