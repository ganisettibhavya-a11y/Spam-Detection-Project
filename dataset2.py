import pandas as pd
import numpy as np

# A list of legitimate and common domains
common_domains = [
    "google.com", "microsoft.com", "github.com", "amazon.com", "wikipedia.org",
    "facebook.com", "twitter.com", "linkedin.com", "youtube.com", "instagram.com",
    "reddit.com", "cnn.com", "bbc.co.uk", "nytimes.com", "stackoverflow.com",
    "apple.com", "paypal.com", "ebay.com", "netflix.com", "spotify.com",
    "yahoo.com", "bing.com", "duckduckgo.com", "cloudflare.com", "adobe.com"
]

# A list of suspicious keywords and characters for phishing URLs
phishing_keywords = [
    "verify", "login", "update", "security", "account", "confirm", "bank",
    "free", "prize", "winner", "urgent", "password", "reset", "invoice",
    "clickhere", "download", "document", "online", "payment", "upgrade"
]
phishing_chars = [
    ".", "-", "_", "~", "@", "!", "$", "%", "&", "*", "(", ")", "="
]

def generate_legitimate_url():
    """Generates a random legitimate URL."""
    domain = np.random.choice(common_domains)
    path = ""
    # Add a random number of path segments
    for _ in range(np.random.randint(0, 3)):
        path += "/" + np.random.choice(["blog", "support", "help", "about", "products"])
    # Add optional parameters
    if np.random.rand() > 0.5:
        path += "?" + np.random.choice(["q=search_term", "id=12345", "cat=electronics"])
    return f"https://www.{domain}{path}"

def generate_phishing_url():
    """Generates a random phishing URL."""
    domain_part = np.random.choice(common_domains).split('.')[0]
    keyword = np.random.choice(phishing_keywords)
    char = np.random.choice(phishing_chars)
    # Different phishing patterns
    patterns = [
        f"http://{domain_part}-{keyword}.{np.random.choice(['com', 'net', 'org'])}",
        f"https://{keyword}{char}{domain_part}.{np.random.choice(['com', 'net', 'org'])}",
        f"http://www.{domain_part}.{keyword}.{np.random.choice(['com', 'net', 'org'])}",
        f"https://{domain_part}.{np.random.choice(['xyz', 'online', 'store', 'shop'])}/{keyword}"
    ]
    return np.random.choice(patterns)

def create_dataset(num_rows=2000):
    """Creates a DataFrame with a mix of legitimate and phishing URLs."""
    data = []
    # Ensure an equal split of legitimate and phishing URLs
    for i in range(num_rows):
        if i < num_rows / 2:
            url = generate_legitimate_url()
            label = "legitimate"
        else:
            url = generate_phishing_url()
            label = "phishing"
        data.append({'url': url, 'label': label})

    # Shuffle the dataset to ensure labels are not in a sequential order
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Create the dataset
    df = create_dataset(num_rows=2000)
    
    # Save the dataset to a CSV file
    file_path = "phishing_dataset.csv"
    df.to_csv(file_path, index=False)
    print(f"Dataset with {len(df)} rows created and saved to '{file_path}'.")
    
    # Print the first 5 rows to show the data format
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
