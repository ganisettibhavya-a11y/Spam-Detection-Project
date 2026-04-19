import pandas as pd
import numpy as np

# --- 1. Load SMS Spam Collection Dataset (Using the common tab-separated format) ---
# Assuming 'spam.csv' is the UCI SMS Spam Collection Dataset (v1=label, v2=text)
try:
    # Use sep='\t' and names=['label', 'text_content'] to fix the common 'v1', 'v2' error
    sms_df = pd.read_csv('spam.csv', sep='\t', names=['label', 'text_content'], encoding='latin-1')
    
    # Remove any completely empty rows that might have been loaded
    sms_df.dropna(subset=['text_content'], inplace=True) 
    
    # Add a column to track content type
    sms_df['type'] = 'SMS' 
    
    # Filter to get a balanced portion, we need > 2000 total unique rows.
    # The original has ~4825 'ham' and ~747 'spam'. We will use all unique spam and sample from ham.
    spam_sms = sms_df[sms_df['label'] == 'spam'].drop_duplicates(subset=['text_content'])
    ham_sms_candidates = sms_df[sms_df['label'] == 'ham'].drop_duplicates(subset=['text_content'])
    
    # Calculate how many 'ham' messages we need to reach over 2000 total
    required_ham_count = 2100 - spam_sms.shape[0] 
    
    # Ensure we don't sample more than available ham
    sample_size = min(required_ham_count, ham_sms_candidates.shape[0])
    
    ham_sms = ham_sms_candidates.sample(n=sample_size, random_state=42)
    
    final_df = pd.concat([ham_sms, spam_sms])
    
except FileNotFoundError:
    print("SMS spam.csv not found. Please download the UCI SMS Spam Collection and place it in the same directory.")
    # Fallback to creating a synthetic dataset if the file is missing
    final_df = None

# --- 2. Synthetic Dataset Fallback (If the file was not loaded or is too small) ---
# If the file wasn't loaded successfully, we generate a synthetic one of 2000+ rows
if final_df is None or final_df.shape[0] < 2000:
    print(f"\nCreating a synthetic dataset of 2200 rows as fallback...")
    
    # Base Content for Spam and Ham (Expand these lists for better quality)
    spam_templates = [
        "URGENT! Your bank account requires immediate verification. Click the link NOW: http://scam.link/{} to avoid suspension. Reply STOP to opt out.",
        "Congratulations! You've won a FREE prize money of $10000. Text WINNER to 88888 to claim. Offer ends today!",
        "Click here for massive savings on your next purchase! Limited time offer, don't miss out! Link: http://deals.com/{}",
    ]
    ham_templates = [
        "Hi, can we reschedule our meeting to 3 PM tomorrow? Let me know if that works for you.",
        "I've attached the final report for the project. Please review it by end of day.",
        "Just a reminder that your appointment is set for next Tuesday at 9:00 AM.",
    ]

    np.random.seed(42)
    data = []
    
    # Generate 1100 Ham messages
    for i in range(1100):
        template = np.random.choice(ham_templates)
        message = template.format(i) if '{}' in template else template
        data.append({'text_content': message + f' ({i})', 'label': 0, 'type': 'SYNTHETIC_HAM'})

    # Generate 1100 Spam messages
    for i in range(1100):
        template = np.random.choice(spam_templates)
        # Use a random number/code to ensure content uniqueness
        unique_id = np.random.randint(10000, 99999) 
        message = template.format(unique_id)
        data.append({'text_content': message, 'label': 1, 'type': 'SYNTHETIC_SPAM'})
    
    final_df = pd.DataFrame(data)
    final_df.drop_duplicates(subset=['text_content'], inplace=True)
    
# --- 3. Finalize and Save Dataset ---
if final_df is not None:
    # Ensure the required columns and labels are correct: 0 for ham/normal, 1 for spam
    final_df['label'] = final_df['label'].replace({'ham': 0, 'spam': 1})
    final_df = final_df[['text_content', 'label']].drop_duplicates()
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True) 

    # Print details
    print(f"\nFinal Dataset Size: {final_df.shape[0]} unique rows")
    print(f"Spam Count: {final_df['label'].sum()}")
    print(f"Normal (Ham) Count: {final_df.shape[0] - final_df['label'].sum()}")

    # Save to CSV
    output_filename = 'spam_detection_dataset.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\nDataset successfully generated and saved to '{output_filename}'")
else:
    print("\nCould not generate the final dataset. Please check your 'spam.csv' file.")