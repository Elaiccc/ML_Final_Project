# Please remember to run "!pip install --upgrade openai pandas --quiet" in terminal before run this code

import os
import openai
import pandas as pd
import re
import time
from tqdm import tqdm

# Set your API key securely; avoid hardcoding in production!
os.environ["OPENAI_API_KEY"] = "Set your own api key here"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Update this to the correct local path
REAL_CSV = "True.csv"
df = pd.read_csv(REAL_CSV)


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9,.!? ]+", "", s)
    return s.strip()


if "cleaned_text" not in df.columns:
    df["cleaned_text"] = df["text"].apply(clean_text)

# Set how many rows you want to generate, here is there will generate 10000 rows
N = 10000
df = df.head(N).reset_index(drop=True)

def generate_fake(text: str) -> str:
    for attempt in range(3):
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative news writer."},
                    {"role": "user", "content":
                        "Rewrite the following real news article as a plausible but false news article, "
                        "keeping the same style and approximate length:\n\n" + text}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error (attempt {attempt + 1}): {e}")
            time.sleep(5 * (2 ** attempt))
    return ""


fake_texts = []
chunk_size = 1000

for i, txt in enumerate(tqdm(df["cleaned_text"], total=N)):
    fake = generate_fake(txt)
    fake_texts.append(fake)

    if (i + 1) % chunk_size == 0:
        partial = df.loc[:i, ["cleaned_text"]].copy()
        partial["fake_openai"] = fake_texts
        partial.to_csv(f"fakes_partial_{i + 1}.csv", index=False)
        print(f"Saved interim results for {i + 1} articles")

df["fake_openai"] = fake_texts
df.to_csv("sample_with_synthetic_fakes_10000.csv", index=False)