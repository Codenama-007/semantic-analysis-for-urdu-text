import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


INPUT_FILE = "processed_quran.csv"
CATEGORY_FILE = "category_names.csv"


def split_categories(value):
    """
    Split categories only on commas outside parentheses.

    Example:
    Faith (Aqeedah), Eschatology (Akhirah)

    becomes:
    [
        "Faith (Aqeedah)",
        "Eschatology (Akhirah)"
    ]

    while:

    Supplication & Spirituality (Dua, Dhikr, Tazkiyah)

    remains ONE category.
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    categories = []
    current = []
    parentheses_level = 0

    for char in value:

        if char == "(":
            parentheses_level += 1

        elif char == ")":
            parentheses_level -= 1

        if char == "," and parentheses_level == 0:
            category = "".join(current).strip()

            if category:
                categories.append(category)

            current = []

        else:
            current.append(char)

    # Add final category
    category = "".join(current).strip()

    if category:
        categories.append(category)

    return categories

def normalize_category(category):
    """
    Normalize spelling and punctuation variations
    into the canonical category names.
    """

    category = category.strip()

    category_mapping = {

        # Divine Attributes
        "Divine Attributes & Signs":
            "Divine Attributes & Signs (Asma wa Sifat)",

        # Eschatology
        "Eschatology":
            "Eschatology (Akhirah)",

        # Ethics
        "Ethics & Morality (Akhlaak)":
            "Ethics & Morality (Akhlaq)",

        # Supplication
        "Supplication & Spiritality (Dua, Dhikr, Tazkiyah)":
            "Supplication & Spirituality (Dua, Dhikr, Tazkiyah)",

        "Supplication & Spiritivity (Dua, Dhikr, Tazkiyah)":
            "Supplication & Spirituality (Dua, Dhikr, Tazkiyah)",

        # Worship
        "Worship (‘Ibadah)'":
            "Worship (‘Ibadah)",

        "Worship (‘Ibadah))":
            "Worship (‘Ibadah)",

        "Worship (‘Ibadah)’":
            "Worship (‘Ibadah)",
    }

    return category_mapping.get(category, category)

def main():

    print("=" * 70)
    print("ENCODING URDU SEMANTIC CATEGORIES")
    print("=" * 70)

    # 1. Load preprocessed dataset

    df = pd.read_csv(INPUT_FILE)

    print(f"\nDataset loaded: {df.shape}")

    # 2. Convert Category column into lists

    df["Category_List"] = (
    df["Category"]
    .apply(split_categories)
    .apply(lambda categories: [
        normalize_category(category)
        for category in categories
    ])
)
    # Category frequency analysis
    category_counts = {}

    for categories in df["Category_List"]:

        for category in categories:

            if category not in category_counts:
                category_counts[category] = 0

            category_counts[category] += 1

    print("\nCategory Frequency:")
    print("--------------------------------")

    for category, count in sorted(
        category_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{count:4d} | {category}")

    print("\nExample category conversion:")
    print("--------------------------------")

    for i in range(5):
        print("Original:", df.iloc[i]["Category"])
        print("List:    ", df.iloc[i]["Category_List"])
        print()
        
    # 3. Multi-label encoding

    mlb = MultiLabelBinarizer()

    category_matrix = mlb.fit_transform(
        df["Category_List"]
    )

    # Create dataframe containing 0/1 labels
    category_df = pd.DataFrame(
        category_matrix,
        columns=mlb.classes_,
        index=df.index
    )
    
    # 4. Add encoded categories to dataset

    df = pd.concat(
        [df, category_df],
        axis=1
    )

    # 5. Save final dataset

    df.to_csv(
        "final_dataset.csv" ,
        index=False,
        encoding="utf-8-sig"
    )

    # 6. Save category names separately

    pd.Series(mlb.classes_).to_csv(
        CATEGORY_FILE,
        index=False,
        header=False,
        encoding="utf-8-sig"
    )

    # 7. Display categories

    print("\nSemantic Categories:")
    print("--------------------------------")

    for index, category in enumerate(mlb.classes_):
        print(f"{index}: {category}")

    print("\nTotal categories:", len(mlb.classes_))
    
    # 8. Show encoded example

    print("\nExample encoded verse:")
    print("--------------------------------")

    example = df.iloc[0]

    print("Urdu:")
    print(example["Urdu"])

    print("\nOriginal Category:")
    print(example["Category"])

    print("\nEncoded Labels:")

    for category in mlb.classes_:

        if example[category] == 1:
            print("✓", category)

    print("\n" + "=" * 70)
    print("CATEGORY ENCODING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()