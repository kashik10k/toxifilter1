from data_augmentation import augment_toxic_list

toxic_words = ['sex', 'porn', 'fuck', 'bitch', 'adult', 'nude']

augmented = augment_toxic_list(toxic_words, max_variants=3)

print("Generated obfuscated samples:")
for word in augmented:
    print(word)
