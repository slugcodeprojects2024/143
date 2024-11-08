import math
from collections import defaultdict, Counter

# Load data function
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip().split() + ['<STOP>'] for sentence in sentences]

# Preprocess data, replacing tokens with <UNK> if they appear fewer than 3 times in training data
def preprocess_data(sentences, vocab=None):
    if vocab is None:  # Creating vocabulary during training
        token_counts = Counter([token for sentence in sentences for token in sentence])
        vocab = {token for token, count in token_counts.items() if count >= 3}
        vocab.add('<UNK>')
        vocab.add('<STOP>')
        vocab = sorted(vocab)

    def replace_oov_words(sentence):
        return [token if token in vocab else '<UNK>' for token in sentence]

    return [replace_oov_words(sentence) for sentence in sentences], vocab

# Build n-gram model
def build_ngram_model(sentences, n):
    model = defaultdict(lambda: defaultdict(int))
    for sentence in sentences:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i + n - 1])
            next_token = sentence[i + n - 1]
            model[ngram][next_token] += 1
    return model

# Calculate probabilities with Laplace smoothing
def calculate_probabilities_with_laplace_smoothing(model, vocab_size, smoothing=1):
    probabilities = {}
    for ngram, next_tokens in model.items():
        total_count = sum(next_tokens.values()) + smoothing * vocab_size  # Apply Laplace smoothing
        probabilities[ngram] = {token: (count + smoothing) / total_count for token, count in next_tokens.items()}
    return probabilities

# Calculate perplexity with Laplace smoothing
def calculate_perplexity_with_smoothing(sentences, probabilities, n, vocab_size, smoothing=1):
    log_prob_sum = 0
    token_count = 0

    for sentence in sentences:
        sentence = ['<START>'] * (n - 1) + sentence  # Add <START> tokens for bigram and trigram models
        for i in range(n - 1, len(sentence)):
            ngram = tuple(sentence[i - n + 1:i])
            next_token = sentence[i]

            # Retrieve smoothed probability or apply smoothing for unseen n-grams
            probability = probabilities.get(ngram, {}).get(next_token, smoothing / (smoothing * vocab_size))

            log_prob_sum += math.log2(probability)
            token_count += 1

    return 2 ** (-log_prob_sum / token_count)

# Calculate probabilities without smoothing
def calculate_probabilities(model):
    probabilities = {}
    for ngram, next_tokens in model.items():
        total_count = sum(next_tokens.values())
        probabilities[ngram] = {token: count / total_count for token, count in next_tokens.items()}
    return probabilities

# Calculate perplexity without smoothing
def calculate_perplexity(sentences, probabilities, n):
    log_prob_sum = 0
    token_count = 0
    for sentence in sentences:
        sentence = ['<START>'] * (n - 1) + sentence
        for i in range(n - 1, len(sentence)):
            ngram = tuple(sentence[i - n + 1:i])
            next_token = sentence[i]

            # Use probability if it exists, otherwise assign a small fallback probability
            probability = probabilities.get(ngram, {}).get(next_token, 1e-10)
            log_prob_sum += math.log2(probability)
            token_count += 1

    return 2 ** (-log_prob_sum / token_count)

# Main function to evaluate on train, dev, and test datasets
def main():
    # Load datasets
    train_sentences = load_data('1b_benchmark.train.tokens')
    dev_sentences = load_data('1b_benchmark.dev.tokens')
    test_sentences = load_data('1b_benchmark.test.tokens')  # Only used for final evaluation
    
    # Preprocess training data to create vocabulary
    train_sentences, vocab = preprocess_data(train_sentences)
    vocab_size = len(vocab)

    # Preprocess dev data using the same vocabulary
    dev_sentences, _ = preprocess_data(dev_sentences, vocab)

    # Build and evaluate models on train and dev datasets
    for n in [1, 2, 3]:
        # Build n-gram model
        model = build_ngram_model(train_sentences, n)
        
        # Calculate probabilities without smoothing
        probabilities = calculate_probabilities(model)
        
        # Calculate and display perplexity without smoothing
        train_perplexity = calculate_perplexity(train_sentences, probabilities, n)
        dev_perplexity = calculate_perplexity(dev_sentences, probabilities, n)
        
        print(f'{n}-gram Model Perplexity without Smoothing:')
        print(f'  Train: {train_perplexity}')
        print(f'  Dev: {dev_perplexity}\n')

        # Calculate probabilities with Laplace smoothing
        smoothed_probabilities = calculate_probabilities_with_laplace_smoothing(model, vocab_size, smoothing=1)
        
        # Calculate and display perplexity with smoothing
        train_perplexity_smooth = calculate_perplexity_with_smoothing(train_sentences, smoothed_probabilities, n, vocab_size, smoothing=1)
        dev_perplexity_smooth = calculate_perplexity_with_smoothing(dev_sentences, smoothed_probabilities, n, vocab_size, smoothing=1)
        
        print(f'{n}-gram Model Perplexity with Laplace Smoothing:')
        print(f'  Train: {train_perplexity_smooth}')
        print(f'  Dev: {dev_perplexity_smooth}\n')

    # Uncomment the following lines only for final evaluation on the test set
    test_sentences, _ = preprocess_data(test_sentences, vocab)
    for n in [1, 2, 3]:
        smoothed_probabilities = calculate_probabilities_with_laplace_smoothing(model, vocab_size, smoothing=1)
        test_perplexity_smooth = calculate_perplexity_with_smoothing(test_sentences, smoothed_probabilities, n, vocab_size, smoothing=1)
        
        print(f'{n}-gram Model Perplexity with Laplace Smoothing on Test Set:')
        print(f'  Test: {test_perplexity_smooth}\n')

if __name__ == '__main__':
    main()
