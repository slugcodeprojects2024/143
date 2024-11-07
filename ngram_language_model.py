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

# Calculate unsmoothed probabilities
def calculate_probabilities(model):
    probabilities = {}
    for ngram, next_tokens in model.items():
        total_count = sum(next_tokens.values())
        probabilities[ngram] = {token: count / total_count for token, count in next_tokens.items()}
    return probabilities

# Interpolated probability function with lambda weights
def interpolated_probability(unigram_probs, bigram_probs, trigram_probs, ngram, token, lambdas):
    lambda1, lambda2, lambda3 = lambdas

    # Unigram probability
    unigram_prob = unigram_probs.get((), {}).get(token, 1e-10)

    # Bigram probability (use last token if available)
    bigram_prob = bigram_probs.get(ngram[-1:], {}).get(token, 1e-10) if len(ngram) > 0 else unigram_prob

    # Trigram probability (use last two tokens if available)
    trigram_prob = trigram_probs.get(ngram, {}).get(token, 1e-10) if len(ngram) > 1 else bigram_prob

    # Return weighted combination
    return lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob

# Calculate perplexity using interpolation with lambda weights
def calculate_perplexity_with_interpolation(sentences, unigram_probs, bigram_probs, trigram_probs, lambdas):
    log_prob_sum = 0
    token_count = 0
    for sentence in sentences:
        sentence = ['<START>', '<START>'] + sentence  # Add <START> tokens for context in bigram/trigram models
        for i in range(2, len(sentence)):
            ngram = tuple(sentence[i-2:i])
            token = sentence[i]
            probability = interpolated_probability(unigram_probs, bigram_probs, trigram_probs, ngram, token, lambdas)
            log_prob_sum += math.log2(probability)
            token_count += 1
    return 2 ** (-log_prob_sum / token_count)

# Main function to evaluate on train, dev, and test datasets
def main():
    # Load datasets
    train_sentences = load_data('1b_benchmark.train.tokens')
    dev_sentences = load_data('1b_benchmark.dev.tokens')
    test_sentences = load_data('1b_benchmark.test.tokens')
    
    # Preprocess training data to create vocabulary
    train_sentences, vocab = preprocess_data(train_sentences)
    
    # Preprocess dev and test data using the same vocabulary
    dev_sentences, _ = preprocess_data(dev_sentences, vocab)
    test_sentences, _ = preprocess_data(test_sentences, vocab)

    # Build unigram, bigram, and trigram models from training data
    unigram_model = build_ngram_model(train_sentences, 1)
    bigram_model = build_ngram_model(train_sentences, 2)
    trigram_model = build_ngram_model(train_sentences, 3)

    # Calculate unsmoothed probabilities for each model
    unigram_probs = calculate_probabilities(unigram_model)
    bigram_probs = calculate_probabilities(bigram_model)
    trigram_probs = calculate_probabilities(trigram_model)

    # Define sets of lambda values to try
    lambda_sets = [
        (0.1, 0.3, 0.6),
        (0.3, 0.3, 0.4),
        (0.2, 0.5, 0.3),
        (0.4, 0.4, 0.2),
        (0.6, 0.2, 0.2)
    ]
    
    # Evaluate each lambda set on train and dev sets
    for lambdas in lambda_sets:
        print(f"Lambda values: λ1={lambdas[0]}, λ2={lambdas[1]}, λ3={lambdas[2]}")
        
        train_perplexity = calculate_perplexity_with_interpolation(train_sentences, unigram_probs, bigram_probs, trigram_probs, lambdas)
        dev_perplexity = calculate_perplexity_with_interpolation(dev_sentences, unigram_probs, bigram_probs, trigram_probs, lambdas)
        
        print(f"  Train Perplexity: {train_perplexity}")
        print(f"  Dev Perplexity: {dev_perplexity}\n")

    # Select best lambda values from the dev set results, e.g., (0.1, 0.3, 0.6) based on performance
    best_lambdas = (0.1, 0.3, 0.6)  # Adjust based on your chosen best lambda set
    test_perplexity = calculate_perplexity_with_interpolation(test_sentences, unigram_probs, bigram_probs, trigram_probs, best_lambdas)
    
    print(f"Best Lambda values on Dev Set: λ1={best_lambdas[0]}, λ2={best_lambdas[1]}, λ3={best_lambdas[2]}")
    print(f"Test Perplexity with Best Lambda values: {test_perplexity}")

if __name__ == '__main__':
    main()
