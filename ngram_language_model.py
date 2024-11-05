import math
from collections import defaultdict, Counter

# Step 1: Load and preprocess data
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip().split() + ['<STOP>'] for sentence in sentences]


def preprocess_data(sentences):
    token_counts = Counter([token for sentence in sentences for token in sentence])
    vocab = {token for token, count in token_counts.items() if count >= 3}
    vocab.add('<UNK>')  # Add <UNK> token
    vocab.add('<STOP>')  # Add <STOP> token
    vocab = sorted(vocab)
    
    def replace_oov_words(sentence):
        return [token if token in vocab else '<UNK>' for token in sentence]

    return [replace_oov_words(sentence) for sentence in sentences], vocab

# Step 2: Build n-gram models
def build_ngram_model(sentences, n):
    model = defaultdict(lambda: defaultdict(int))
    for sentence in sentences:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i + n - 1])
            next_token = sentence[i + n - 1]
            model[ngram][next_token] += 1
    return model

# Step 3: Calculate probabilities
def calculate_probabilities(model):
    probabilities = {}
    for ngram, next_tokens in model.items():
        total_count = sum(next_tokens.values())
        probabilities[ngram] = {token: count / total_count for token, count in next_tokens.items()}
    return probabilities

# Step 4: Calculate perplexity
def calculate_perplexity(sentences, probabilities, n):
    log_prob_sum = 0
    token_count = 0
    for sentence in sentences:
        sentence = ['<START>'] * (n - 1) + sentence  # Add <START> tokens for bigram and trigram
        for i in range(n - 1, len(sentence)):
            ngram = tuple(sentence[i - n + 1:i])
            next_token = sentence[i]
            probability = probabilities.get(ngram, {}).get(next_token, 1e-10)  # Small value for zero probabilities
            log_prob_sum += math.log2(probability)
            token_count += 1
    return 2 ** (-log_prob_sum / token_count)

# Step 5: Run experiments and report results
def main():
    # Load data
    train_sentences = load_data('1b_benchmark.train.tokens')
    dev_sentences = load_data('1b_benchmark.dev.tokens')
    test_sentences = load_data('1b_benchmark.test.tokens')

    # Preprocess data
    train_sentences, vocab = preprocess_data(train_sentences)
    dev_sentences, _ = preprocess_data(dev_sentences)
    test_sentences, _ = preprocess_data(test_sentences)

    # Build and evaluate models
    for n in [1, 2, 3]:
        model = build_ngram_model(train_sentences, n)
        probabilities = calculate_probabilities(model)
        
        train_perplexity = calculate_perplexity(train_sentences, probabilities, n)
        dev_perplexity = calculate_perplexity(dev_sentences, probabilities, n)
        test_perplexity = calculate_perplexity(test_sentences, probabilities, n)
        
        print(f'{n}-gram Model Perplexity:')
        print(f'  Train: {train_perplexity}')
        print(f'  Dev: {dev_perplexity}')
        print(f'  Test: {test_perplexity}\n')

if __name__ == '__main__':
    main()
