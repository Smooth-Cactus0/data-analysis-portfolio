"""
Synthetic Movie Review Generator v2.0 - Realistic difficulty for ~85-90% accuracy.
Features: shared vocabulary, negation patterns, neutral filler, label noise.
Author: Alexy Louis
"""
import random
import numpy as np
from typing import List, Tuple, Dict

class SyntheticReviewGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Strongly positive words
        self.strong_positive = ['masterpiece', 'brilliant', 'outstanding', 'excellent', 'phenomenal', 
            'incredible', 'amazing', 'superb', 'exceptional', 'magnificent', 'wonderful', 'fantastic',
            'loved', 'captivating', 'riveting', 'unforgettable', 'flawless', 'perfect', 'tremendous']
        
        # Strongly negative words
        self.strong_negative = ['terrible', 'awful', 'horrible', 'dreadful', 'atrocious', 'pathetic',
            'disaster', 'worst', 'waste', 'unbearable', 'painful', 'boring', 'tedious', 'hated',
            'disappointing', 'failed', 'garbage', 'trash', 'unwatchable', 'abysmal']
        
        # AMBIGUOUS - appear in both positive AND negative (key for realistic difficulty)
        self.ambiguous = ['interesting', 'different', 'unique', 'surprising', 'unexpected', 'intense',
            'dramatic', 'emotional', 'complex', 'simple', 'slow', 'fast', 'long', 'short', 'dark',
            'heavy', 'light', 'strange', 'weird', 'odd', 'familiar', 'classic', 'modern', 'old',
            'ambitious', 'bold', 'challenging', 'provocative', 'unconventional', 'artistic',
            'atmospheric', 'stylish', 'raw', 'gritty', 'intimate', 'epic', 'subtle', 'quiet',
            'loud', 'messy', 'polished', 'rough', 'smooth', 'uneven', 'consistent', 'mixed']
        
        # Mild positive
        self.mild_positive = ['good', 'nice', 'enjoyable', 'solid', 'decent', 'pleasant', 'fine',
            'likable', 'entertaining', 'fun', 'engaging', 'worthwhile', 'satisfying', 'charming']
        
        # Mild negative
        self.mild_negative = ['mediocre', 'bland', 'flat', 'weak', 'forgettable', 'generic',
            'predictable', 'cliche', 'uninspired', 'dull', 'lackluster', 'average', 'meh']
        
        # Neutral plot/context filler (no sentiment signal)
        self.neutral_filler = [
            "The film is set in the 1990s.", "It runs about two hours.", "The cast includes several newcomers.",
            "I watched this with friends.", "This was recommended to me.", "I saw this in theaters.",
            "The director previously worked on indie films.", "It's based on a true story.",
            "The soundtrack features original compositions.", "Shot on location in Europe.",
            "This is a sequel to the original.", "The runtime felt appropriate.",
            "I had no expectations going in.", "This came out last year.",
            "The film takes place over several decades.", "It features an ensemble cast.",
            "The cinematographer won awards for previous work.", "Production took three years.",
            "The script went through many revisions.", "It premiered at a major festival.",
            "Several scenes were improvised.", "The director used practical effects.",
            "Filming locations included multiple countries.", "The score was composed specifically.",
            "It draws from various influences.", "The story spans multiple timelines."
        ]
        
        # Negation patterns (confuse simple classifiers)
        self.negation_positive = [  # Sound negative but ARE positive
            "wasn't bad at all", "not disappointing", "never boring", "couldn't look away",
            "didn't disappoint", "not a waste of time", "never felt long", "wasn't the disaster I expected"
        ]
        self.negation_negative = [  # Sound positive but ARE negative
            "not as good as expected", "wasn't great", "didn't live up to the hype",
            "not particularly impressive", "couldn't get into it", "wasn't what I hoped for",
            "not worth the praise", "didn't work for me"
        ]
        
        # Templates by complexity
        self.templates = {
            'clear_positive': [
                "This movie was {strong_pos}. The {aspect} was {mild_pos} and I {rec_pos}.",
                "Absolutely {strong_pos}! {neutral} The {aspect} alone makes it {mild_pos}.",
            ],
            'clear_negative': [
                "This movie was {strong_neg}. The {aspect} was {mild_neg} and I {rec_neg}.",
                "Completely {strong_neg}. {neutral} The {aspect} was particularly {mild_neg}.",
            ],
            'subtle_positive': [
                "The {aspect} was {ambig} but ultimately {mild_pos}. {neutral}",
                "{neutral} Despite being {ambig}, it was {mild_pos} overall.",
                "Not {strong_neg}, just {ambig}. But still {mild_pos} enough.",
                "Some {mild_neg} parts, but the {aspect} was {mild_pos}. {neutral}",
            ],
            'subtle_negative': [
                "The {aspect} was {ambig} and ultimately {mild_neg}. {neutral}",
                "{neutral} Despite being {ambig}, it felt {mild_neg} overall.",
                "Not {strong_pos}, just {ambig}. And rather {mild_neg}.",
                "Some {mild_pos} parts, but the {aspect} was {mild_neg}. {neutral}",
            ],
            'ambiguous_positive': [
                "{neutral} Very {ambig}. Had {mild_neg} moments but {neg_pos}.",
                "The {aspect} felt {ambig}. {neutral} Still, somewhat {mild_pos}.",
                "{ambig} and {ambig}. {neutral} But I'd say {mild_pos}.",
                "Hard to describe. {ambig} yet {ambig}. Leaning {mild_pos}. {neutral}",
            ],
            'ambiguous_negative': [
                "{neutral} Very {ambig}. Had {mild_pos} moments but {neg_neg}.",
                "The {aspect} felt {ambig}. {neutral} Still, somewhat {mild_neg}.",
                "{ambig} and {ambig}. {neutral} But I'd say {mild_neg}.",
                "Hard to describe. {ambig} yet {ambig}. Leaning {mild_neg}. {neutral}",
            ],
        }
        
        self.aspects = ['acting', 'story', 'plot', 'direction', 'cinematography', 'script',
            'pacing', 'dialogue', 'characters', 'ending', 'soundtrack', 'visuals']
        self.rec_positive = ['highly recommend it', 'would watch again', 'loved every minute']
        self.rec_negative = ['cannot recommend', 'would not watch again', 'regret watching']
    
    def generate_review(self, sentiment: int, complexity: str = 'mixed') -> str:
        """Generate a single review. sentiment: 1=positive, 0=negative."""
        if complexity == 'mixed':
            complexity = random.choices(
                ['clear', 'subtle', 'ambiguous'],
                weights=[0.15, 0.40, 0.45]
            )[0]
        
        template_key = f"{complexity}_{'positive' if sentiment == 1 else 'negative'}"
        template = random.choice(self.templates[template_key])
        
        review = template.format(
            strong_pos=random.choice(self.strong_positive),
            strong_neg=random.choice(self.strong_negative),
            mild_pos=random.choice(self.mild_positive),
            mild_neg=random.choice(self.mild_negative),
            ambig=random.choice(self.ambiguous),
            neutral=random.choice(self.neutral_filler),
            aspect=random.choice(self.aspects),
            rec_pos=random.choice(self.rec_positive),
            rec_neg=random.choice(self.rec_negative),
            neg_pos=random.choice(self.negation_positive),
            neg_neg=random.choice(self.negation_negative),
        )
        return review
    
    def generate_dataset(self, n_samples: int, noise_rate: float = 0.12) -> Tuple[List[str], List[int]]:
        """Generate dataset with label noise for realism."""
        texts, labels = [], []
        for i in range(n_samples):
            true_label = i % 2  # Balanced classes
            texts.append(self.generate_review(true_label))
            # Add label noise (simulates annotation disagreement)
            if random.random() < noise_rate:
                labels.append(1 - true_label)
            else:
                labels.append(true_label)
        
        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        return list(texts), list(labels)


def create_imdb_like_dataset(n_train: int = 25000, n_test: int = 25000, seed: int = 42) -> Dict:
    """Create train/test split mimicking IMDB dataset structure."""
    gen = SyntheticReviewGenerator(seed=seed)
    train_texts, train_labels = gen.generate_dataset(n_train)
    gen_test = SyntheticReviewGenerator(seed=seed + 1000)
    test_texts, test_labels = gen_test.generate_dataset(n_test)
    
    return {
        'train': {'text': train_texts, 'label': train_labels},
        'test': {'text': test_texts, 'label': test_labels}
    }


if __name__ == "__main__":
    # Quick test
    dataset = create_imdb_like_dataset(n_train=100, n_test=100)
    print(f"Train: {len(dataset['train']['text'])} samples")
    print(f"Test: {len(dataset['test']['text'])} samples")
    print(f"\nSample positive:\n  {dataset['train']['text'][0]}")
    print(f"\nSample negative:\n  {dataset['train']['text'][1]}")
