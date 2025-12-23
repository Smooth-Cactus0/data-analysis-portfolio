"""
Generate visualizations for the RAG system evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_retrieval_scores_chart():
    """Create horizontal bar chart of retrieval scores by query."""

    # Data from evaluation
    queries = [
        "What is RAG?",
        "How do transformers work?",
        "What are large language models?",
        "What are word embeddings?",
        "What is a vector database?",
        "What is deep learning?",
        "What is NLP?",
        "What is machine learning?"
    ]

    scores = [1.25, 0.99, 0.63, 0.63, 0.58, 0.54, 0.54, 0.44]

    # Sort by score for better visualization
    sorted_pairs = sorted(zip(scores, queries), reverse=False)
    scores_sorted = [s for s, q in sorted_pairs]
    queries_sorted = [q for s, q in sorted_pairs]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color based on score quality
    colors = []
    for score in scores_sorted:
        if score >= 0.9:
            colors.append('#2ecc71')  # Green - excellent
        elif score >= 0.6:
            colors.append('#3498db')  # Blue - good
        elif score >= 0.5:
            colors.append('#f39c12')  # Orange - moderate
        else:
            colors.append('#e74c3c')  # Red - low

    # Create horizontal bar chart
    bars = ax.barh(queries_sorted, scores_sorted, color=colors, edgecolor='white', linewidth=0.5)

    # Add score labels on bars
    for bar, score in zip(bars, scores_sorted):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=10, fontweight='bold')

    # Add threshold lines
    ax.axvline(x=0.9, color='#2ecc71', linestyle='--', alpha=0.5, label='Excellent (≥0.9)')
    ax.axvline(x=0.6, color='#3498db', linestyle='--', alpha=0.5, label='Good (≥0.6)')
    ax.axvline(x=0.5, color='#f39c12', linestyle='--', alpha=0.5, label='Moderate (≥0.5)')

    # Styling
    ax.set_xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('RAG Retrieval Performance by Query', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.4)

    # Add legend
    ax.legend(loc='lower right', fontsize=9)

    # Add interpretation text
    fig.text(0.5, -0.02,
             'Higher scores indicate the retrieved documents are more semantically similar to the query.\n'
             'Specific queries ("RAG", "transformers") achieve better matches than broad queries ("machine learning").',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'retrieval_scores.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_dir / 'retrieval_scores.png'}")


def create_faithfulness_chart():
    """Create a simple faithfulness summary visualization."""

    fig, ax = plt.subplots(figsize=(8, 4))

    # Data
    topics = ['ML', 'Transformers', 'RAG', 'Embeddings', 'Deep Learning', 'Vector DB', 'NLP', 'LLMs']
    faithfulness = [96.6, 96.4, 96.6, 96.6, 96.6, 96.7, 96.2, 96.9]

    # Create bar chart
    colors = ['#27ae60'] * len(topics)  # All green since all are high
    bars = ax.bar(topics, faithfulness, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, faithfulness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val}%', ha='center', fontsize=9, fontweight='bold')

    # Styling
    ax.set_ylabel('Faithfulness Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('Answer Faithfulness by Topic', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(90, 100)

    # Add average line
    avg = np.mean(faithfulness)
    ax.axhline(y=avg, color='#e74c3c', linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')
    ax.legend(loc='lower right')

    # Rotate labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'images'
    plt.savefig(output_dir / 'faithfulness_scores.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_dir / 'faithfulness_scores.png'}")


if __name__ == "__main__":
    print("Generating RAG evaluation visualizations...")
    create_retrieval_scores_chart()
    create_faithfulness_chart()
    print("Done!")
