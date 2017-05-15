# Movie Recommendation System

## System Overview 
In the system, I implemented several algorithms to make recommendation for movies. 

## Implementation 
- User-Based Collaborative Filtering Algorithms
  * Cosine similarity method
  * Pearson Correlation method
  * Inverse user frequency
  * Case modification
  
- Item-Based Collaborative Filtering Algorithms
  * Adjusted cosine similarity method
  
- Ensemble Algorithm
  * 0.4 * user-based cosine similarity CF with case amplification(p = 2) +
    0.3 * user-based Pearson correlation CF with IUF +
    0.3 * item-based CF with adjusted cosine similarity
    
## License
Copyright 2017 Sen Zhang
