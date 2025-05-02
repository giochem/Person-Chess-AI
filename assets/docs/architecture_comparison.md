# Chess Engine Architecture Comparison

## Basic Level: KnightVision

### Location: `/src/chess_ai/knightvision/`

- Classical chess programming fundamentals
- Basic search with alpha-beta pruning
- Material + piece-square evaluation
- Perfect for learning foundations

**Complexity: ★☆☆☆☆**

```python
class KnightVision:
    def evaluate(self, position):
        material = self.evaluate_material()
        position = self.evaluate_piece_squares()
        mobility = self.evaluate_mobility()
        return material + position + mobility
```

## Intermediate Level: CNN Implementation

### Location: `/src/chess_ai/cnn/`

- Modern ML approach to chess
- Visual pattern recognition
- GPU-accelerated training
- Good for understanding neural networks

**Complexity: ★★☆☆☆**

```python
class BasicCNN:
    def __init__(self):
        self.model = Sequential([
            Conv2D(64, (3,3), input_shape=(8,8,12)),
            ReLU(),
            Conv2D(32, (3,3)),
            Flatten(),
            Dense(1, activation='tanh')
        ])
```

## Advanced Level: NNUE Implementation

### Location: `/src/chess_ai/nnue/`

- Efficiently Updatable Neural Network
- Incremental position updates
- HalfKP input features
- Optimized for CPU inference
- Complex architecture with proven strength

**Complexity: ★★★★☆**

```cpp
struct NNUEArchitecture {
    FeatureTransformer transformer; // 768->256 features
    Layer1 hidden1;   // 256->32 nodes
    Layer2 hidden2;   // 32->32 nodes
    LayerOutput out;  // 32->1 output
};
```

## Expert Level: Stockfish Integration

### Location: `/src/chess_ai/stockfish/`

- State-of-the-art search techniques
- Advanced pruning strategies
- Sophisticated evaluation function
- Multi-threaded search
- Tournament-proven strength

**Complexity: ★★★★★**

```cpp
class StockfishEngine {
    // Core components
    TranspositionTable tt;
    ThreadPool threads;
    NNUE::Network network;
    Search::SearchManager searchManager;
};
```

## Comparison Table

| Feature          | KnightVision | CNN      | NNUE  | Stockfish |
| ---------------- | ------------ | -------- | ----- | --------- |
| Search Depth     | 4-6          | 0        | 6-12  | 12+       |
| Nodes/second     | ~10K         | N/A      | ~100K | ~1M+      |
| Memory Usage     | Low          | Medium   | High  | Very High |
| Training Needed  | No           | Yes      | Yes   | No        |
| Implementation   | Easy         | Moderate | Hard  | Very Hard |
| Playing Strength | 500          | 1000     | 1200+ | 3000+     |

## Learning Path

1. Start with KnightVision for classical chess programming

   - Learn search algorithms
   - Understand basic evaluation
   - Implement simple pruning

2. Progress to CNN for ML concepts

   - Neural network basics
   - Pattern recognition
   - Training pipelines

3. Advance to NNUE for optimization

   - Feature engineering
   - Efficient updates
   - Performance tuning

4. Master Stockfish techniques
   - Advanced search
   - Multi-threading
   - Tournament features

## Skill Development Map

| Engine       | Core Skills             | Prerequisites      | Study Time |
| ------------ | ----------------------- | ------------------ | ---------- |
| KnightVision | Basic algorithms        | Programming basics | 2-4 weeks  |
| CNN          | Neural networks         | Python, PyTorch/TF | 1-2 months |
| NNUE         | Optimization techniques | C++, SIMD          | 2-3 months |
| Stockfish    | Professional techniques | All previous       | 3-6 months |
