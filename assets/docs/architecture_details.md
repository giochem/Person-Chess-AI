# Detailed Architecture Specifications

## Complexity Progression (Basic to Advanced)

### Level 1: KnightVision (Basic)

- Classical chess programming approach
- Minimal dependencies
- Simple evaluation metrics
- Perfect for learning foundations

### Level 2: CNN (Intermediate)

- Neural network fundamentals
- Pattern recognition focus
- Modern ML techniques
- GPU acceleration benefits

### Level 3: NNUE (Advanced)

- Hybrid classical-neural approach
- Complex feature engineering
- Performance optimization required
- Hardware-specific tuning

### Level 4: Stockfish (Expert)

- State-of-the-art algorithms
- Multiple evaluation networks
- Advanced search techniques
- Production-grade codebase

## KnightVision Details

```python
class SearchConfig:
    MAX_DEPTH = 4  # Beginner-friendly depth
    QUIESCENCE_DEPTH = 2
    HASH_SIZE_MB = 16
    NUM_THREADS = 1  # Single-threaded for simplicity
```

### Key Features

- Simple alpha-beta search
- Basic position evaluation
- Material counting
- Piece-square tables
- Ideal learning implementation

## CNN Details

```python
class CNNConfig:
    BOARD_SIZE = 8
    PIECE_PLANES = 12
    FILTERS = 64
    TRAINING_POSITIONS = 1_000_000
```

### Key Features

- Modern deep learning approach
- Visual pattern recognition
- Dataset-based training
- GPU acceleration
- Scalable architecture

## NNUE Details

```cpp
struct NNUEConfig {
    static constexpr int FEATURE_DIMENSIONS = 768;
    static constexpr int HIDDEN_DIMENSIONS = 256;
    static constexpr int OUTPUT_DIMENSIONS = 1;
    static constexpr bool USE_HALF_KA = true;  // Advanced feature
```

### Key Features

- Efficiently updatable networks
- SIMD/AVX optimizations
- Incremental updates
- Feature engineering expertise

## Stockfish Details

```cpp
struct StockfishConfig {
    static constexpr int MAX_THREADS = 512;
    static constexpr int MAX_HASH_MB = 33554432;
    static constexpr int MAX_MULTIPV = 500;
    static constexpr bool USE_NNUE = true;
};
```

### Key Features

- Professional-grade search
- Advanced pruning techniques
- Tournament time management
- Parallel search algorithms

## Learning Path Recommendations

| Stage        | Focus Area        | Prerequisites      | Time Investment |
| ------------ | ----------------- | ------------------ | --------------- |
| KnightVision | Basic algorithms  | Programming basics | 1-2 weeks       |
| CNN          | Neural networks   | Python, ML basics  | 2-4 weeks       |
| NNUE         | Optimization      | C++, SIMD          | 1-2 months      |
| Stockfish    | Advanced concepts | All previous       | 3+ months       |

## Resource Requirements

| Engine       | CPU Cores | RAM   | Training | GPU      |
| ------------ | --------- | ----- | -------- | -------- |
| CNN          | 1-2       | 1GB   | Required | Optional |
| KnightVision | 1-4       | 2GB   | Optional | No       |
| NNUE         | 2-8       | 4GB   | Required | Optional |
| Stockfish    | 4-64      | 16GB+ | No       | No       |
