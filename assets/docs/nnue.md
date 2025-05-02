# NNUE (Efficiently Updatable Neural Network) Documentation

## Architecture Overview

### Network Structure

```cpp
struct NNUE_Architecture {
    // Input layer dimensions
    static constexpr int PieceFeatureDimensions = 768;
    static constexpr int TransformedFeatures = 256;

    // Network layers
    FeatureTransformer featureTransformer;
    HiddenLayer1 hidden1;
    HiddenLayer2 hidden2;
    OutputLayer output;
};
```

## Feature Transformer

### 1. Input Features

- King position relative features
- Piece position features
- Attack-defense patterns
- Material configuration

### 2. Feature Processing

```cpp
struct FeatureTransformer {
    // Weight dimensions
    static constexpr int InputDimensions = 64 * 11;
    static constexpr int OutputDimensions = 256;

    // Accumulator for incremental updates
    struct Accumulator {
        alignas(64) int16_t accumulation[2][OutputDimensions];
        bool computed[2];
    };
};
```

## Hidden Layers

### 1. First Hidden Layer

- Clipped ReLU activation
- Sparse matrix multiplication
- SIMD-optimized operations

### 2. Second Hidden Layer

- Further feature abstraction
- Pattern recognition
- Position evaluation refinement

## Incremental Updates

### 1. Accumulator Management

```cpp
void updateAccumulator(Position& pos, StateInfo& st) {
    if (!st.accumulator.computed[pos.sideToMove()]) {
        refreshAccumulator(pos, st);
    } else {
        incrementalUpdate(pos, st);
    }
}
```

### 2. Differential Updates

- Track changed features
- Update only affected neurons
- Maintain accuracy with efficiency

## Training Process

### 1. Data Preparation

- Position sampling
- Label generation
- Data augmentation

### 2. Training Pipeline

```cpp
struct TrainingPipeline {
    void prepare_batch(const std::vector<Position>& positions);
    void forward_pass();
    void backward_pass();
    void update_weights();
    void save_checkpoint();
};
```

### 3. Loss Functions

- Mean Squared Error
- Gradient scaling
- Learning rate scheduling

## Optimization Techniques

### 1. SIMD Operations

- AVX2/AVX512 support
- Vectorized matrix operations
- Optimized memory access

### 2. Quantization

- Weight quantization
- Activation quantization
- Inference optimization

### 3. Memory Management

- Cache alignment
- Memory pooling
- Efficient data structures

## Integration with Search

### 1. Evaluation Interface

```cpp
int evaluate_nnue(Position& pos) {
    if (pos.shouldSkipNNUE())
        return evaluate_classical(pos);

    return evaluate_with_nnue(pos);
}
```

### 2. Search Integration

- Position evaluation
- Move ordering hints
- Search extensions

## Performance Monitoring

### 1. Metrics

- Nodes per second
- Cache hit rate
- Memory usage
- Evaluation accuracy

### 2. Debugging Tools

- Network visualization
- Feature importance analysis
- Performance profiling

## Implementation Guidelines

1. **Setup**

   - Initialize network architecture
   - Load pretrained weights
   - Configure SIMD support

2. **Usage**

   - Position evaluation
   - Incremental updates
   - Cache management

3. **Maintenance**

   - Weight updates
   - Performance monitoring
   - Accuracy validation

4. **Optimization**
   - SIMD utilization
   - Memory alignment
   - Cache efficiency
