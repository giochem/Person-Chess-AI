# Chess Position Evaluation Documentation

## Overview

The evaluation function assigns a numerical value to a chess position representing the relative advantage for each side.

## Static Evaluation Components

### 1. Material Evaluation

```cpp
int evaluateMaterial(Position& pos) {
    int score = 0;
    score += pos.count<PAWN>() * PawnValue;
    score += pos.count<KNIGHT>() * KnightValue;
    score += pos.count<BISHOP>() * BishopValue;
    score += pos.count<ROOK>() * RookValue;
    score += pos.count<QUEEN>() * QueenValue;
    return score;
}
```

### 2. Piece Square Tables

- Early game tables
- Late game tables
- Dynamic interpolation based on game phase

### 3. Pawn Structure Analysis

- Doubled pawns
- Isolated pawns
- Backward pawns
- Passed pawns
- Pawn chains
- Pawn shields

### 4. King Safety

- Pawn shield integrity
- Open files near king
- Attack patterns
- Piece concentration
- Castling rights

### 5. Mobility

- Available squares for pieces
- Control of center
- piece coordination
- Development

## Dynamic Evaluation

### 1. Tapered Evaluation

```cpp
Score taperedEval(Position& pos) {
    int gamePhase = calculateGamePhase(pos);
    Score middleGame = evaluateMiddlegame(pos);
    Score endGame = evaluateEndgame(pos);

    return interpolate(middleGame, endGame, gamePhase);
}
```

### 2. Pattern Recognition

- Piece formations
- Common tactical patterns
- Strategic patterns
- Endgame patterns

## NNUE Integration

### 1. Feature Transformer

- Input feature processing
- Piece-square combinations
- Half KA (King-Attacks) features

### 2. Network Architecture

- Input layer (transformed features)
- Hidden layers with clipped ReLU
- Output layer (single evaluation score)

## Evaluation Cache

### 1. Cache Structure

```cpp
struct EvalCacheEntry {
    uint64_t key;
    int16_t score;
    uint8_t generation;
    bool isValid;
};
```

### 2. Cache Management

- Size and memory allocation
- Replacement policies
- Entry invalidation
- Generation handling

## Performance Considerations

### 1. SIMD Optimizations

- Vectorized operations for PSQTs
- Parallel feature processing
- Optimized accumulator updates

### 2. Memory Access Patterns

- Cache-friendly data structures
- Aligned memory access
- Minimal pointer chasing

### 3. Incremental Updates

- Maintains evaluation components
- Updates only changed features
- Efficient accumulator management

## Testing Framework

### 1. Position Suite Testing

- FEN test positions
- Known evaluation cases
- Performance benchmarks

### 2. Tuning Framework

- SPSA tuning
- Texel tuning
- Parameter optimization

## Integration Guidelines

1. Initialize evaluation tables
2. Set up NNUE network
3. Configure caching
4. Implement incremental updates
5. Add evaluation debugging tools
